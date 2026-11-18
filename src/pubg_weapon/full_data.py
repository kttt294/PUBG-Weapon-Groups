"""Prepare every event file in bounded-memory, match-complete partitions."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from functools import lru_cache
import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import MIN_CLASS_FRACTION, ROOT, SEED
from .io import new_directory, read_chunks, write_json
from .preprocessing import OTHER, build_features, clean_events, load_mapping, merge_rare
from .resources import measure
from .splits import source_partition

SPLITS = ('train', 'validation', 'test')


def save_frame(frame, path, writers, stack):
    table = pa.Table.from_pandas(frame, preserve_index=False)
    if path not in writers:
        writers[path] = stack.enter_context(pq.ParquetWriter(path, table.schema, compression='snappy'))
    writers[path].write_table(table)


def partition_sources(sources, output, partitions, chunk_size):
    match_ids, previous_source_ids, files, causes = set(), set(), [], Counter()
    with ExitStack() as stack:
        writers = {}
        for source in sources:
            rows = 0
            first_source = source.name == 'kill_match_stats_final_0.csv'
            for frame in read_chunks(source, chunk_size):
                rows += len(frame)
                ids = frame.match_id.str.strip()
                distinct = set(ids.dropna().unique()) - {''}
                match_ids.update(distinct)
                if first_source:
                    previous_source_ids.update(distinct)
                causes.update(frame.killed_by.str.strip().fillna('<missing>').value_counts().to_dict())
                buckets = pd.util.hash_pandas_object(ids, index=False).to_numpy() % partitions
                for bucket, part in frame.groupby(buckets, sort=False):
                    save_frame(part, output / f'{bucket:04d}.parquet', writers, stack)
            files.append({'name': source.name, 'bytes': source.stat().st_size, 'rows': rows})
            print(f'Đọc xong {source.name}: {rows:,} dòng', flush=True)
    return match_ids, previous_source_ids, files, causes


def configure_worker():
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)


def process_partitions(worker, paths, shared_args, workers):
    calls = ((path, *shared_args) for path in paths)
    if workers == 1:
        for args in calls:
            yield worker(args)
    else:
        with ProcessPoolExecutor(workers, mp_context=get_context('spawn'), initializer=configure_worker) as pool:
            yield from pool.map(worker, calls)


@lru_cache(maxsize=1)
def load_assignments(path):
    return pd.read_csv(path, dtype={'match_id': 'string'}).set_index('match_id').split


def clean_partition(args):
    path, cleaned, mapping, assignment_file = args
    events = pd.read_parquet(path)
    frame, audit = clean_events(events, mapping, allow_empty=True)
    del events
    counts, matches = {}, {}
    if not frame.empty:
        frame['split'] = frame.match_id.map(load_assignments(assignment_file))
        if frame['split'].isna().any():
            raise RuntimeError('A cleaned event has no match split')
        for name, part in frame.groupby('split', sort=False):
            counts[name] = part.weapon_group.value_counts().to_dict()
            matches[name] = part.match_id.nunique()
        frame.to_parquet(cleaned / path.name, index=False)
    path.unlink()
    return audit, counts, matches


def clean_partitions(raw, cleaned, mapping, assignment_file, workers):
    audit_totals = None
    removed_causes = {}
    raw_counts = {split: Counter() for split in SPLITS}
    match_counts = Counter()
    largest = 0
    for audit, per_split, matches in process_partitions(clean_partition, raw, (cleaned, mapping, assignment_file), workers):
        largest = max(largest, int(audit.before.iloc[0]))
        counts = audit.set_index('rule')[['before', 'removed', 'remaining']]
        audit_totals = counts if audit_totals is None else audit_totals.add(counts)
        for row in audit.itertuples():
            removed_causes.setdefault(row.rule, Counter()).update(json.loads(row.removed_causes))
        for split, counts in per_split.items():
            raw_counts[split].update(counts)
        match_counts.update(matches)
    audit_totals['removed_causes'] = [json.dumps(removed_causes[rule], ensure_ascii=False, sort_keys=True)
                                     for rule in audit_totals.index]
    audit_totals['fraction_of_input'] = audit_totals.removed / max(1, audit_totals.loc['input', 'before'])
    audit_totals.to_csv(cleaned.parent / 'cleaning_counts.csv')
    return raw_counts, match_counts, int(largest)


def feature_partition(args):
    path, output, rare = args
    frame = pd.read_parquet(path)
    phase_counts = Counter()
    for split, part in frame.groupby('split', sort=False):
        part = part.copy()
        part['weapon_group'] = merge_rare(part.weapon_group, rare)
        features = build_features(part)
        if split == 'train':
            phase_counts.update(features.time_phase.value_counts().to_dict())
        features.to_parquet(output / split / path.name, index=False)
    path.unlink()
    return phase_counts


def prepare_full(sources, output, mapping_path=None, partitions=128, chunk_size=400_000, workers=1):
    if min(partitions, chunk_size, workers) < 1:
        raise ValueError('Partitions, chunk size and workers must be positive')
    sources = [Path(source).resolve() for source in sources]
    if not sources or len(set(sources)) != len(sources) or not all(path.is_file() for path in sources):
        raise ValueError('Supply distinct existing event CSV files')
    output = new_directory(output)
    raw = new_directory(output / 'raw_parts')
    cleaned = new_directory(output / 'cleaned_parts')
    resources = {}
    with measure(resources, 'scan', output):
        match_ids, previous_ids, files, causes = partition_sources(sources, raw, partitions, chunk_size)
    if any(file['rows'] == 0 for file in files):
        raise ValueError('One or more event files are empty')

    mapping = load_mapping(mapping_path or ROOT / 'configs' / 'label_mapping.json')
    unknown_counts = {key: value for key, value in causes.items() if key not in mapping and key != '<missing>'}
    mapping.update(dict.fromkeys(unknown_counts))
    pd.DataFrame(sorted(causes.items()), columns=['killed_by', 'count']).to_csv(output / 'source_weapon_counts.csv', index=False)
    pd.DataFrame(sorted(unknown_counts.items()), columns=['killed_by', 'count']).to_csv(output / 'unmapped_causes.csv', index=False)
    print(f'Nhãn chưa ánh xạ: {unknown_counts}', flush=True)

    # Reconstruct the cohort whose test results were already used in the old report.
    previous_sample = np.random.default_rng(SEED).choice(sorted(previous_ids), 50_000, replace=False) if len(previous_ids) >= 50_000 else ()
    development = pd.read_csv(ROOT / 'configs' / 'development_matches.csv', dtype={'match_id': 'string'})
    excluded_from_test = set(development.match_id) | set(previous_sample)
    assignments = source_partition(match_ids, excluded_from_test).set_index('match_id').split
    assignment_file = output / 'source_match_splits.csv'
    assignments.to_csv(assignment_file)
    available = len(match_ids)
    excluded_count = len(match_ids & excluded_from_test)
    del assignments, match_ids, development, previous_ids, previous_sample, excluded_from_test

    raw_paths = sorted(raw.glob('*.parquet'))
    with measure(resources, 'clean', output):
        raw_counts, match_counts, largest = clean_partitions(raw_paths, cleaned, mapping, assignment_file, workers)
    raw.rmdir()

    train_total = sum(raw_counts['train'].values())
    if not train_total:
        raise ValueError('No training events remain')
    rare = sorted(label for label, count in raw_counts['train'].items()
                  if label != OTHER and count / train_total < MIN_CLASS_FRACTION)
    effective_counts = {}
    for split, counts in raw_counts.items():
        effective = Counter()
        for label, count in counts.items():
            effective[OTHER if label in rare else label] += count
        effective_counts[split] = effective
    classes = sorted(effective_counts['train'])
    if len(classes) < 2 or any(not counts or set(counts) - set(classes) for counts in effective_counts.values()):
        raise ValueError('Validation and test must be nonempty with labels known to train')

    for split in SPLITS:
        (output / split).mkdir()
    phase_counts = Counter()
    with measure(resources, 'features', output):
        for counts in process_partitions(feature_partition, sorted(cleaned.glob('*.parquet')),
                                         (output, rare), workers):
            phase_counts.update(counts)
    cleaned.rmdir()
    checks = {split: {
        'rows': int(sum(effective_counts[split].values())),
        'matches': match_counts[split],
        'class_counts': dict(effective_counts[split]),
    } for split in SPLITS}
    pd.DataFrame(sorted(phase_counts.items()), columns=['time_phase', 'count']).to_csv(output / 'train_phase_counts.csv', index=False)
    manifest = {
        'files': files, 'source_rows': sum(item['rows'] for item in files),
        'available_matches': available, 'classes': classes, 'rare_groups_merged': rare,
        'seed': SEED, 'partitions': partitions, 'largest_raw_partition_rows': largest,
        'previous_matches_ineligible_for_test': excluded_count,
        'unmapped_cause_rows': sum(unknown_counts.values()), 'max_train_rows': None,
        'workers': workers, 'chunk_rows': chunk_size, 'resources': resources,
        'stage_seconds': {stage: values['wall_seconds'] for stage, values in resources.items()},
        'splits': checks,
    }
    write_json(output / 'dataset_manifest.json', manifest)
    return manifest
