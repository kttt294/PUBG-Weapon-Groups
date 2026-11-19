"""Group-complete comparison cohorts and train-only summaries from full partitions."""
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .config import RAW_COLUMNS
from .full_training import Encoding, batches
from .io import file_sha256, read_json, write_json


def bounded_sample(paths, columns, limit, seed=42):
    rng = np.random.default_rng(seed)
    result = None
    priority = np.empty(0)
    for frame in batches(paths, columns):
        keys = rng.random(len(frame))
        result = frame if result is None else pd.concat([result, frame], ignore_index=True)
        priority = np.r_[priority, keys]
        if len(priority) > limit:
            keep = np.argpartition(priority, limit - 1)[:limit]
            result = result.iloc[keep].reset_index(drop=True)
            priority = priority[keep]
    return result


def raw_quality(frame):
    result = []
    for col in RAW_COLUMNS:
        values = frame[col]
        missing = values.isna()
        blank = values.astype('string').str.strip().eq('').fillna(False)
        is_number = col not in ('killed_by', 'killer_name', 'victim_name', 'map', 'match_id')
        invalid_number = infinite = 0
        if is_number:
            # Audit numeric fields only; names and identifiers remain text.
            numeric = pd.to_numeric(values.to_numpy(dtype=object, na_value=np.nan), errors='coerce')
            invalid_number = int((pd.isna(numeric) & ~missing.to_numpy()).sum())
            infinite = int(np.isinf(numeric).sum())
        result.append({'column': col, 'rows': len(frame), 'missing': int(missing.sum()),
                       'blank': int(blank.sum()), 'invalid_number': invalid_number,
                       'infinite': infinite})
    return pd.DataFrame(result).set_index('column')


def benchmark_cohorts(prepared, output, config):
    prepared, output = Path(prepared), Path(output)
    output.mkdir(parents=True)
    assignments = pd.read_csv(prepared / 'source_match_splits.csv', dtype={'match_id': 'string'})
    chosen, metadata = {}, []
    for seed in config['seeds']:
        rng = np.random.default_rng(42)
        chosen[seed] = {}
        for split, count in config['benchmark_matches'].items():
            available = sorted(assignments.loc[assignments.split.eq(split), 'match_id'])
            chosen[seed][split] = set(rng.choice(available, min(count, len(available)), replace=False))
    for split in config['benchmark_matches']:
        combined = set.union(*(chosen[seed][split] for seed in config['seeds']))
        chunks = []
        paths = sorted((prepared / split).glob('*.parquet'))
        columns = ['match_id', 'weapon_group', *Encoding(read_json(prepared / 'dataset_manifest.json')['classes']).columns]
        for frame in batches(paths, columns):
            part = frame.loc[frame.match_id.isin(combined)]
            if not part.empty:
                chunks.append(part)
        collected = pd.concat(chunks, ignore_index=True)
        for seed in config['seeds']:
            frame = collected.loc[collected.match_id.isin(chosen[seed][split])].copy()
            # Drop complete matches in fixed random order to bound ICL context size.
            sizes = frame.groupby('match_id', sort=True).size()
            order = np.random.default_rng(42).permutation(sizes.index)
            cumulative = sizes.loc[order].cumsum()
            keep = set(cumulative.index[cumulative <= config['benchmark_max_rows']])
            frame = frame.loc[frame.match_id.isin(keep)].sort_values(['match_id', 'time'], kind='stable')
            if len(frame) == 0 or frame.match_id.nunique() < 2:
                raise ValueError(f'Insufficient benchmark matches for {seed}/{split}')
            path = output / f'{seed}_{split}.parquet'
            frame.to_parquet(path, index=False)
            metadata.append({'seed': seed, 'split': split, 'rows': len(frame), 'matches': frame.match_id.nunique(),
                             'sha256': file_sha256(path), 'class_counts': frame.weapon_group.value_counts().to_dict()})
    for seed in config['seeds']:
        frames = {split: pd.read_parquet(output / f'{seed}_{split}.parquet') for split in config['benchmark_matches']}
        groups = [set(frame.match_id) for frame in frames.values()]
        if any(groups[i] & groups[j] for i, j in ((0, 1), (0, 2), (1, 2))):
            raise RuntimeError('Overlapping benchmark matches')
        train_classes = set(frames['train'].weapon_group)
        if any(set(frame.weapon_group) != train_classes for frame in frames.values()):
            raise RuntimeError('A comparison cohort is missing target classes. Increase match counts.')
    write_json(output / 'manifest.json', {'cohorts': metadata, 'sampling': 'whole matches within fixed source splits',
                                         'seeds': config['seeds'], 'data_seed': 42,
                                         'test_policy': 'Same cohorts across model seeds. Fixed holdout reused from earlier run; no test-based tuning.'})
    return output


def summarize_train(prepared, output):
    prepared, output = Path(prepared), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = read_json(prepared / 'dataset_manifest.json')
    encoder = Encoding(manifest['classes'])
    numeric = [col for col in encoder.columns if col not in encoder.vocab]
    accum = {col: {'rows': 0, 'sum': 0., 'sum_sq': 0., 'min': float('inf'), 'max': -float('inf')} for col in numeric}
    edges = np.r_[np.arange(0., 1001.), np.arange(1010., 12011., 10.)]
    hist = {label: np.zeros(len(edges)-1, dtype=np.int64) for label in manifest['classes']}
    maps, phases = Counter(), Counter()
    for frame in batches(sorted((prepared / 'train').glob('*.parquet')), ['weapon_group', *encoder.columns]):
        for col in numeric:
            arr = frame[col].to_numpy(dtype=np.float64)
            row = accum[col]
            row['rows'] += len(arr)
            row['sum'] += float(arr.sum())
            row['sum_sq'] += float(arr @ arr)
            row['min'] = min(row['min'], float(arr.min()))
            row['max'] = max(row['max'], float(arr.max()))
        for label, part in frame.groupby('weapon_group', observed=True):
            hist[label] += np.histogram(part.kill_distance_m, bins=edges)[0]
        maps.update(frame['map'].value_counts().to_dict())
        phases.update(frame.time_phase.value_counts().to_dict())
    stats = []
    for col, row in accum.items():
        mean = row['sum'] / row['rows']
        stats.append({'feature': col, 'count': row['rows'], 'mean': mean,
                      'std': np.sqrt(max(0., row['sum_sq'] / row['rows'] - mean**2)),
                      'min': row['min'], 'max': row['max']})
    pd.DataFrame(stats).to_csv(output / 'train_numeric_summary.csv', index=False)
    pd.DataFrame({'lower_m': edges[:-1], 'upper_m': edges[1:], **hist}).to_csv(output / 'distance_histogram.csv', index=False)
    quantiles = []
    for label, count in hist.items():
        cumulative = count.cumsum()
        indices = np.minimum(np.searchsorted(cumulative, cumulative[-1] * np.array([.25, .5, .75])), len(count)-1)
        quantiles.append({'class': label, 'count': int(cumulative[-1]),
                          **{name: float(edges[index]) for name, index in zip(('q25_lower_m', 'median_lower_m', 'q75_lower_m'), indices)}})
    pd.DataFrame(quantiles).to_csv(output / 'train_distance_quantiles.csv', index=False)
    write_json(output / 'train_distributions.json', {'map': dict(maps), 'time_phase': dict(phases),
               'quantile_method': 'lower edge of histogram bin, width 1 m up to 1000 m and 10 m thereafter'})
