"""Fit on all training rows and score validation/test in bounded batches."""
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .bootstrap import bootstrap_from_counts, paired_difference
from .config import FEATURE_SETS, SEED
from .io import new_directory, read_json, write_json
from .resources import measure

BATCH_ROWS = 250_000
FEATURE_LEVELS = dict(zip('ABCD', ('distance', 'context', 'full', 'loadout_history')))


class Encoding:
    def __init__(self, classes):
        self.labels = pd.Index(classes)
        vocab = {
            'map': ('ERANGEL', 'MIRAMAR'),
            'time_phase': ('early', 'mid', 'late'),
            'last_weapon_group': ('NONE', *classes),
            'dominant_prior_weapon': ('NONE', *classes),
        }
        # Arrange columns so every ablation level is a prefix of the full input.
        self.columns = tuple(dict.fromkeys(col for name in FEATURE_LEVELS.values()
                                          for col in FEATURE_SETS[name]))
        self.vocab = {col: pd.Index(values) for col, values in vocab.items()}
        self.width = {level: sum(len(vocab[col]) if col in vocab else 1 for col in FEATURE_SETS[name])
                      for level, name in FEATURE_LEVELS.items()}
        self.scaler = StandardScaler()

    def encode(self, frame, level='D'):
        result = np.zeros((len(frame), self.width[level]), dtype=np.float32)
        pos = 0
        active = set(FEATURE_SETS[FEATURE_LEVELS[level]])
        for col in (column for column in self.columns if column in active):
            if col in self.vocab:
                vocab = self.vocab[col]
                indices = vocab.get_indexer(frame[col])
                if (indices < 0).any():
                    raise ValueError(f'Unexpected category in {col}')
                result[np.arange(len(frame)), pos + indices] = 1
                pos += len(vocab)
            else:
                result[:, pos] = frame[col].to_numpy(dtype=np.float32)
                pos += 1
        y = self.labels.get_indexer(frame.weapon_group).astype(np.int32)
        if (y < 0).any() or not np.isfinite(result).all():
            raise ValueError('Invalid feature value or unknown target')
        return result, y


def batches(paths, columns, size=BATCH_ROWS):
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=size, columns=columns, use_threads=False):
            yield batch.to_pandas()


class ParquetTrainIterator(xgb.DataIter):
    def __init__(self, paths, encoder, weights, cache_prefix):
        self.paths = paths
        self.encoder = encoder
        self.weights = weights
        super().__init__(cache_prefix=str(cache_prefix), release_data=True)

    def reset(self):
        self.frames = iter(batches(self.paths, ['weapon_group', *self.encoder.columns]))

    def next(self, input_data):
        frame = next(self.frames, None)
        if frame is None:
            return False
        features, labels = self.encoder.encode(frame)
        input_data(data=features, label=labels, weight=self.weights[labels])
        return True


@dataclass
class Variant:
    family: str
    level: str
    model: object

    @property
    def name(self):
        return f'{self.family}_{self.level}'

    def predict(self, raw, scaled, encoded_width, n_classes):
        if self.family == 'majority':
            probability = np.zeros((len(raw), n_classes), dtype=np.float32)
            probability[:, self.model] = 1
            return probability
        if self.family == 'xgboost':
            return self.model.inplace_predict(raw)
        return self.model.predict_proba(scaled[:, :encoded_width])


def scores(matrix, top_two):
    actual, predicted = matrix.sum(axis=1), matrix.sum(axis=0)
    correct = np.diag(matrix)
    denominator = actual + predicted
    f1 = np.divide(2 * correct, denominator, out=np.zeros_like(correct, dtype=float), where=denominator != 0)
    count = int(actual.sum())
    return {
        'macro_f1': float(f1.mean()) if count else None,
        'accuracy': float(correct.sum() / count) if count else None,
        'weighted_f1': float(f1 @ actual / count) if count else None,
        'top_2_accuracy': top_two / count if count else None,
        'rows': count,
    }


def scale(features, scaler):
    return scaler.transform(features, copy=True)


def evaluate(paths, variants, encoder, groups=None):
    k = len(encoder.labels)
    conf = np.zeros((len(variants), 3, k, k), dtype=np.int64)
    top2 = np.zeros((len(variants), 3), dtype=np.int64)
    support = None if groups is None else np.zeros((len(groups), k), dtype=np.int32)
    clustered = {} if groups is None else {
        variant.name: (np.zeros_like(support), np.zeros_like(support)) for variant in variants}
    columns = ['weapon_group', *encoder.columns]
    if groups is not None:
        columns.append('match_id')
    for frame in batches(paths, columns):
        raw, y = encoder.encode(frame)
        scaled = scale(raw, encoder.scaler)
        subgroup = frame.has_prior_kill.to_numpy(dtype=bool)
        if groups is not None:
            match_index = groups.get_indexer(frame.match_id)
            if (match_index < 0).any():
                raise ValueError('Unknown test match')
            np.add.at(support, (match_index, y), 1)
        for i, variant in enumerate(variants):
            probability = variant.predict(raw, scaled, encoder.width[variant.level], k)
            order = np.argsort(-probability, axis=1, kind='stable')[:, :2]
            labels = order[:, 0]
            hit = (order == y[:, None]).any(axis=1)
            pair = y * k + labels
            conf[i, 0] += np.bincount(pair, minlength=k*k).reshape(k, k)
            top2[i, 0] += int(hit.sum())
            for j in (0, 1):
                keep = subgroup == j
                conf[i, j+1] += np.bincount(pair[keep], minlength=k*k).reshape(k, k)
                top2[i, j+1] += int(hit[keep].sum())
            if groups is not None:
                pred_count, correct_count = clustered[variant.name]
                np.add.at(pred_count, (match_index, labels), 1)
                ok = labels == y
                np.add.at(correct_count, (match_index[ok], y[ok]), 1)
    rows, subset = [], []
    for i, variant in enumerate(variants):
        rows.append({'name': variant.name, 'family': variant.family, 'feature_set': FEATURE_LEVELS[variant.level],
                     **scores(conf[i, 0], int(top2[i, 0]))})
        for j in (0, 1):
            subset.append({'name': variant.name, 'has_prior_kill': j, **scores(conf[i, j+1], int(top2[i, j+1]))})
    return pd.DataFrame(rows), pd.DataFrame(subset), conf[:, 0], support, clustered


def write_test(output, paths, variants, encoder, selected, groups, test_rows, repeats):
    test, subset, matrices, support, clustered = evaluate(paths, variants, encoder, groups)
    if not test.rows.eq(test_rows).all():
        raise RuntimeError('Test row counts do not match the data manifest')
    observed = support.sum(axis=1) > 0
    draws = bootstrap_from_counts(support[observed], {
        name: tuple(block[observed] for block in blocks)
        for name, blocks in clustered.items()
    }, repeats=repeats)
    test['macro_f1_ci95_low'] = test.name.map(draws.quantile(.025))
    test['macro_f1_ci95_high'] = test.name.map(draws.quantile(.975))
    test[test.feature_set.eq(FEATURE_LEVELS['D'])].to_csv(output / 'test_metrics.csv', index=False)
    subset.to_csv(output / 'history_subset_metrics.csv', index=False)
    draws.to_csv(output / 'bootstrap_macro_f1.csv', index=False)
    ablation = test[test.family.eq('sgd')].copy()
    ablation.insert(1, 'ablation', list(FEATURE_LEVELS))
    ablation.to_csv(output / 'ablation.csv', index=False)
    values = test.set_index('name').macro_f1.to_dict()
    pd.DataFrame([
        {'ablation_change': f'{a}->{b}', **paired_difference(draws, values, f'sgd_{a}', f'sgd_{b}')}
        for a, b in zip('ABC', 'BCD')
    ]).to_csv(output / 'ablation_gains.csv', index=False)

    k = len(encoder.labels)
    per_class = []
    for variant, matrix in zip(variants, matrices):
        for i, label in enumerate(encoder.labels):
            tp, n, m = int(matrix[i, i]), int(matrix[i].sum()), int(matrix[:, i].sum())
            per_class.append({'name': variant.name, 'class': label, 'support': n,
                              'precision': tp / m if m else 0, 'recall': tp / n if n else 0,
                              'f1-score': 2 * tp / (n + m) if n + m else 0})
    pd.DataFrame(per_class).to_csv(output / 'per_class_metrics.csv', index=False)
    matrix = matrices[next(i for i, variant in enumerate(variants) if variant.name == selected)]
    labels = encoder.labels.tolist()
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(output / 'confusion_matrix.csv')
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(matrix.reshape(k, k), display_labels=labels).plot(ax=ax, colorbar=False)
    fig.tight_layout()
    fig.savefig(output / 'confusion_matrix.png', dpi=130)
    plt.close(fig)
    write_json(output / 'evaluation_summary.json', {
        'best_selected_on_validation': selected, 'test_rows': test_rows,
        'bootstrap': {'unit': 'match_id', 'repeats': repeats, 'matches': int(observed.sum())},
        'best_metrics': test.set_index('name').loc[selected].to_dict(),
    })


def run_full_training(prepared, output, jobs=2, xgb_rounds=60, sgd_epochs=2, bootstrap_repeats=1000):
    if min(jobs, xgb_rounds, sgd_epochs) < 1 or bootstrap_repeats < 2:
        raise ValueError('Invalid training configuration')
    prepared, output = Path(prepared), Path(output)
    manifest = read_json(prepared / 'dataset_manifest.json')
    paths = {split: sorted((prepared / split).glob('*.parquet')) for split in ('train', 'validation', 'test')}
    expected_rows = {split: manifest['splits'][split]['rows'] for split in paths}
    for split, files in paths.items():
        if not files or sum(pq.ParquetFile(path).metadata.num_rows for path in files) != expected_rows[split]:
            raise ValueError(f'Incomplete {split} partitions')

    output = new_directory(output)
    model_dir = new_directory(output / 'models')
    encoder = Encoding(manifest['classes'])
    columns = ['weapon_group', *encoder.columns]
    k = len(encoder.labels)
    class_ids = np.arange(k)
    class_counts = manifest['splits']['train']['class_counts']
    balance = np.array([sum(class_counts.values()) / (k * class_counts[label]) for label in encoder.labels], dtype=np.float32)
    majority = max(range(k), key=lambda i: class_counts[encoder.labels[i]])
    stages = {}
    with threadpool_limits(limits=jobs):
        seen = Counter()
        with measure(stages, 'scaler', output):
            for frame in batches(paths['train'], columns):
                features, y = encoder.encode(frame)
                encoder.scaler.partial_fit(features)
                seen.update(dict(enumerate(np.bincount(y, minlength=k))))
        if any(seen[i] != class_counts[label] for i, label in enumerate(encoder.labels)):
            raise RuntimeError('Training label counts do not match the data manifest')

        nb = GaussianNB()
        sgd = {level: SGDClassifier(loss='log_loss', alpha=1e-4, random_state=SEED)
               for level in FEATURE_LEVELS}
        rng = np.random.default_rng(SEED)
        with measure(stages, 'incremental_models', output):
            for epoch in range(sgd_epochs):
                order = rng.permutation(len(paths['train']))
                for frame in batches([paths['train'][i] for i in order], columns):
                    features, y = encoder.encode(frame)
                    features = scale(features, encoder.scaler)
                    if epoch == 0:
                        nb.partial_fit(features, y, classes=class_ids)
                    for level, model in sgd.items():
                        model.partial_fit(features[:, :encoder.width[level]], y,
                                          classes=class_ids, sample_weight=balance[y])
                print(f'SGD: xong lượt {epoch+1}/{sgd_epochs} trên {expected_rows["train"]:,} dòng', flush=True)

        cache = new_directory(model_dir / 'xgb_cache')
        with measure(stages, 'xgboost', output):
            iterator = ParquetTrainIterator(paths['train'], encoder, balance, cache / 'train')
            matrix = xgb.ExtMemQuantileDMatrix(iterator, nthread=jobs, max_bin=64)
            if matrix.num_row() != expected_rows['train']:
                raise RuntimeError('XGBoost did not load the entire training split')
            booster = xgb.train({
                'objective': 'multi:softprob', 'num_class': k, 'tree_method': 'hist', 'device': 'cpu',
                'nthread': jobs, 'max_bin': 64, 'max_depth': 5, 'eta': .20, 'seed': SEED,
            }, matrix, num_boost_round=xgb_rounds)
            del matrix, iterator
        for path in cache.iterdir():
            path.unlink()
        cache.rmdir()
        booster.save_model(model_dir / 'xgboost.ubj')
    variants = [Variant('majority', 'D', majority), Variant('naive_bayes', 'D', nb),
                *(Variant('sgd', level, model) for level, model in sgd.items()),
                Variant('xgboost', 'D', booster)]
    # Store sklearn objects and the encoder state without depending on this module's class pickling.
    joblib.dump({'classes': encoder.labels.tolist(), 'scaler': encoder.scaler,
                 'majority': majority, 'naive_bayes': nb, 'sgd': sgd}, model_dir / 'incremental.joblib')
    with threadpool_limits(limits=jobs), measure(stages, 'validation', output):
        val, _, _, _, _ = evaluate(paths['validation'], variants, encoder)
    if not val.rows.eq(expected_rows['validation']).all():
        raise RuntimeError('Validation row counts do not match the data manifest')
    selected = val[val.feature_set.eq(FEATURE_LEVELS['D'])].sort_values(
        ['macro_f1', 'name'], ascending=[False, True]).iloc[0]['name']
    val.to_csv(output / 'validation_metrics.csv', index=False)
    write_json(output / 'selection.json', {'best': selected, 'main_feature_set': FEATURE_LEVELS['D'],
                                          'ablation': {level: f'sgd_{level}' for level in FEATURE_LEVELS}})
    print(f'Mô hình được chọn trên validation: {selected}', flush=True)
    assignment = pd.read_csv(prepared / 'source_match_splits.csv', dtype={'match_id': 'string'})
    groups = pd.Index(assignment.loc[assignment.split.eq('test'), 'match_id'])
    test_output = new_directory(output / 'test')
    with threadpool_limits(limits=jobs), measure(stages, 'test', output):
        write_test(test_output, paths['test'], variants, encoder, selected, groups,
                   expected_rows['test'], bootstrap_repeats)
    write_json(output / 'run_metadata.json', {'jobs': jobs, 'xgb_rounds': xgb_rounds, 'sgd_epochs': sgd_epochs,
               'resources': stages, 'stage_seconds': {name: val['wall_seconds'] for name, val in stages.items()},
               'train_rows_seen': expected_rows['train'], 'xgboost_version': xgb.__version__})
    return selected
