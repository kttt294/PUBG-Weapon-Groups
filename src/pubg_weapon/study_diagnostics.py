"""Full-holdout error profiles and permutation importance after selection is locked."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

from .full_training import Encoding, Variant, batches, scores, scale
from .io import read_json
from .study_data import bounded_sample


def selected_predictor(prepared, results):
    manifest = read_json(prepared / 'dataset_manifest.json')
    selected = read_json(results / 'selection.json')['best']
    encoder = Encoding(manifest['classes'])
    fitted = joblib.load(results / 'models' / 'incremental.joblib')
    encoder.scaler = fitted['scaler']
    family = selected.removesuffix('_D')
    if family.startswith('xgboost'):
        model = xgb.Booster()
        model.load_model(results / 'models' / f'{family}.ubj')
    elif family == 'naive_bayes':
        model = fitted['naive_bayes']
    elif family == 'majority':
        model = fitted['majority']
    else:
        key = 'sgd_D' if family == 'sgd' else family
        model = fitted['sgd_candidates'][key]
    variant = Variant(family, 'D', model)

    def predict(frame):
        raw, y = encoder.encode(frame)
        scaled = scale(raw, encoder.scaler)
        probability = variant.predict(raw, scaled, encoder.width['D'], len(encoder.labels))
        return y, probability.argmax(axis=1)
    return encoder, predict


def error_profiles(prepared, results, output, jobs=2):
    prepared, results, output = Path(prepared), Path(results), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    encoder, predict = selected_predictor(prepared, results)
    k = len(encoder.labels)
    paths = sorted((prepared / 'test').glob('*.parquet'))
    cols = list(dict.fromkeys(['weapon_group', 'cause_for_audit', *encoder.columns]))
    matrices = {}
    role_matrix = np.zeros((3, 3), dtype=np.int64)
    role_mapping = np.array([{'AR': 1, 'DMR': 2, 'SR': 2}.get(label, 0) for label in encoder.labels])
    with threadpool_limits(limits=jobs):
        for frame in batches(paths, cols):
            y, prediction = predict(frame)
            distances = pd.cut(frame.kill_distance_m, [-np.inf, 50, 150, 300, np.inf],
                               labels=['0-50 m', '50-150 m', '150-300 m', '>300 m'])
            for category, values in [('distance', distances), ('map', frame['map']), ('cause', frame.cause_for_audit)]:
                for label, indices in values.groupby(values, observed=True).groups.items():
                    keep = frame.index.get_indexer(indices)
                    matrix = np.bincount(y[keep] * k + prediction[keep], minlength=k*k).reshape(k, k)
                    key = (category, str(label))
                    matrices[key] = matrices.get(key, np.zeros((k, k), dtype=np.int64)) + matrix
            role_matrix += np.bincount(role_mapping[y] * 3 + role_mapping[prediction], minlength=9).reshape(3, 3)
    records = [{'category': cat, 'group': label, **scores(matrix, 0)} for (cat, label), matrix in matrices.items()]
    frame = pd.DataFrame(records).drop(columns='top_2_accuracy')
    frame.to_csv(output / 'error_profiles.csv', index=False)
    pd.DataFrame([scores(role_matrix, 0)]).drop(columns='top_2_accuracy').to_csv(output / 'range_role_metrics.csv', index=False)
    pd.DataFrame(role_matrix, index=['Other_SMG_Shotgun', 'AR', 'DMR_SR'],
                 columns=['Other_SMG_Shotgun', 'AR', 'DMR_SR']).to_csv(output / 'range_role_confusion.csv')
    # This post-selection interpretation sample never changes feature sets or model choice.
    sample = bounded_sample(paths, ['weapon_group', *encoder.columns], 10_000)
    baseline_y, baseline_prediction = predict(sample)
    baseline = f1_score(baseline_y, baseline_prediction, labels=np.arange(k), average='macro', zero_division=0)
    rows = []
    with threadpool_limits(limits=jobs):
        for column in encoder.columns:
            values = sample[column].copy()
            differences = []
            for repeat in range(3):
                sample[column] = np.random.default_rng(42+repeat).permutation(values.to_numpy())
                y, prediction = predict(sample)
                value = f1_score(y, prediction, labels=np.arange(k), average='macro', zero_division=0)
                differences.append(baseline-value)
            sample[column] = values
            rows.append({'feature': column, 'macro_f1_drop_mean': float(np.mean(differences)),
                         'macro_f1_drop_std': float(np.std(differences, ddof=1)), 'sample_rows': len(sample), 'repeats': 3})
    pd.DataFrame(rows).sort_values('macro_f1_drop_mean', ascending=False).to_csv(output / 'permutation_importance.csv', index=False)


def resource_comparison(prepared, output):
    from .full_data import process_partitions
    from .resources import measure
    prepared, output = Path(prepared), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Feature Parquet lacks player identifiers, so this comparison measures read/encode work only.
    paths = sorted((prepared / 'train').glob('*.parquet'))[:4]
    jobs = [(p, read_json(prepared / 'dataset_manifest.json')['classes']) for p in paths]
    measurements, outcomes = {}, {}
    for workers in (1, 2):
        with measure(measurements, f'workers_{workers}', output):
            outcomes[workers] = list(process_partitions(encode_partition, paths, (jobs[0][1],), workers))
    if outcomes[1] != outcomes[2]:
        raise RuntimeError('Parallel encode verification mismatch')
    pd.DataFrame([{'workers': n, 'rows': sum(r['rows'] for r in outcomes[n]),
                   'rows_per_second': sum(r['rows'] for r in outcomes[n]) / measurements[f'workers_{n}']['wall_seconds'],
                   **measurements[f'workers_{n}']} for n in (1, 2)]).to_csv(output / 'worker_comparison.csv', index=False)


def encode_partition(args):
    import hashlib
    path, classes = args
    encoder = Encoding(classes)
    digest, count = hashlib.sha256(), 0
    for frame in batches([path], ['weapon_group', *encoder.columns]):
        x, y = encoder.encode(frame)
        digest.update(x.tobytes())
        digest.update(y.tobytes())
        count += len(y)
    return {'rows': count, 'sha256': digest.hexdigest()}
