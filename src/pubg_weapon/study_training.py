"""Full-data models with validation-only selection and saved learning curves."""
from collections import Counter
from copy import deepcopy
import gc
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
from sklearn.naive_bayes import GaussianNB
from threadpoolctl import threadpool_limits

from .config import SEED
from .full_training import Encoding, FEATURE_LEVELS, Variant, ParquetTrainIterator, batches, evaluate, scale, write_test, sgd_probabilities
from .io import new_directory, read_json, write_json
from .resources import measure
from .study_data import bounded_sample


def run_study_training(prepared, output, config):
    prepared, output = Path(prepared), new_directory(output)
    seed = config['seed']
    manifest = read_json(prepared / 'dataset_manifest.json')
    paths = {s: sorted((prepared / s).glob('*.parquet')) for s in ('train', 'validation', 'test')}
    for split, files in paths.items():
        if sum(pq.ParquetFile(p).metadata.num_rows for p in files) != manifest['splits'][split]['rows']:
            raise RuntimeError(f'Incomplete {split} data')
    models = new_directory(output / 'models')
    encoder = Encoding(manifest['classes'])
    columns = ['weapon_group', *encoder.columns]
    k = len(encoder.labels)
    class_ids = np.arange(k)
    counts = manifest['splits']['train']['class_counts']
    balance = np.array([sum(counts.values()) / (k * counts[c]) for c in encoder.labels], dtype=np.float32)
    majority = max(range(k), key=lambda i: counts[encoder.labels[i]])
    stages, curves = {}, []
    with threadpool_limits(limits=config['jobs']):
        seen = Counter()
        with measure(stages, 'scaler', output):
            for frame in batches(paths['train'], columns):
                x, y = encoder.encode(frame)
                encoder.scaler.partial_fit(x)
                seen.update(dict(enumerate(np.bincount(y, minlength=k))))
        if any(seen[i] != counts[label] for i, label in enumerate(encoder.labels)):
            raise RuntimeError('Incomplete training labels')
        with measure(stages, 'learning_curve_samples', output):
            val_frame = bounded_sample(paths['validation'], columns, 100_000)
            train_frame = bounded_sample(paths['train'], columns, 20_000)
        xv, yv = encoder.encode(val_frame)
        xt, yt = encoder.encode(train_frame)
        xvs, xts = scale(xv, encoder.scaler), scale(xt, encoder.scaler)
        # All comparison choices are locked before test is scored.
        candidates = {f'sgd_alpha{a:g}': (SGDClassifier(loss='log_loss', alpha=a, average=True, random_state=seed), 'D')
                      for a in (1e-5, 1e-3)}
        candidates.update({f'sgd_{level}': (SGDClassifier(loss='log_loss', alpha=1e-4, average=True, random_state=seed), level)
                           for level in FEATURE_LEVELS})
        nb = GaussianNB()
        best, best_score, stale = {}, dict.fromkeys(candidates, -1.), dict.fromkeys(candidates, 0)
        rng = np.random.default_rng(seed)
        with measure(stages, 'incremental_models', output):
            for epoch in range(config['sgd_epochs']):
                active = {name for name in candidates if stale[name] < config['sgd_patience']}
                if not active:
                    break
                for frame in batches([paths['train'][i] for i in rng.permutation(len(paths['train']))], columns):
                    raw, y = encoder.encode(frame)
                    features = scale(raw, encoder.scaler)
                    if epoch == 0:
                        nb.partial_fit(features, y, classes=class_ids)
                    for name in sorted(active):
                        model, level = candidates[name]
                        model.partial_fit(features[:, :encoder.width[level]], y, classes=class_ids, sample_weight=balance[y])
                for name in sorted(active):
                    model, level = candidates[name]
                    width = encoder.width[level]
                    vp = sgd_probabilities(model, xvs[:, :width])
                    from sklearn.metrics import f1_score
                    value = f1_score(yv, vp.argmax(axis=1), labels=class_ids, average='macro', zero_division=0)
                    curves.append({'model': name, 'epoch': epoch+1, 'train_rows_per_epoch': sum(counts.values()),
                                   'train_sample_log_loss': log_loss(yt, sgd_probabilities(model, xts[:, :width]), labels=class_ids),
                                   'validation_sample_log_loss': log_loss(yv, vp, labels=class_ids),
                                   'validation_sample_macro_f1': value})
                    if value > best_score[name] + 1e-5:
                        best[name], best_score[name], stale[name] = deepcopy(model), value, 0
                    else:
                        stale[name] += 1
                pd.DataFrame(curves).to_csv(output / 'sgd_learning_curves.csv', index=False)
                print(f'SGD epoch {epoch+1}: {sum(counts.values()):,} training rows', flush=True)
        variants = [Variant('majority', 'D', majority), Variant('naive_bayes', 'D', nb)]
        variants.extend(Variant(name, level, best[name]) for name, (_, level) in candidates.items() if name not in {f'sgd_{l}' for l in FEATURE_LEVELS})
        variants.extend(Variant('sgd', level, best[f'sgd_{level}']) for level in FEATURE_LEVELS)
        joblib.dump({'classes': encoder.labels.tolist(), 'scaler': encoder.scaler, 'majority': majority,
                     'naive_bayes': nb, 'sgd_candidates': best}, models / 'incremental.joblib')
        xgb_configs = [(4, 1.), (6, 1.), (6, 0.)]
        booster_curves = []
        for depth, weight_power in xgb_configs:
            family = f'xgboost_depth{depth}_weight{weight_power:g}'
            cache = new_directory(models / f'{family}_cache')
            with measure(stages, family, output):
                iterator = ParquetTrainIterator(paths['train'], encoder, balance**weight_power, cache / 'train')
                matrix = xgb.ExtMemQuantileDMatrix(iterator, nthread=config['jobs'], max_bin=64)
                if matrix.num_row() != sum(counts.values()):
                    raise RuntimeError('XGBoost training row mismatch')
                validation_matrix = xgb.DMatrix(xv, label=yv, nthread=config['jobs'])
                training_sample = xgb.DMatrix(xt, label=yt, nthread=config['jobs'])
                history = {}
                booster = xgb.train({'objective': 'multi:softprob', 'num_class': k, 'tree_method': 'hist',
                                     'device': 'cpu', 'nthread': config['jobs'], 'max_bin': 64,
                                     'max_depth': depth, 'eta': .10, 'seed': seed, 'subsample': .85,
                                     'colsample_bytree': .9, 'eval_metric': 'mlogloss'},
                                    matrix, num_boost_round=config['xgb_rounds'],
                                    evals=[(training_sample, 'train_sample'), (validation_matrix, 'validation_sample')],
                                    early_stopping_rounds=config['xgb_patience'], evals_result=history, verbose_eval=25)
                booster = booster[:booster.best_iteration+1]
                booster.save_model(models / f'{family}.ubj')
                for epoch, value in enumerate(history['validation_sample']['mlogloss'], 1):
                    booster_curves.append({'model': family, 'round': epoch,
                                           'train_sample_log_loss': history['train_sample']['mlogloss'][epoch-1],
                                           'validation_sample_log_loss': value})
                del matrix, iterator, validation_matrix, training_sample
                gc.collect()
            for path in cache.iterdir():
                path.unlink()
            cache.rmdir()
            variants.append(Variant(family, 'D', booster))
        pd.DataFrame(booster_curves).to_csv(output / 'xgboost_learning_curves.csv', index=False)
        with measure(stages, 'validation', output):
            val, _, _, _, _ = evaluate(paths['validation'], variants, encoder)
        if not val.rows.eq(manifest['splits']['validation']['rows']).all():
            raise RuntimeError('Incomplete validation evaluation')
        val.to_csv(output / 'validation_metrics.csv', index=False)
        selected = val[val.feature_set.eq(FEATURE_LEVELS['D'])].sort_values(['macro_f1', 'name'], ascending=[False, True]).iloc[0]['name']
        write_json(output / 'selection.json', {'best': selected, 'main_feature_set': FEATURE_LEVELS['D'],
                   'ablation': {l: f'sgd_{l}' for l in FEATURE_LEVELS},
                   'epoch_rule': 'best Macro-F1 on a fixed 100000-row validation sample, then full validation model selection',
                   'holdout_policy': 'Same fixed holdout as earlier experiments, reused for the prespecified comparison.'})
        test_output = new_directory(output / 'test')
        assignment = pd.read_csv(prepared / 'source_match_splits.csv', dtype={'match_id': 'string'})
        groups = pd.Index(assignment.loc[assignment.split.eq('test'), 'match_id'])
        with measure(stages, 'test', output):
            write_test(test_output, paths['test'], variants, encoder, selected, groups,
                       manifest['splits']['test']['rows'], config['bootstrap_repeats'])
    write_json(output / 'run_metadata.json', {'config': config, 'resources': stages, 'jobs': config['jobs'],
               'train_rows_seen': sum(counts.values()), 'classes': encoder.labels.tolist(),
               'xgboost_version': xgb.__version__, 'selected': selected})
    return selected
