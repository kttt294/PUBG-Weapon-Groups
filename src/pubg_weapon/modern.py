"""TabM and TabICLv2 on the same match-complete cohorts as classical controls."""
import gc
from itertools import combinations
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, log_loss
from sklearn.naive_bayes import GaussianNB
from threadpoolctl import threadpool_limits
import torch
import xgboost as xgb

from .bootstrap import bootstrap_macro_f1, paired_difference
from .full_training import Encoding, scores
from .io import read_json, write_json
from .resources import measure


def predict_tabm(model, values, device, batch_rows=1024):
    model.eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(values), batch_rows):
            x = torch.as_tensor(values[start:start+batch_rows], device=device)
            chunks.append(model(x).softmax(dim=-1).mean(dim=1).cpu().numpy())
    return np.concatenate(chunks)


def fit_tabm(x, y, xv, yv, config, seed, learning_rate, path):
    from tabm import TabM
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    width = 256
    model = TabM.make(n_num_features=x.shape[1], d_out=len(np.unique(y)), k=16,
                      n_blocks=3, d_block=width, dropout=.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    best, stale, curves = -1., 0, []
    features = torch.as_tensor(x, device=device)
    labels = torch.as_tensor(y, device=device, dtype=torch.long)
    for epoch in range(config['tabm_epochs']):
        model.train()
        losses = 0.
        for start in range(0, len(x), config['tabm_batch_rows']):
            if start == 0:
                order = rng.permutation(len(x))
            index = torch.as_tensor(order[start:start+config['tabm_batch_rows']], device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features[index])
            targets = labels[index].unsqueeze(1).expand(-1, logits.shape[1]).reshape(-1)
            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets)
            loss.backward()
            optimizer.step()
            losses += float(loss.detach()) * len(index)
        probability = predict_tabm(model, xv, device)
        value = f1_score(yv, probability.argmax(axis=1), average='macro', zero_division=0)
        curves.append({'epoch': epoch+1, 'train_cross_entropy': losses / len(x),
                       'validation_log_loss': log_loss(yv, probability), 'validation_macro_f1': value})
        if value > best + 1e-5:
            best, stale = value, 0
            torch.save({key: tensor.detach().cpu() for key, tensor in model.state_dict().items()}, path)
        else:
            stale += 1
        if epoch % 10 == 0:
            print(f'TabM seed={seed} lr={learning_rate} epoch={epoch+1} val_macro_f1={value:.4f}', flush=True)
        if stale >= config['tabm_patience']:
            break
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    pd.DataFrame(curves).to_csv(path.with_suffix('.csv'), index=False)
    return model, device


def tabicl_model(config, seed):
    from tabicl import TabICLClassifier
    return TabICLClassifier(checkpoint_version=config['tabicl_checkpoint'], n_estimators=config['tabicl_estimators'],
                            batch_size=1, kv_cache='repr', device='cuda' if torch.cuda.is_available() else 'cpu',
                            offload_mode='cpu', use_amp=torch.cuda.is_available(), use_fa3=False,
                            random_state=seed, n_jobs=config['jobs'], verbose=False)


def predict_icl(model, x, batch_rows=512):
    return np.concatenate([model.predict_proba(x[start:start+batch_rows]) for start in range(0, len(x), batch_rows)])


def preflight(config, output):
    from tabm import TabM
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if config['require_gpu'] and not torch.cuda.is_available():
        raise RuntimeError('Select a Kaggle GPU accelerator before starting this study.')
    torch.set_num_threads(config['jobs'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TabM.make(n_num_features=4, d_out=3, k=4, d_block=32, n_blocks=2, dropout=.1).to(device)
    loss = model(torch.zeros(6, 4, device=device)).sum()
    loss.backward()
    del model, loss
    model = tabicl_model(config, 42)
    rng = np.random.default_rng(42)
    model.fit(rng.normal(size=(36, 4)), np.tile(np.arange(3), 12))
    probability = model.predict_proba(rng.normal(size=(3, 4)))
    if probability.shape != (3, 3) or not np.isfinite(probability).all():
        raise RuntimeError('TabICLv2 preflight failed')
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    from importlib.metadata import version
    write_json(output / 'preflight.json', {'device': device, 'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None,
               'torch': torch.__version__, 'tabm': version('tabm'), 'tabicl': version('tabicl'),
               'checkpoint': config['tabicl_checkpoint'], 'checks': 'TabM forward/backward and TabICLv2 fit/predict passed'})


def metric(y, probability, n_classes):
    if probability.shape != (len(y), n_classes) or not np.isfinite(probability).all():
        raise ValueError('Invalid probability matrix')
    probability = np.clip(probability, 0., None)
    probability /= probability.sum(axis=1, keepdims=True)
    order = np.argsort(-probability, axis=1, kind='stable')[:, :2]
    matrix = confusion_matrix(y, order[:, 0], labels=np.arange(n_classes))
    return scores(matrix, int((order == y[:, None]).any(axis=1).sum())), order[:, 0]


def run_modern(cohorts, output, config):
    cohorts, output = Path(cohorts), Path(output)
    output.mkdir(parents=True)
    all_metrics, all_val, differences, selections, timing = [], [], [], [], {}
    for seed in config['seeds']:
        print(f'Modern comparison seed={seed}', flush=True)
        frames = {s: pd.read_parquet(cohorts / f'{seed}_{s}.parquet') for s in ('train', 'validation', 'test')}
        labels = sorted(frames['train'].weapon_group.unique())
        encoder = Encoding(labels)
        raw = {s: encoder.encode(f) for s, f in frames.items()}
        encoder.scaler.fit(raw['train'][0])
        scaled = {s: encoder.scaler.transform(x) for s, (x, _) in raw.items()}
        x, y = raw['train']
        xv, yv = raw['validation']
        xt, yt = raw['test']
        xs, xvs, xts = (scaled[s] for s in ('train', 'validation', 'test'))
        folder = output / str(seed)
        folder.mkdir()
        models = {}
        val_metrics = {}
        fit_times, validation_times = {}, {}
        with threadpool_limits(limits=config['jobs']):
            controls = {
                'majority': DummyClassifier(strategy='most_frequent'),
                'naive_bayes': GaussianNB(),
                'logistic': LogisticRegression(max_iter=1000, C=1., random_state=seed),
                'random_forest': RandomForestClassifier(n_estimators=300, max_depth=18, min_samples_leaf=5,
                                                        n_jobs=config['jobs'], random_state=seed),
                'hist_gradient_boosting': HistGradientBoostingClassifier(max_iter=200, learning_rate=.1,
                                            early_stopping=False, random_state=seed),
            }
            for name, model in controls.items():
                started = time.perf_counter()
                with measure(timing, f'{seed}_{name}_fit', output):
                    model.fit(xs, y)
                fit_times[name] = time.perf_counter() - started
                started = time.perf_counter()
                val_metrics[name] = metric(yv, model.predict_proba(xvs), len(labels))[0]
                validation_times[name] = time.perf_counter() - started
                models[name] = model
            # The same benchmark rows and natural class distribution are used by every family.
            best_xgb, best_value = None, -1.
            for depth in (4, 6):
                started = time.perf_counter()
                with measure(timing, f'{seed}_xgboost_depth{depth}_fit', output):
                    fit = xgb.XGBClassifier(n_estimators=240, max_depth=depth, learning_rate=.1,
                          tree_method='hist', max_bin=64, n_jobs=config['jobs'], random_state=seed,
                          subsample=.85, colsample_bytree=.9,
                          early_stopping_rounds=25, eval_metric='mlogloss')
                    fit.fit(x, y, eval_set=[(xv, yv)], verbose=False)
                duration = time.perf_counter() - started
                values, _ = metric(yv, fit.predict_proba(xv), len(labels))
                selections.append({'seed': seed, 'family': 'xgboost', 'candidate': f'depth={depth}', **values})
                if values['macro_f1'] > best_value:
                    best_xgb, best_value = fit, values['macro_f1']
                    fit_times['xgboost'], val_metrics['xgboost'] = duration, values
            models['xgboost'] = best_xgb
            best_tabm, best_value = None, -1.
            for lr in (.001, .0003):
                name = f'tabm_lr{lr}'
                started = time.perf_counter()
                with measure(timing, f'{seed}_{name}_fit', output):
                    model, device = fit_tabm(xs, y, xvs, yv, config, seed, lr, folder / f'{name}.pt')
                duration = time.perf_counter() - started
                values, _ = metric(yv, predict_tabm(model, xvs, device), len(labels))
                selections.append({'seed': seed, 'family': 'tabm', 'candidate': f'lr={lr}', **values})
                if values['macro_f1'] > best_value:
                    best_tabm, best_value = {key: value.cpu().clone() for key, value in model.state_dict().items()}, values['macro_f1']
                    fit_times['tabm'], val_metrics['tabm'] = duration, values
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            from tabm import TabM
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = TabM.make(n_num_features=x.shape[1], d_out=len(labels), k=16, n_blocks=3, d_block=256, dropout=.1).to(device)
            model.load_state_dict(best_tabm)
            torch.save(best_tabm, folder / 'tabm_selected.pt')
            del model, best_tabm
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            icl = tabicl_model(config, seed)
            started = time.perf_counter()
            with measure(timing, f'{seed}_tabiclv2_fit', output):
                icl.fit(x, y)
            fit_times['tabiclv2'] = time.perf_counter() - started
            started = time.perf_counter()
            with measure(timing, f'{seed}_tabiclv2_validation', output):
                probability = predict_icl(icl, xv)
            validation_times['tabiclv2'] = time.perf_counter()-started
            val_metrics['tabiclv2'] = metric(yv, probability, len(labels))[0]
            write_json(folder / 'selection.json', {'model': max(val_metrics, key=lambda name: val_metrics[name]['macro_f1']),
                       'validation': val_metrics, 'candidates': [r for r in selections if r['seed'] == seed]})
            test_predictions = {}
            for name in [*models, 'tabm', 'tabiclv2']:
                started = time.perf_counter()
                with measure(timing, f'{seed}_{name}_test', output):
                    if name == 'tabm':
                        model = TabM.make(n_num_features=x.shape[1], d_out=len(labels), k=16,
                                          n_blocks=3, d_block=256, dropout=.1).to(device)
                        model.load_state_dict(torch.load(folder / 'tabm_selected.pt', map_location=device, weights_only=True))
                        probability = predict_tabm(model, xts, device)
                        del model
                    elif name == 'tabiclv2':
                        probability = predict_icl(icl, xt)
                    else:
                        probability = models[name].predict_proba(xt if name == 'xgboost' else xts)
                    values, prediction = metric(yt, probability, len(labels))
                duration = time.perf_counter()-started
                test_predictions[name] = prediction
                all_metrics.append({'seed': seed, 'name': name, **values, 'fit_seconds': fit_times[name],
                                    'predict_seconds': duration, 'train_rows': len(y), 'test_matches': frames['test'].match_id.nunique()})
                all_val.append({'seed': seed, 'name': name, **val_metrics[name]})
                pd.DataFrame(probability, columns=labels).to_parquet(folder / f'{name}_test_probabilities.parquet', index=False)
            draws = bootstrap_macro_f1(yt, test_predictions, frames['test'].match_id,
                                        np.arange(len(labels)), repeats=config['bootstrap_repeats'], seed=seed)
            draw_scores = {row['name']: row['macro_f1'] for row in all_metrics if row['seed'] == seed}
            for before, after in combinations(test_predictions, 2):
                differences.append({'seed': seed, **paired_difference(draws, draw_scores, before, after)})
            pd.DataFrame({'match_id': frames['test'].match_id.to_numpy(), 'actual': yt, **test_predictions}).to_parquet(folder / 'predictions.parquet', index=False)
            draws.to_csv(folder / 'bootstrap_macro_f1.csv', index=False)
            models['xgboost'].save_model(folder / 'xgboost.ubj')
            joblib.dump({'scaler': encoder.scaler, 'classes': labels, 'controls': {n: m for n, m in models.items() if n != 'xgboost'}}, folder / 'controls.joblib')
            for row in all_metrics:
                if row['seed'] == seed:
                    row['macro_f1_ci95_low'] = float(draws[row['name']].quantile(.025))
                    row['macro_f1_ci95_high'] = float(draws[row['name']].quantile(.975))
            del icl, models, frames
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        pd.DataFrame(all_metrics).to_csv(output / 'test_metrics.csv', index=False)
        pd.DataFrame(all_val).to_csv(output / 'validation_metrics.csv', index=False)
        pd.DataFrame(differences).to_csv(output / 'paired_differences.csv', index=False)
        pd.DataFrame(selections).to_csv(output / 'tuning.csv', index=False)
        write_json(output / 'resources.json', timing)
    results = pd.DataFrame(all_metrics)
    results[['seed', 'name', 'macro_f1', 'macro_f1_ci95_low', 'macro_f1_ci95_high',
             'accuracy', 'fit_seconds', 'predict_seconds']].to_csv(output / 'seed_summary.csv', index=False)
    write_json(output / 'protocol.json', {'cohorts': read_json(cohorts / 'manifest.json'),
               'methods': list(test_predictions), 'natural_class_distribution': True, 'class_weights': False,
               'selection': 'validation Macro-F1 within each seed', 'test': 'same cohort for all families within a seed',
               'pretrained_model': config['tabicl_checkpoint'], 'pretraining_cost_included': False,
               'tabm_variant': 'TabM without feature embeddings, k=16, 3 blocks, width 256',
               'uncertainty': 'One model seed (42). CI95 uses match-level bootstrap of fixed predictions; training-seed variation is not measured.',
               'checkpoint_download_in_preflight': True})
