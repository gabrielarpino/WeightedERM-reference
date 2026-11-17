"""Paired M2 comparisons with the paper's empirical theta adjustment.

This experiment is isolated from the existing estimators and saved M2 runs.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')

import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed, parallel_config
from WERM.config import extrapolated_marginals
from WERM.experiment_utils import data_hash, evaluate_changepoints, trial_rng
from WERM.proximal import fit_weighted
from WERM.signal_priors import SparseDiffSignal
from WERM.signal_strength import empirical_adjustment

DELTAS = (1.5, 3.5, 7.5, 11.5)
METHODS = ('WERM', 'WERM adjusted')
FRACTIONS = {'unknown': (0.2, 0.5), 'known': (0.25, 0.45, 0.7)}


def spacing(n):
    """Original M2 CV's fold-local spacing, also used for known-count fits."""
    return max(int(np.ceil(n / 50)), int(n / 40), 1)


def squared_partition(theta, y, gap):
    """Exact fixed-count DP, with the same feasible set/ties as chgpt_gof_dp.

    Additive segment costs allow a prefix minimum instead of the quadratic
    predecessor loop. End segments have no minimum length beyond one row;
    adjacent internal boundaries must be at least gap apart.
    """
    theta, y = np.asarray(theta), np.asarray(y)
    n, count = theta.shape
    if y.shape != (n,) or count < 1 or n < count or gap < 1:
        raise ValueError('Invalid partition dimensions')
    losses = (theta - y[:, None]) ** 2
    if not np.isfinite(losses).all():
        raise ValueError('Nonfinite partition loss')
    prefix = np.vstack((np.zeros(count), np.cumsum(losses, axis=0)))
    if count == 1:
        return np.empty(0, dtype=int)
    previous = np.full(n, np.inf)
    previous[1:] = prefix[1:n, 0]
    back = np.full((count, n), -1, dtype=int)
    for segments in range(2, count):
        candidates = previous - prefix[:n, segments - 1]
        candidates[0] = np.inf
        running = np.minimum.accumulate(candidates)
        improves = candidates < np.r_[np.inf, running[:-1]]
        first_argmin = np.maximum.accumulate(np.where(improves, np.arange(n), -1))
        current = np.full(n, np.inf)
        ends = np.arange(gap + 1, n)
        current[ends] = prefix[ends, segments - 1] + running[ends - gap]
        back[segments, ends] = first_argmin[ends - gap]
        previous = current
    final = previous[1:] + prefix[n, count - 1] - prefix[1:n, count - 1]
    last = int(np.argmin(final)) + 1
    if not np.isfinite(final[last - 1]):
        raise ValueError('Infeasible fixed count')
    boundaries = [last]
    for segments in range(count - 1, 1, -1):
        last = int(back[segments, last])
        boundaries.append(last)
    return np.array(boundaries[::-1], dtype=int)


def coefficient_fit(X, y, weights):
    result = fit_weighted(X, y, weights, loss='least_squares', loss_scale=2.,
                          penalty='none', penalty_strength=0.)
    if not result.converged:
        raise RuntimeError('Least-squares fit failed KKT tolerance')
    return result.coef


def fit_candidate(X, y, count, adjusted, weights):
    """No ground-truth inputs; adjustment uses only this fit's own responses."""
    n, p = X.shape
    phi = weights[n, count]
    B = np.column_stack([coefficient_fit(X, y, w / w.sum()) for w in phi])
    theta = X @ B
    b = np.empty(0)
    residual = 0.
    if adjusted:
        theta, b = empirical_adjustment(theta, y, phi, n / p, loss='squared')
        residual = float(np.max(np.abs(np.mean(1 / (1 + 2*b[:, None]*phi), axis=1)
                                        - (1 - p/n))))
    cp = squared_partition(theta, y, spacing(n))
    return dict(cp=cp, b=b, b_residual=residual)


def validation_score(X, y, train, test, training_cp):
    """Original M2: map training boundaries, hard refit, absolute held-out loss."""
    mapped = train[np.asarray(training_cp, dtype=int)]
    boundaries = np.r_[0, mapped, len(y)]
    coefficients = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        rows = train[(train >= start) & (train < end)]
        coefficients.append(coefficient_fit(X[rows], y[rows], np.ones(len(rows)))
                             if len(rows) else np.zeros(X.shape[1]))
    segment = np.searchsorted(boundaries[1:], test, side='right')
    prediction = np.einsum('ij,ij->i', X[test], np.asarray(coefficients)[segment])
    return float(np.abs(prediction - y[test]).sum())


def select_count(X, y, adjusted, weights, folds=10):
    indices = np.arange(len(y))
    scores = np.zeros((4, folds))
    b_residual = 0.
    failures = []
    for count in range(1, 5):
        for fold in range(folds):
            test = indices[fold::folds]
            train = indices[indices % folds != fold]
            try:
                fitted = fit_candidate(X[train], y[train], count, adjusted, weights)
                b_residual = max(b_residual, fitted['b_residual'])
                scores[count-1, fold] = validation_score(X, y, train, test, fitted['cp'])
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                scores[count-1, fold] = np.inf
                failures.append(dict(count=count, fold=fold, error=str(error)))
    totals = scores.sum(axis=1)
    if not np.isfinite(totals).any():
        raise RuntimeError(f'All CV candidates failed: {failures}')
    best = int(np.argmin(totals)) + 1
    fitted = fit_candidate(X, y, best, adjusted, weights)
    fitted.update(selected_segments=best, cv_scores=scores,
                  cv_failures=failures,
                  b_residual=max(b_residual, fitted['b_residual']))
    return fitted


def generate_data(mode, setting, trial, p=100):
    delta = DELTAS[setting]
    # Unknown-count data use the same task seed scheme as the final M2 notebook.
    rng = trial_rng('M2c-alpha-0.3' if mode == 'unknown' else 'M2-adjusted-known', setting, trial)
    n = int(p * delta)
    prior = SparseDiffSignal(8, np.sqrt(400 * delta), .3, 4)
    B = prior.sample(p, rng=rng)
    X = rng.normal(0, np.sqrt(1/n), (n, p))
    noise = rng.normal(size=n) * np.sqrt(.1)
    truth = (n * np.asarray(FRACTIONS[mode])).astype(int)
    labels = np.searchsorted(truth, np.arange(n), side='right')
    # Retain the original M2 multiplication/scaling order.
    theta = (X * np.sqrt(n)) @ (B / np.sqrt(n))
    y = theta[np.arange(n), labels] + noise
    return X, y, truth


def atomic_pickle(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_bytes(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
    temporary.replace(path)


def prepare(root=Path('results/M2_adjusted_comparison'), p=100, trials=100):
    """Freeze shared native marginal interpolation and checkpoint provenance."""
    sources = ['tools/m2_adjusted_experiment.py', 'WERM/config.py', 'WERM/proximal.py',
               'WERM/signal_strength.py', 'WERM/signal_priors.py', 'WERM/experiment_utils.py']
    spec = dict(schema=1, p=p, trials=trials, deltas=DELTAS, fractions=FRACTIONS,
                folds=10, Lmax=4, marginal_seed=0, minimum_gap='max(ceil(n/50),floor(n/40),1)',
                sources={s: hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in sources})
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]
    directory = Path(root) / digest
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'specification.json').write_text(json.dumps(spec, indent=2))
    weight_file = directory / 'weights.pkl'
    if weight_file.exists():
        weights = pickle.loads(weight_file.read_bytes())
    else:
        import torch
        np.random.seed(0)
        torch.manual_seed(0)
        weights = {}
        sizes = sorted({size for delta in DELTAS for size in (int(p*delta), int(p*delta)*9//10)})
        for n in sizes:
            for count in range(1, 5):
                # Same n_small=50 interpolator and fixed-count prior as M2.
                phi = np.asarray(extrapolated_marginals(count, count, int(spacing(n)*50/n),
                                 n_small=50, n_large=n, p_l=None, verbose=False))
                if phi.shape != (count, n) or np.any(phi < 0) or not np.isfinite(phi).all():
                    raise ValueError('Invalid native marginal weights')
                np.testing.assert_allclose(phi.sum(axis=0), 1., atol=1e-8)
                weights[n, count] = phi
        atomic_pickle(weight_file, weights)
    return directory, spec, weights


def run_trial(mode, setting, trial, directory, spec, weights):
    path = directory / f'{mode}_delta{DELTAS[setting]:g}_trial{trial:03d}.pkl'
    if path.exists():
        return pickle.loads(path.read_bytes())
    X, y, truth = generate_data(mode, setting, trial, spec['p'])
    record = dict(mode=mode, setting=setting, trial=trial, n=len(y), truth=truth,
                  data_hash=data_hash(X, y), methods={})
    for adjusted, name in enumerate(METHODS):
        started = time.perf_counter()
        try:
            fitted = (select_count(X, y, bool(adjusted), weights) if mode == 'unknown'
                      else fit_candidate(X, y, 4, bool(adjusted), weights))
            fitted.setdefault('selected_segments', 4)
            fitted.setdefault('cv_failures', [])
            fitted.update(evaluate_changepoints(fitted['cp'], truth, len(y)))
            fitted['error'] = None
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            fitted = dict(cp=None, status='failed', haus=np.nan, size=np.nan, error=repr(error))
        fitted['runtime'] = time.perf_counter() - started
        record['methods'][name] = fitted
    atomic_pickle(path, record)
    return record


def run_experiment(mode, directory, spec, weights, jobs=12):
    tasks = [(setting, trial) for setting in range(len(DELTAS)) for trial in range(spec['trials'])]
    with parallel_config(backend='loky', inner_max_num_threads=1):
        records = Parallel(n_jobs=jobs, verbose=10)(
            delayed(run_trial)(mode, setting, trial, directory, spec, weights)
            for setting, trial in tasks)
    payload = dict(spec=spec, records=records, mode=mode)
    atomic_pickle(directory / f'{mode}.pkl', payload)
    return payload


def summarize(payload):
    import pandas as pd
    rows = []
    for setting, delta in enumerate(DELTAS):
        records = [r for r in payload['records'] if r['setting'] == setting]
        true_count = len(FRACTIONS[payload['mode']])
        for name in METHODS:
            fits = [r['methods'][name] for r in records]
            valid = [f for f in fits if f['error'] is None]
            nonempty = [f['haus'] for f in valid if f['size'] > 0]
            rows.append(dict(delta=delta, method=name, attempted=len(fits), valid=len(valid),
                failed=len(fits)-len(valid), empty=sum(f['size']==0 for f in valid),
                mean_haus=float(np.mean(nonempty)) if nonempty else np.nan,
                median_haus=float(np.median(nonempty)) if nonempty else np.nan,
                mean_count=float(np.mean([f['size'] for f in valid])) if valid else np.nan,
                correct_count=sum(f['size']==true_count for f in valid)/len(valid) if valid else np.nan,
                max_b_residual=max((f['b_residual'] for f in valid), default=np.nan),
                cv_failed_folds=sum(len(f['cv_failures']) for f in valid)))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=100)
    parser.add_argument('--jobs', type=int, default=12)
    args = parser.parse_args()
    directory, spec, weights = prepare(trials=args.trials)
    print('Results:', directory, flush=True)
    for mode in ('unknown', 'known'):
        result = run_experiment(mode, directory, spec, weights, args.jobs)
        table = summarize(result)
        table.to_csv(directory / f'{mode}_summary.csv', index=False)
        print(table.to_string(index=False), flush=True)
