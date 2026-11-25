"""Paired upper-bound sensitivity: canonical WERM+CV and WERM+P.

Both methods receive the same data; their conditional-count and upper-bound
prior weights remain distinct, as required by their definitions. Only the
penalty procedure receives a state-evolution overlay in this experiment.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'

from contextlib import redirect_stderr, redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
import pickle
import random
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

from tools import el_state_evolution as base
from tools import werm_cv_empirical as cv

N, P, SEED = 1000, 100, 20261120
SIGMA, GAMMA = .5, 6.
TRUE_FRACTIONS = (.25, .5, .75)
BOUNDS = tuple(range(1, 7))
PARENT = Path('results/E_L_SE_penalty_fixed/982fa979e869')


def generate_data(trial, true_fractions=TRUE_FRACTIONS):
    """Original E_L coefficient/design/noise draws with the requested locations."""
    rng = np.random.default_rng(np.random.SeedSequence([SEED, N, P, trial]))
    beta = np.empty((P, 4))
    beta[:, 0] = rng.normal(size=P)*np.sqrt(N/P)
    for k in range(1, 4):
        beta[:, k] = beta[:, k-1] + 1.5*np.sqrt(N/P)*rng.normal(size=P)
    X = rng.normal(size=(N, P))/np.sqrt(N)
    truth = np.rint(N*np.asarray(true_fractions)).astype(int)
    labels = np.searchsorted(truth, np.arange(N), side='right')
    mu = np.einsum('ij,ij->i', X, beta[:, labels].T)
    y = mu + rng.normal(0, SIGMA, N)
    return X, y, beta, truth, labels


def prepare(trials=100, theory_draws=20, true_fractions=TRUE_FRACTIONS,
            frozen_weights=None, delta_fraction=.1):
    if trials < 1 or theory_draws < 1:
        raise ValueError('Positive trial and theory-draw counts are required')
    true_fractions = tuple(float(v) for v in true_fractions)
    if (len(true_fractions) != 3 or
            not np.all(np.diff((0., *true_fractions, 1.)) > 0)):
        raise ValueError('Three strictly ordered change-point fractions in (0,1) are required')
    delta_fraction = float(delta_fraction)
    if not np.isfinite(delta_fraction) or not 0 < delta_fraction <= 1/max(BOUNDS):
        raise ValueError('The spacing fraction must admit all candidate segment counts')
    grid_size = 50
    full_gap = cv.minimum_gap(N, delta_fraction)
    fold_gap = cv.minimum_gap(N*9//10, delta_fraction)
    grid_gap = max(1, int(full_gap*grid_size/N))
    assert max(1, int(fold_gap*grid_size/(N*9//10))) == grid_gap
    use_parent_weights = delta_fraction == .1
    parent_bytes = (PARENT/'weights.pkl').read_bytes() if use_parent_weights else None
    source_names = [
        'tools/el_matched_selection.py', 'tools/el_state_evolution.py',
        'tools/werm_cv_empirical.py', 'tools/werm_cv_partition.py',
        'tools/m2_adjusted_experiment.py', 'WERM/config.py',
        'WERM/proximal.py', 'WERM/signal_strength.py', 'WERM/gof.py',
        'WERM/estimators.py', 'WERM/experiment_utils.py',
    ]
    spec = dict(schema=1, n=N, p=P, seed=SEED, trials=int(trials),
        theory_draws=int(theory_draws), true_fractions=true_fractions,
        upper_bounds=BOUNDS, gamma=GAMMA, noise_std=SIGMA,
        initial_coefficient_variance=N/P, jump_variance=1.5**2*N/P,
        folds=10, delta_fraction=delta_fraction, endpoint_minimum=1,
        coefficient_penalty='none', fit_loss='half squared', cv_score='absolute error',
        cv_selection='exhaustive; prefix of shared candidate scores at each upper bound',
        cv_prior='uniform admissible locations conditional on exactly K segments',
        penalty_prior='uniform K in 1..L, then uniform admissible locations',
        marginal_seed=0, marginal_grid_size=grid_size, marginal_grid_gap=grid_gap,
        frozen_full_weights=str(PARENT/'weights.pkl') if use_parent_weights else None,
        parent_weights_sha256=hashlib.sha256(parent_bytes).hexdigest() if use_parent_weights else None,
        theory='penalty only; independent proximal draws using oracle beta.T@beta/n',
        sources={name:hashlib.sha256(Path(name).read_bytes()).hexdigest()
                 for name in source_names})
    frozen_bytes = None
    if frozen_weights is not None:
        frozen_spec = json.loads(Path(frozen_weights).with_name('specification.json').read_text())
        for key in ('n', 'p', 'folds', 'delta_fraction', 'endpoint_minimum',
                    'marginal_seed', 'marginal_grid_size', 'marginal_grid_gap'):
            if frozen_spec[key] != spec[key]:
                raise ValueError(f'Frozen weights do not match requested {key}')
        frozen_bytes = Path(frozen_weights).read_bytes()
        spec['frozen_all_weights'] = str(frozen_weights)
        spec['all_weights_sha256'] = hashlib.sha256(frozen_bytes).hexdigest()
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]
    root = Path('results/E_L_matched_CV_penalty')/digest
    root.mkdir(parents=True, exist_ok=True)
    (root/'specification.json').write_text(json.dumps(spec, indent=2))
    weight_path = root/'weights.pkl'
    if weight_path.exists():
        if frozen_bytes is not None:
            assert hashlib.sha256(weight_path.read_bytes()).hexdigest() == spec['all_weights_sha256']
        weights = pickle.loads(weight_path.read_bytes())
    elif frozen_bytes is not None:
        weights = pickle.loads(frozen_bytes)
        weight_path.write_bytes(frozen_bytes)
    else:
        import torch
        from WERM.config import extrapolated_marginals
        parent = pickle.loads(parent_bytes) if use_parent_weights else None
        weights = dict(cv={}, penalty={})

        def native_weights(low, high, n_large, probabilities=None):
            np.random.seed(0); torch.manual_seed(0); random.seed(0)
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                return np.asarray(extrapolated_marginals(
                    low, high, grid_gap, n_small=grid_size, n_large=n_large,
                    p_l=probabilities, verbose=False))

        for count in BOUNDS:
            if use_parent_weights:
                weights['penalty'][count] = parent['penalty', count].copy()
                weights['cv'][N, count] = parent[
                    ('penalty', 1) if count == 1 else ('fixed', count)].copy()
            else:
                weights['penalty'][count] = native_weights(1, count, N, np.ones(count)/count)
                weights['cv'][N, count] = native_weights(count, count, N)
            weights['cv'][N*9//10, count] = native_weights(count, count, N*9//10)
        base.atomic_pickle(weight_path, weights)
    for (n, count), phi in weights['cv'].items():
        cv.get_weights(weights['cv'], n, count, cv.minimum_gap(n, delta_fraction))
    for count, phi in weights['penalty'].items():
        cv.get_weights({(N, count):phi}, N, count, full_gap)
    return root, spec, weights


def run_trial(trial, root, spec, weights):
    path = root/f'trial{trial:03d}.pkl'
    if path.exists():
        result = pickle.loads(path.read_bytes())
        assert result['trial'] == trial and result['spec_digest'] == root.name
        return result
    started = time.perf_counter()
    X, y, beta, truth, labels = generate_data(trial, spec['true_fractions'])
    data_digest = base.data_hash(X, y)
    # Neither algorithm receives beta, truth, or true regime labels.
    spacing = spec['delta_fraction']
    full_gap = cv.minimum_gap(N, spacing)
    cv_result = cv.run_cv(X, y, max(BOUNDS), weights['cv'], spacing, folds=10,
                          minimum_segment_fraction=None)
    result = dict(trial=trial, n=N, p=P, truth=truth, data_hash=data_digest,
                  spec_digest=root.name, cv=cv_result, fits={}, se_residual=0.)
    strength = beta.T@beta/N
    result['signal_strength'] = strength
    draws = spec['theory_draws']
    rng = np.random.default_rng(np.random.SeedSequence([SEED, trial, 917341]))
    Z = rng.normal(size=(draws, N, 4))@np.linalg.cholesky(strength).T
    response = Z[:, np.arange(N), labels]+rng.normal(0, SIGMA, (draws, N))
    standard_noise = rng.normal(size=(draws, N, max(BOUNDS)))
    for bound in BOUNDS:
        selected = cv_result['selected_by_bound'][bound]
        cv_fit = base.add_metrics(dict(cp=selected['cp'].copy(),
            count=selected['selected_segments']-1,
            segments=selected['selected_segments'],
            validation_mean=selected['validation_mean']), truth, N)
        phi = weights['penalty'][bound]
        theta = base.empirical_fit(X, y, phi)
        penalty_fit = base.add_metrics(base.search(theta, y, full_gap, True, GAMMA), truth, N)
        se = base.squared_se(phi, strength, labels, N/P, SIGMA**2)
        theory = []
        for draw in range(draws):
            V = Z[draw]@se['lam']+standard_noise[draw, :, :bound]@se['covariance_root'].T
            predicted = (1-se['omega'])*V+se['omega']*response[draw, :, None]
            theory.append(base.add_metrics(
                base.search(predicted, response[draw], full_gap, True, GAMMA), truth, N))
        result['fits'][bound] = dict(cv=cv_fit, penalty=penalty_fit, theory=theory,
            se={key:value for key,value in se.items() if key not in ('omega','covariance_root')})
        result['se_residual'] = max(result['se_residual'], se['b_residual'],
                                    se['bias_residual'], se['covariance_residual'])
        for fit in [cv_fit, penalty_fit, *theory]:
            assert fit['count'] == len(fit['cp']) and 0 <= fit['count'] < bound
            assert np.all((fit['cp'] > 0) & (fit['cp'] < N))
            assert np.all(np.diff(fit['cp']) >= full_gap)
    # Confirm that fitting/search did not mutate the shared paired data.
    assert base.data_hash(X, y) == data_digest
    assert result['se_residual'] < 1e-9
    result['seconds'] = time.perf_counter()-started
    base.atomic_pickle(path, result)
    return result


def summary(payload):
    rows = []
    for bound in BOUNDS:
        for source, label in [('cv','WERM+CV'), ('penalty','WERM+P'), ('theory','Theory')]:
            selected = [r['fits'][bound][source] for r in payload['records']]
            fits = [f for draws in selected for f in draws] if source == 'theory' else selected
            counts = np.array([f['count'] for f in fits])
            haus = np.array([f['haus'] for f in fits])
            finite = haus[np.isfinite(haus)]
            rows.append(dict(L=bound, method=label, trials=len(selected), draws=len(fits),
                mean_count=float(counts.mean()), correct_count=float(np.mean(counts == 3)),
                empty=int(np.sum(counts == 0)),
                over_count=float(np.mean(counts > 3)), under_count=float(np.mean(counts < 3)),
                mean_haus=float(finite.mean()) if len(finite) else np.nan,
                median_haus=float(np.median(finite)) if len(finite) else np.nan))
    return pd.DataFrame(rows)


def run(root, spec, weights, jobs=12):
    with parallel_config(backend='loky', inner_max_num_threads=1):
        records = Parallel(n_jobs=jobs, verbose=10)(
            delayed(run_trial)(trial, root, spec, weights) for trial in range(spec['trials']))
    assert len({r['data_hash'] for r in records}) == spec['trials']
    payload = dict(specification=spec, records=records)
    base.atomic_pickle(root/'study.pkl', payload)
    summary(payload).to_csv(root/'summary.csv', index=False)
    return payload


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs', type=int, default=12)
    args = parser.parse_args()
    root, spec, weights = prepare()
    print('Results:', root, flush=True)
    start = time.perf_counter()
    payload = run(root, spec, weights, jobs=args.jobs)
    print(summary(payload).to_string(index=False), flush=True)
    print('Finished in', time.perf_counter()-start, 'seconds.', flush=True)
