"""Matched CV/penalty companion at n=1500, with independent theory for both.

The empirical design preserves native internal-gap constraints. Every candidate
and every endpoint refit is audited; unsupported local ratios are never hidden
by excluding trials or candidates.
"""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'

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
from tools import werm_cv_empirical as empirical
from tools import werm_cv_theory as theory

N, P, SEED = 1500, 100, 20261120
SIGMA, GAMMA = .5, 6.
TRUE_FRACTIONS = (.25, .5, .75)
BOUNDS = tuple(range(1, 7))


def generate_data(trial):
    rng = np.random.default_rng(np.random.SeedSequence([SEED, N, P, trial]))
    beta = np.empty((P, 4))
    beta[:, 0] = rng.normal(size=P)*np.sqrt(N/P)
    for k in range(1, 4):
        beta[:, k] = beta[:, k-1]+1.5*np.sqrt(N/P)*rng.normal(size=P)
    X = rng.normal(size=(N, P))/np.sqrt(N)
    truth = np.rint(N*np.asarray(TRUE_FRACTIONS)).astype(int)
    labels = np.searchsorted(truth, np.arange(N), side='right')
    mu = np.einsum('ij,ij->i', X, beta[:, labels].T)
    y = mu+rng.normal(0, SIGMA, N)
    return X, y, beta, truth, labels


def prepare(trials=100, theory_draws=20):
    if trials < 1 or theory_draws < 1:
        raise ValueError('Positive trial and theory-draw counts are required')
    sources = ['tools/el_matched_cv_theory.py', 'tools/el_state_evolution.py',
        'tools/werm_cv_empirical.py', 'tools/werm_cv_theory.py',
        'tools/werm_cv_partition.py', 'tools/m2_adjusted_experiment.py',
        'WERM/config.py', 'WERM/proximal.py', 'WERM/signal_strength.py',
        'WERM/gof.py', 'WERM/estimators.py', 'WERM/experiment_utils.py']
    spec = dict(schema=1, n=N, p=P, seed=SEED, trials=int(trials),
        theory_draws=int(theory_draws), true_fractions=TRUE_FRACTIONS,
        upper_bounds=BOUNDS, noise_std=SIGMA, gamma=GAMMA,
        initial_coefficient_variance=N/P, jump_variance=1.5**2*N/P,
        folds=10, delta_fraction=.1, endpoint_minimum=1,
        minimum_segment_fraction=None, coefficient_penalty='none',
        fit_loss='half squared', cv_score='absolute error',
        cv_selection='exhaustive; prefixes of candidate risk vectors',
        cv_prior='uniform admissible locations conditional on exactly K segments',
        penalty_prior='uniform K in 1..L, then uniform admissible locations',
        marginal_seed=0, marginal_grid_size=50, marginal_grid_gap=5,
        cv_theory='first-order composition: expected-loss partition, uniform-refit SE, '
                  'absolute held-out risk, own count selection, full-data proximal SE',
        penalty_theory='independent proximal SE draws followed by penalized search',
        companion_to='results/E_L_matched_CV_penalty/3af038cb5e34',
        sources={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in sources})
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]
    root = Path('results/E_L_matched_CV_penalty_both_theory')/digest
    root.mkdir(parents=True, exist_ok=True)
    (root/'specification.json').write_text(json.dumps(spec, indent=2))
    path = root/'weights.pkl'
    if path.exists():
        weights = pickle.loads(path.read_bytes())
    else:
        import torch
        from WERM.config import extrapolated_marginals
        weights = dict(cv={}, penalty={})
        for count in BOUNDS:
            for n, upper in ((N, False), (N*9//10, False), (N, True)):
                np.random.seed(0); torch.manual_seed(0); random.seed(0)
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    phi = np.asarray(extrapolated_marginals(
                        1 if upper else count, count, 5, n_small=50, n_large=n,
                        p_l=np.ones(count)/count if upper else None, verbose=False))
                if upper:
                    weights['penalty'][count] = phi
                else:
                    weights['cv'][n, count] = phi
        base.atomic_pickle(path, weights)
    for (n, count), phi in weights['cv'].items():
        empirical.get_weights(weights['cv'], n, count, empirical.minimum_gap(n, .1))
    for count, phi in weights['penalty'].items():
        empirical.get_weights({(N, count):phi}, N, count, N//10)
    return root, spec, weights


def cv_prediction(strength, labels, weights, trial, draws):
    """Only oracle model parameters and frozen weights enter this prediction."""
    predicted = theory.predict_cv(strength, labels, P, SIGMA**2, max(BOUNDS),
        weights['cv'], .1, folds=10, minimum_segment_fraction=None)
    cached = {}
    truth = np.flatnonzero(np.diff(labels))+1
    for bound, selected in predicted['selected_by_bound'].items():
        count = selected['selected_segments']
        if count not in cached:
            rng = np.random.default_rng(np.random.SeedSequence([SEED, N, P, trial, 2001, count]))
            cached[count] = theory.final_surrogates(strength, labels, P, SIGMA**2,
                count, weights['cv'], .1, draws, rng, minimum_segment_fraction=None)
            for fit in cached[count]['final_fits']:
                fit.update(segments=count, count=count-1)
                base.add_metrics(fit, truth, N)
        selected.update(cached[count])
    return predicted


def run_trial(trial, root, spec, weights):
    path = root/f'trial{trial:03d}.pkl'
    if path.exists():
        record = pickle.loads(path.read_bytes())
        assert record['trial'] == trial and record['spec_digest'] == root.name
        return record
    started = time.perf_counter()
    X, y, beta, truth, labels = generate_data(trial)
    strength = beta.T@beta/N
    data_digest = base.data_hash(X, y)
    cv_fit = empirical.run_cv(X, y, max(BOUNDS), weights['cv'], .1, folds=10,
                              minimum_segment_fraction=None)
    predicted = cv_prediction(strength, labels, weights, trial, spec['theory_draws'])
    min_empirical = min(f['segment_sample_ratios'].min() for f in cv_fit['fold_results'].values())
    min_theory = min(f['segment_ratios'].min() for f in predicted['fold_results'].values())
    if min_empirical <= 1 or min_theory <= 1:
        raise ValueError(f'Trial {trial}: unsupported local refit ratio '
                         f'(empirical {min_empirical}, theory {min_theory}); no candidate removed')
    residuals = [f['se_residual'] for f in predicted['fold_results'].values()]
    residuals += [f['se_residual'] for f in predicted['selected_by_bound'].values()]
    record = dict(trial=trial, n=N, p=P, truth=truth, data_hash=data_digest,
        spec_digest=root.name, signal_strength=strength, cv=cv_fit,
        cv_theory=predicted, fits={})
    draws = spec['theory_draws']
    rng = np.random.default_rng(np.random.SeedSequence([SEED, N, P, trial, 2002]))
    Z = rng.normal(size=(draws, N, 4))@np.linalg.cholesky(strength).T
    response = Z[:, np.arange(N), labels]+rng.normal(0, SIGMA, (draws, N))
    standard_noise = rng.normal(size=(draws, N, max(BOUNDS)))
    for bound in BOUNDS:
        selected = cv_fit['selected_by_bound'][bound]
        empirical_cv = base.add_metrics(dict(cp=selected['cp'].copy(),
            count=selected['selected_segments']-1, segments=selected['selected_segments'],
            validation_mean=selected['validation_mean']), truth, N)
        phi = weights['penalty'][bound]
        theta = base.empirical_fit(X, y, phi)
        penalty = base.add_metrics(base.search(theta, y, N//10, True, GAMMA), truth, N)
        state = base.squared_se(phi, strength, labels, N/P, SIGMA**2)
        residuals.extend(state[key] for key in ('b_residual','bias_residual','covariance_residual'))
        penalty_theory = []
        for draw in range(draws):
            V = Z[draw]@state['lam']+standard_noise[draw, :, :bound]@state['covariance_root'].T
            theta_se = (1-state['omega'])*V+state['omega']*response[draw, :, None]
            penalty_theory.append(base.add_metrics(
                base.search(theta_se, response[draw], N//10, True, GAMMA), truth, N))
        record['fits'][bound] = dict(cv=empirical_cv, penalty=penalty,
            cv_theory=predicted['selected_by_bound'][bound]['final_fits'],
            penalty_theory=penalty_theory,
            penalty_se={key:value for key,value in state.items() if key not in ('omega','covariance_root')})
        for fit in [empirical_cv, penalty, *record['fits'][bound]['cv_theory'], *penalty_theory]:
            assert fit['count'] == len(fit['cp']) == fit['segments']-1
            assert 0 <= fit['count'] < bound
            assert np.all((fit['cp'] > 0) & (fit['cp'] < N))
            assert np.all(np.diff(fit['cp']) >= N//10)
    record['diagnostics'] = dict(min_empirical_segment_ratio=float(min_empirical),
        min_theory_segment_ratio=float(min_theory), max_se_residual=float(max(residuals)),
        cv_score_mae=float(np.mean(np.abs(cv_fit['normalized_cv_scores']-predicted['normalized_cv_scores']))))
    assert record['diagnostics']['max_se_residual'] < 1e-9
    assert base.data_hash(X, y) == data_digest
    record['seconds'] = time.perf_counter()-started
    base.atomic_pickle(path, record)
    return record


def summary(payload):
    rows = []
    for bound in BOUNDS:
        for source, label in [('cv','WERM+CV'), ('penalty','WERM+P'),
                              ('cv_theory','Theory (CV)'), ('penalty_theory','Theory (P)')]:
            selected = [r['fits'][bound][source] for r in payload['records']]
            fits = [fit for draws in selected for fit in draws] if 'theory' in source else selected
            # CV Theory selects once per signal matrix; final draws vary only locations.
            count_fits = [draws[0] for draws in selected] if source == 'cv_theory' else fits
            counts = np.array([fit['count'] for fit in count_fits])
            haus = np.array([fit['haus'] for fit in fits]); finite = haus[np.isfinite(haus)]
            rows.append(dict(L=bound, method=label, trials=len(selected), draws=len(fits),
                count_draws=len(count_fits),
                mean_count=float(counts.mean()), correct_count=float(np.mean(counts == 3)),
                empty=int(np.sum(counts == 0)),
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
    payload = run(root, spec, weights, jobs=args.jobs)
    print(summary(payload).to_string(index=False), flush=True)
