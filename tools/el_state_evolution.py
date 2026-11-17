"""Squared-loss SE overlays for penalized and fixed-count L sensitivity.

The four true regimes and the fitted regression columns have separate dimensions.
No CV, adjusted-fit search, or empirical count is used in the SE prediction.
"""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')

import hashlib
import json
import pickle
import time
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed, parallel_config
from WERM.config import extrapolated_marginals
from WERM.experiment_utils import data_hash, evaluate_changepoints
from WERM.gof import manuscript_log_binomial_k_penalty
from WERM.proximal import fit_weighted
from WERM.signal_strength import solve_b
from tools.m2_adjusted_experiment import squared_partition

N, P, SEED = 1000, 100, 20261120
SIGMA = .5
TRUTH = np.array([200, 500, 800])
MODES = {'penalty': tuple(range(1, 7)), 'fixed': tuple(range(2, 7))}


def generate_data(trial, n=N, p=P):
    """Exactly the existing E_L_upper_bound data generator and seed scheme."""
    rng = np.random.default_rng(np.random.SeedSequence([SEED, n, p, trial]))
    beta = np.empty((p, 4))
    beta[:, 0] = rng.normal(size=p)*np.sqrt(n/p)
    for k in range(1, 4):
        beta[:, k] = beta[:, k-1] + 1.5*np.sqrt(n/p)*rng.normal(size=p)
    X = rng.normal(size=(n, p))/np.sqrt(n)
    truth = np.rint(n*np.array([.2, .5, .8])).astype(int)
    state = np.searchsorted(truth, np.arange(n), side='right')
    mu = np.einsum('ij,ij->i', X, beta[:, state].T)
    y = mu + rng.normal(0, SIGMA, n)
    return X, y, beta, truth, state, mu


def squared_se(phi, gamma, labels, delta, noise_variance, loss_curvature=1.):
    """Finite-sum specialization of eq:ls-b/lambda/kappa-linear-model.

    M=0.5*(u-y)^2 has curvature 1 (paper's unhalved loss has curvature 2).
    Gamma is R-by-R; Lambda is R-by-L; K is L-by-L. R need not equal L.
    """
    phi, gamma, labels = np.asarray(phi), np.asarray(gamma), np.asarray(labels)
    count, n = phi.shape
    regimes = gamma.shape[0]
    if delta <= 1 or gamma.shape != (regimes, regimes) or labels.shape != (n,):
        raise ValueError('Invalid SE dimensions or aspect ratio')
    if np.any(labels < 0) or np.any(labels >= regimes):
        raise ValueError('True regime index outside Gamma')
    if np.linalg.eigvalsh(gamma).min() <= 0 or np.any(phi < 0):
        raise ValueError('Positive definite Gamma and nonnegative weights required')
    b = np.array([solve_b(loss_curvature*w, delta) for w in phi])
    a = loss_curvature*b[:, None]*phi
    omega = (a/(1+a)).T
    true_onehot = np.eye(regimes)[labels]
    lam = delta*true_onehot.T@omega/n
    s = delta*omega.T@omega/n
    numerator = noise_variance*s
    # Integrate Z analytically in each true regime; keep cross-column covariance.
    for r in range(regimes):
        d = lam - np.eye(regimes)[:, r, None]
        weight_cross = delta*omega[labels == r].T@omega[labels == r]/n
        numerator = numerator + weight_cross*(d.T@gamma@d)
    if np.any(1-s <= 0):
        raise ValueError('Nonpositive covariance denominator')
    kappa = numerator/(1-s)
    kappa = (kappa+kappa.T)/2
    eigenvalues, eigenvectors = np.linalg.eigh(kappa)
    if eigenvalues.min() < -1e-10:
        raise ValueError('SE covariance is not positive semidefinite')
    # Numerical roundoff only; retain all nonnegative eigenvalues without ridge.
    covariance_root = eigenvectors*np.sqrt(np.maximum(eigenvalues, 0))
    b_residual = float(np.max(np.abs(np.mean(1/(1+a), axis=1)-(1-1/delta))))
    mean_weighted_d = lam*omega.mean(axis=0)[None, :] - true_onehot.T@omega/n
    bias_residual = float(np.linalg.norm(gamma@mean_weighted_d))
    covariance_residual = float(np.linalg.norm(kappa - (numerator+s*kappa)))
    return dict(b=b, lam=lam, kappa=kappa, omega=omega, covariance_root=covariance_root,
                b_residual=b_residual, bias_residual=bias_residual,
                covariance_residual=covariance_residual, min_eigenvalue=float(eigenvalues.min()))


def search(theta, y, gap, upper_bound, penalty_weight=1.):
    """Exact same fixed/upper-bound objective as the repository DP."""
    n, count = theta.shape
    candidates = range(1, count+1) if upper_bound else (count,)
    penalty = manuscript_log_binomial_k_penalty(n, penalty_weight)
    fits = []
    for k in candidates:
        cp = squared_partition(theta[:, :k], y, gap)
        signal = np.searchsorted(cp, np.arange(n), side='right')
        raw = float(.5*np.sum((theta[np.arange(n), signal]-y)**2))
        total = raw + (penalty(k) if upper_bound else 0.)
        fits.append(dict(cp=cp, count=k-1, segments=k, raw_loss=raw, total_loss=total))
    best = min(fits, key=lambda f: (f['total_loss'], f['segments']))
    return dict(best, candidate_raw=np.array([f['raw_loss'] for f in fits]),
                candidate_total=np.array([f['total_loss'] for f in fits]))


def empirical_fit(X, y, phi):
    coefficients = []
    for w in phi:
        fit = fit_weighted(X, y, w/w.sum(), loss='least_squares', loss_scale=1.,
                           penalty='none', penalty_strength=0.)
        if not fit.converged:
            raise RuntimeError('Weighted LS failed KKT tolerance')
        coefficients.append(fit.coef)
    return X@np.column_stack(coefficients)


def add_metrics(fit, truth, n):
    fit.update(evaluate_changepoints(fit['cp'], truth, n))
    fit['empty'] = len(fit['cp']) == 0
    return fit


def atomic_pickle(path, obj):
    path = Path(path)
    tmp = path.with_suffix(f'.{os.getpid()}.tmp')
    tmp.write_bytes(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
    tmp.replace(path)


def prepare(trials=100, theory_draws=20, penalty_weight=1.):
    source_names = ['tools/el_state_evolution.py', 'tools/m2_adjusted_experiment.py',
                    'WERM/config.py', 'WERM/proximal.py', 'WERM/signal_strength.py',
                    'WERM/gof.py', 'WERM/experiment_utils.py']
    specification = dict(schema=1, n=N, p=P, seed=SEED, trials=trials,
        theory_draws=theory_draws, penalty_weight=penalty_weight, sigma=SIGMA,
        true_fractions=[.2,.5,.8], modes=MODES, gap=N//10, loss='half squared',
        marginal_seed=0, gamma_source='beta.T @ beta / n (oracle, simulation only)',
        sources={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_names})
    digest = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()[:12]
    root = Path('results/E_L_SE_penalty_fixed')/digest
    root.mkdir(parents=True, exist_ok=True)
    (root/'specification.json').write_text(json.dumps(specification, indent=2))
    if (root/'weights.pkl').exists():
        weights = pickle.loads((root/'weights.pkl').read_bytes())
    else:
        import torch
        weights = {}
        for mode, counts in MODES.items():
            for count in counts:
                np.random.seed(0)
                torch.manual_seed(0)
                p_l = np.ones(count)/count if mode == 'penalty' else None
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    phi = np.asarray(extrapolated_marginals(
                        1 if mode=='penalty' else count, count, 5,
                        n_small=50, n_large=N, p_l=p_l, verbose=False))
                if phi.shape != (count, N) or not np.isfinite(phi).all() or np.any(phi<0):
                    raise ValueError('Invalid marginal weights')
                np.testing.assert_allclose(phi.sum(axis=0), 1, atol=1e-8)
                weights[mode, count] = phi
        atomic_pickle(root/'weights.pkl', weights)
    return root, specification, weights


def run_trial(trial, root, specification, weights):
    path = root/f'trial{trial:03d}.pkl'
    if path.exists():
        return pickle.loads(path.read_bytes())
    started = time.perf_counter()
    X, y, beta, truth, labels, _ = generate_data(trial)
    gamma = beta.T@beta/N
    result = dict(trial=trial, data_hash=data_hash(X,y), gamma=gamma, truth=truth, fits={})
    # Theory draws are independent of empirical X/noise. Common standard draws
    # across settings reduce comparison noise without conditioning on fitted outputs.
    rng = np.random.default_rng(np.random.SeedSequence([SEED, trial, 917341]))
    draws = specification['theory_draws']
    Z = rng.normal(size=(draws, N, 4))@np.linalg.cholesky(gamma).T
    eps = rng.normal(0, SIGMA, size=(draws, N))
    response = Z[:, np.arange(N), labels]+eps
    standard_noise = rng.normal(size=(draws, N, 6))
    for mode, counts in MODES.items():
        for count in counts:
            phi = weights[mode, count]
            theta = empirical_fit(X, y, phi)
            empirical = add_metrics(search(theta, y, N//10, mode=='penalty',
                                           specification['penalty_weight']), truth, N)
            se = squared_se(phi, gamma, labels, N/P, SIGMA**2)
            theory = []
            for draw in range(draws):
                G = standard_noise[draw, :, :count]@se['covariance_root'].T
                V = Z[draw]@se['lam']+G
                # Proximal fitted values, NOT the adjusted values V.
                predicted = (1-se['omega'])*V + se['omega']*response[draw, :, None]
                fit = search(predicted, response[draw], N//10, mode=='penalty',
                             specification['penalty_weight'])
                theory.append(add_metrics(fit, truth, N))
            result['fits'][mode, count] = dict(empirical=empirical, theory=theory,
                se={key:value for key,value in se.items() if key not in ('omega','covariance_root')},
                empirical_mean_squared_fit=float(np.mean(theta**2)))
    result['seconds'] = time.perf_counter()-started
    atomic_pickle(path, result)
    return result


def summary(payload):
    import pandas as pd
    rows = []
    for mode, counts in MODES.items():
        for count in counts:
            settings = [r['fits'][mode,count] for r in payload['records']]
            for source in ('empirical','theory'):
                fits = ([s['empirical'] for s in settings] if source=='empirical'
                        else [f for s in settings for f in s['theory']])
                haus = np.array([f['haus'] for f in fits])
                selected = np.array([f['count'] for f in fits])
                finite = haus[np.isfinite(haus)]
                rows.append(dict(mode=mode, fitted_segments=count, x=count if mode=='penalty' else count-1,
                    source=source, draws=len(fits), empty=int(np.sum(selected==0)),
                    mean_count=float(selected.mean()), correct_count=float(np.mean(selected==3)),
                    mean_haus=float(finite.mean()) if len(finite) else np.nan,
                    median_haus=float(np.median(finite)) if len(finite) else np.nan))
    return pd.DataFrame(rows)


def run(root, specification, weights, jobs=12):
    with parallel_config(backend='loky', inner_max_num_threads=1):
        records = Parallel(n_jobs=jobs, verbose=10)(
            delayed(run_trial)(i, root, specification, weights) for i in range(specification['trials']))
    payload = dict(specification=specification, records=records)
    atomic_pickle(root/'study.pkl', payload)
    summary(payload).to_csv(root/'summary.csv', index=False)
    return payload


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=100)
    parser.add_argument('--theory-draws', type=int, default=20)
    parser.add_argument('--jobs', type=int, default=12)
    args = parser.parse_args()
    root, spec, weights = prepare(args.trials, args.theory_draws)
    print('Results:', root, flush=True)
    payload = run(root, spec, weights, args.jobs)
    print(summary(payload).to_string(index=False), flush=True)
