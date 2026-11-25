"""State-evolution and fitting helpers for the paired upper-bound experiment."""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')

import pickle
from pathlib import Path

import numpy as np
from WERM.experiment_utils import data_hash, evaluate_changepoints
from WERM.gof import manuscript_log_binomial_k_penalty
from WERM.proximal import fit_weighted
from WERM.signal_strength import solve_b
from tools.m2_adjusted_experiment import squared_partition

N, P, SEED = 1000, 100, 20261120


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


