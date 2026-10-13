"""Matched, zero-start, explicitly unpenalized finite logistic fits."""
import warnings
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.exceptions import ConvergenceWarning
from WERM.proximal import fit_weighted


def fit_column(X, y, weights, method, budget, step_scale=1., seed=42):
    X, y, weights = np.asarray(X, float), np.asarray(y, float), np.asarray(weights, float)
    if method not in ('fista', 'gd', 'lbfgs', 'sgd', 'saga'):
        raise ValueError('supported unpenalized solvers: fista, gd, lbfgs, sgd, saga')
    if isinstance(budget, bool) or int(budget) != budget or budget < 1:
        raise ValueError('positive integer budget required')
    if (X.ndim != 2 or y.shape != (len(X),) or weights.shape != y.shape or
        not np.isfinite(X).all() or not np.isfinite(weights).all() or
        np.any(weights < 0) or weights.sum() <= 0 or not np.isin(y, [0, 1]).all() or
        not np.isfinite(step_scale) or not 0 < step_scale <= 1):
        raise ValueError('invalid design, binary response, weights, or step scale')
    wn = weights / weights.sum(); beta = np.zeros(X.shape[1]); budget = int(budget)
    def objective(b):
        z = X @ b
        return float(wn @ (np.logaddexp(0, z)-y*z)), X.T @ (wn*(expit(z)-y))
    L = float(np.linalg.eigvalsh(X.T @ (wn[:, None]*X))[-1])/4
    step = step_scale / L if L > 0 else step_scale
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        if method == 'fista':
            r = fit_weighted(X, y, loss='logistic', sample_weight=weights,
                penalty='none', penalty_strength=0., initial=beta,
                initial_step=step, max_iter=budget, tol=1e-14)
            beta, iterations = r.coef, r.n_iter
        elif method == 'gd':
            for _ in range(budget): beta -= step*objective(beta)[1]
            iterations = budget
        elif method == 'lbfgs':
            r = minimize(objective, beta, jac=True, method='L-BFGS-B',
                options=dict(maxiter=budget, ftol=0., gtol=0., maxcor=10, maxls=20))
            beta, iterations = r.x, r.nit
        elif method == 'sgd':
            # Mean-one weights preserve the normalized objective. Per-observation
            # curvature bounds give a stable fixed learning rate across epochs.
            bound = float(np.max(len(y)*wn*np.sum(X*X, axis=1)))/4
            step = step_scale/bound if bound > 0 else step_scale
            r = SGDClassifier(loss='log_loss', penalty=None, fit_intercept=False,
                learning_rate='constant', eta0=step, max_iter=budget, tol=None,
                shuffle=True, random_state=seed, average=False)
            r.fit(X, y, sample_weight=len(y)*wn)
            beta, iterations = r.coef_[0], r.n_iter_
        else:
            r = LogisticRegression(penalty=None, solver='saga', fit_intercept=False,
                max_iter=budget, tol=0., random_state=seed)
            r.fit(X, y, sample_weight=len(y)*wn)
            beta, iterations = r.coef_[0], int(r.n_iter_[0])
    value, grad = objective(beta)
    if not np.isfinite(beta).all() or not np.isfinite(value):
        raise FloatingPointError('nonfinite fit')
    return dict(coef=beta, iterations=int(iterations), objective=value,
        kkt=float(np.max(np.abs(grad))), norm=float(np.linalg.norm(beta)),
        intercept=0., penalty='none', step=float(step), seed=seed,
        success=bool(np.max(np.abs(grad)) < 1e-8))
