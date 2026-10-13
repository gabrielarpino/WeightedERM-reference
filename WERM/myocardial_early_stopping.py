"""Explicit finite-budget logistic fitting for the myocardial sensitivity study."""
import numpy as np
from WERM.proximal import fit_weighted


def early_logistic_fit(X, y, weights, budget):
    """Zero-start unpenalized FISTA, no intercept, capped at ``budget`` updates.

    The initial/maximum step is 1/L, where L is the exact global weighted
    logistic curvature bound. Backtracking and monotone restart are supplied
    by the package solver. No warm start or explicit penalty is used.
    Very small convergence tolerance may terminate earlier on a stationary
    problem; the returned n_iter records the actual number of updates.
    """
    if isinstance(budget, (bool, np.bool_)) or int(budget) != budget or budget < 1:
        raise ValueError('budget must be a positive integer')
    X = np.asarray(X, float); weights = np.asarray(weights, float)
    if (X.ndim != 2 or weights.shape != (len(X),) or
            not np.isfinite(X).all() or not np.isfinite(weights).all() or
            np.any(weights < 0) or weights.sum() <= 0):
        raise ValueError('finite design and nonnegative positive-sum weights required')
    wn = weights/weights.sum()
    L = float(np.linalg.eigvalsh(X.T@(wn[:,None]*X))[-1])/4
    return fit_weighted(X, y, loss='logistic', sample_weight=weights,
        penalty='none', penalty_strength=0., initial=np.zeros(X.shape[1]),
        initial_step=1/L if L > 0 else 1., max_iter=int(budget), tol=1e-14)
