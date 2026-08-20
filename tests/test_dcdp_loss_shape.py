import numpy as np

from WERM import DCDP_utils as D


def test_loss_func_is_residual_sum_of_squares():
    est = D.dcdp_cv_grid_linear(10, [0.1], [1])
    m, p = 40, 5
    rng = np.random.default_rng(0)
    X = rng.normal(size=(m, p))
    beta = rng.normal(size=(p, 1))
    y2 = X @ beta + 0.1 * rng.normal(size=(m, 1))
    expected = float(np.sum((y2 - X @ beta) ** 2))
    assert np.isclose(est.loss_func(y2, X, beta), expected)
    assert not np.isclose(est.loss_func(y2.ravel(), X, beta), expected)


def test_loss_is_subadditive_under_correct_shape():
    """Splitting a homogeneous segment must not reduce the loss by construction."""
    est = D.dcdp_cv_grid_linear(10, [0.1], [1])
    rng = np.random.default_rng(1)
    m, p = 80, 10
    X = rng.normal(size=(m, p))
    beta = rng.normal(size=(p, 1))
    y = X @ beta + 0.5 * rng.normal(size=(m, 1))
    whole = est.loss_func(y, X, est.estimate_func(y, X, 0.1))
    halves = sum(
        est.loss_func(y[a:b], X[a:b], est.estimate_func(y[a:b], X[a:b], 0.1))
        for a, b in [(0, 40), (40, 80)]
    )
    assert whole - halves < 0.5 * whole
