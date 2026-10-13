import numpy as np
from scipy.special import expit
from WERM.myocardial_early_stopping import early_logistic_fit


def test_one_iteration_matches_weighted_gradient_step():
    X = np.array([[-1., 2.], [1., 1.], [2., -1.], [0., 1.]])
    y = np.array([0., 1., 1., 0.]); w = np.array([1., 2., 3., 1.])
    fit = early_logistic_fit(X, y, w, 1)
    normalized = w/w.sum()
    L = np.linalg.eigvalsh(X.T@(normalized[:, None]*X)).max()/4
    expected = -X.T@(normalized*(.5-y))/L
    np.testing.assert_allclose(fit.coef, expected, atol=1e-12)
    assert fit.n_iter == 1


def test_separable_finite_fit_and_weight_scale_invariance():
    X = np.array([[-2.], [-1.], [1.], [2.]])
    y = np.array([0., 0., 1., 1.]); w = np.array([1., 2., 1., 3.])
    short = early_logistic_fit(X, y, w, 3)
    long = early_logistic_fit(X, y, w, 100)
    scaled = early_logistic_fit(X, y, 11*w, 100)
    assert np.isfinite(long.coef).all()
    assert long.objective < short.objective
    assert np.linalg.norm(long.coef) > np.linalg.norm(short.coef)
    np.testing.assert_allclose(long.coef, scaled.coef, rtol=1e-12)
    assert long.n_iter <= 100


def test_invalid_budget_rejected():
    import pytest
    with pytest.raises(ValueError):
        early_logistic_fit(np.ones((3,1)), np.array([0,1,0]), np.ones(3), 0)
