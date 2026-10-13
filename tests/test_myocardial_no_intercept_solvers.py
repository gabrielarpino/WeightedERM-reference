import numpy as np
import pytest
from scipy.special import expit
from WERM.myocardial_no_intercept_solvers import fit_column


def data():
    rng = np.random.default_rng(82)
    X = rng.normal(size=(60, 4))
    y = rng.binomial(1, expit(X @ np.array([.3, -.5, .1, .2])))
    return X, y, rng.uniform(.1, 2, len(y))


@pytest.mark.parametrize('method', ['gd', 'fista'])
def test_first_step(method):
    X, y, w = data(); wn = w / w.sum()
    L = np.linalg.eigvalsh(X.T @ (wn[:, None] * X))[-1] / 4
    r = fit_column(X, y, w, method, 1, .1)
    np.testing.assert_allclose(r['coef'], -.1 / L * (X.T @ (wn * (.5-y))), atol=1e-12)


@pytest.mark.parametrize('method', ['gd', 'fista', 'lbfgs', 'sgd', 'saga'])
def test_weight_invariance_and_reproducibility(method):
    X, y, w = data()
    a = fit_column(X, y, w, method, 3)
    b = fit_column(X, y, w*7, method, 3)
    c = fit_column(X, y, w, method, 3)
    np.testing.assert_allclose(a['coef'], b['coef'], rtol=1e-9, atol=1e-10)
    np.testing.assert_array_equal(a['coef'], c['coef'])
    assert a['iterations'] <= 3
    assert a['intercept'] == 0 and a['penalty'] == 'none'


def test_gradient_and_converged_reference():
    from sklearn.linear_model import LogisticRegression
    X, y, w = data()
    a = fit_column(X, y, w, 'lbfgs', 300)
    reference = LogisticRegression(penalty=None, fit_intercept=False, solver='lbfgs',
                                   tol=1e-11, max_iter=1000).fit(X, y, sample_weight=w)
    np.testing.assert_allclose(a['coef'], reference.coef_[0], atol=2e-5)
    assert a['kkt'] < 1e-7


def test_liblinear_rejected():
    with pytest.raises(ValueError, match='unpenalized'):
        fit_column(*data(), 'liblinear', 2)
