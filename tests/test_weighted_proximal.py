import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge

import WERM.estimators as estimators
from WERM.proximal import (
    fit_weighted,
    infer_loss,
    infer_penalty,
    weighted_smooth_value_and_gradient,
)


def _huber(prediction, truth, delta=0.9):
    residual = prediction - truth
    absolute = jnp.abs(residual)
    return jnp.where(
        absolute <= delta,
        0.5 * residual**2,
        delta * (absolute - 0.5 * delta),
    )


def _logistic(prediction, truth):
    return jnp.logaddexp(0.0, prediction) - truth * prediction


@pytest.mark.parametrize(
    "loss_function, expected_name, expected_delta, expected_scale",
    [
        (lambda prediction, truth: (prediction - truth) ** 2,
         "least_squares", None, 2.0),
        (_huber, "huber", 0.9, 1.0),
        (_logistic, "logistic", None, 1.0),
    ],
)
def test_infer_supported_werm_loss_contracts(
    loss_function, expected_name, expected_delta, expected_scale
):
    inferred = infer_loss(loss_function)
    assert inferred.name == expected_name
    assert inferred.scale == pytest.approx(expected_scale, rel=2e-5)
    if expected_delta is None:
        assert inferred.huber_delta is None
    else:
        assert inferred.huber_delta == pytest.approx(expected_delta, rel=2e-5)


@pytest.mark.parametrize("norm, exponent", [("l1", 1), ("l2", 2)])
def test_infer_supported_penalty_and_unpenalized_coordinates(norm, exponent):
    strength = 0.37

    def penalty(beta):
        return strength * jnp.sum(jnp.abs(beta[1:]) ** exponent)

    inferred = infer_penalty(penalty, dimension=5)
    assert inferred.name == norm
    assert inferred.strength == pytest.approx(strength)
    np.testing.assert_array_equal(inferred.penalized_mask, [False, True, True, True, True])


def test_exact_unpenalized_weighted_least_squares_matches_sklearn():
    rng = np.random.default_rng(601)
    X = rng.normal(size=(180, 12))
    y = X @ rng.normal(size=12) + rng.normal(scale=0.2, size=180)
    weights = rng.uniform(0.1, 3.0, size=180)

    actual = fit_weighted(
        X, y, weights, loss="least_squares", penalty="none"
    )
    reference = LinearRegression(fit_intercept=False).fit(
        X, y, sample_weight=weights
    )

    assert actual.solver == "weighted-least-squares-exact"
    assert actual.converged
    assert actual.n_iter == 0
    np.testing.assert_allclose(actual.coef, reference.coef_, rtol=2e-10, atol=2e-10)


def test_exact_least_squares_rank_deficiency_uses_minimum_norm_solution():
    rng = np.random.default_rng(606)
    base = rng.normal(size=(80, 4))
    X = np.c_[base, base[:, 1]]
    y = rng.normal(size=80)
    weights = rng.uniform(0.2, 2.0, size=80)

    actual = fit_weighted(
        X, y, weights, loss="least_squares", penalty="none"
    )
    weighted_X = np.sqrt(weights / weights.sum())[:, None] * X
    weighted_y = np.sqrt(weights / weights.sum()) * y
    reference = np.linalg.lstsq(weighted_X, weighted_y, rcond=None)[0]

    np.testing.assert_allclose(actual.coef, reference, rtol=2e-10, atol=2e-10)
    assert actual.coef[1] == pytest.approx(actual.coef[-1], abs=2e-10)


def test_exact_weighted_least_squares_l2_matches_sklearn_ridge():
    rng = np.random.default_rng(602)
    X = rng.normal(size=(160, 10))
    y = X @ rng.normal(size=10) + rng.normal(scale=0.3, size=160)
    weights = rng.uniform(0.2, 2.5, size=160)
    strength = 0.08

    actual = fit_weighted(
        X,
        y,
        weights,
        loss="least_squares",
        loss_scale=2.0,  # M(pred, y) = (pred-y)^2 in the package.
        penalty="l2",
        penalty_strength=strength,
    )
    # sklearn minimizes sum_i w_i (y_i-x_i b)^2 + alpha ||b||_2^2.
    reference = Ridge(
        alpha=strength * weights.sum(), fit_intercept=False, solver="cholesky"
    ).fit(X, y, sample_weight=weights)

    assert actual.solver == "weighted-ridge-exact"
    assert actual.n_iter == 0
    np.testing.assert_allclose(actual.coef, reference.coef_, rtol=2e-9, atol=2e-9)


def test_exact_weighted_ridge_does_not_penalize_intercept():
    rng = np.random.default_rng(603)
    features = rng.normal(size=(120, 5))
    X = np.c_[np.ones(120), features]
    y = 3.5 + features @ np.array([1.0, -0.5, 0.0, 0.2, 0.0]) + rng.normal(
        scale=0.1, size=120
    )
    weights = rng.uniform(0.5, 2.0, size=120)

    actual = fit_weighted(
        X,
        y,
        weights,
        loss="least_squares",
        loss_scale=2.0,
        penalty="l2",
        penalty_strength=100.0,
        unpenalized_indices=(0,),
    )

    assert actual.coef[0] == pytest.approx(np.average(y, weights=weights), abs=0.03)
    assert np.linalg.norm(actual.coef[1:]) < 0.02


def test_weighted_least_squares_l1_matches_sklearn_lasso():
    rng = np.random.default_rng(605)
    X = rng.normal(size=(220, 14))
    beta = np.r_[1.2, -0.8, 0.4, np.zeros(11)]
    y = X @ beta + rng.normal(scale=0.25, size=220)
    weights = rng.uniform(0.2, 2.2, size=220)
    strength = 0.055

    actual = fit_weighted(
        X,
        y,
        weights,
        loss="least_squares",
        penalty="l1",
        penalty_strength=strength,
        tol=1e-9,
    )
    reference = Lasso(
        alpha=strength,
        fit_intercept=False,
        tol=1e-12,
        max_iter=100000,
        selection="cyclic",
    ).fit(X, y, sample_weight=weights)

    assert actual.converged
    assert actual.solver == "fista-backtracking"
    np.testing.assert_allclose(actual.coef, reference.coef_, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("loss", ["huber", "logistic"])
@pytest.mark.parametrize("penalty", ["none", "l2"])
def test_fista_smooth_objectives_match_independent_scipy(loss, penalty):
    rng = np.random.default_rng(700 + (loss == "logistic") + 2 * (penalty == "l2"))
    X = rng.normal(size=(100, 8))
    true_beta = np.array([0.8, -0.6, 0.3, 0.0, 0.0, 0.2, 0.0, 0.0])
    if loss == "logistic":
        probability = 1.0 / (1.0 + np.exp(-(X @ true_beta)))
        y = rng.binomial(1, probability)
    else:
        y = X @ true_beta + rng.standard_t(4, size=100) * 0.2
    weights = rng.uniform(0.2, 2.0, size=100)
    strength = 0.06 if penalty == "l2" else 0.0

    actual = fit_weighted(
        X,
        y,
        weights,
        loss=loss,
        huber_delta=0.9,
        penalty=penalty,
        penalty_strength=strength,
        tol=1e-9,
        max_iter=10000,
    )

    def objective(beta):
        smooth, gradient = weighted_smooth_value_and_gradient(
            beta, X, y, weights, loss=loss, huber_delta=0.9
        )
        value = smooth + strength * np.sum(beta**2)
        jacobian = gradient + 2.0 * strength * beta
        return value, jacobian

    reference = minimize(
        lambda beta: objective(beta)[0],
        np.zeros(X.shape[1]),
        jac=lambda beta: objective(beta)[1],
        method="L-BFGS-B",
        options={"ftol": 1e-14, "gtol": 1e-11, "maxiter": 10000},
    )
    assert reference.success, reference.message
    assert actual.converged
    assert actual.kkt_violation <= 2e-8
    np.testing.assert_allclose(actual.coef, reference.x, atol=2e-5, rtol=2e-5)


def test_unpenalized_logistic_matches_sklearn():
    rng = np.random.default_rng(604)
    X = rng.normal(size=(300, 9))
    probability = 1.0 / (1.0 + np.exp(-(X @ rng.normal(size=9))))
    y = rng.binomial(1, probability)
    weights = rng.uniform(0.2, 2.0, size=300)

    actual = fit_weighted(
        X, y, weights, loss="logistic", penalty="none", tol=1e-9
    )
    reference = LogisticRegression(
        penalty=None, fit_intercept=False, solver="lbfgs", tol=1e-12, max_iter=10000
    ).fit(X, y, sample_weight=weights)

    np.testing.assert_allclose(
        actual.coef, reference.coef_.reshape(-1), atol=2e-5, rtol=2e-5
    )


def test_unpenalized_separable_logistic_does_not_grow_step_without_bound():
    """A nonexistent finite MLE must not make local backtracking overflow."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(24, 48))
    y = (X[:, 0] > 0.0).astype(float)

    actual = fit_weighted(
        X,
        y,
        loss="logistic",
        penalty="none",
        initial_step=1.0,
        step_growth=1.2,
        max_iter=3000,
        tol=1e-4,
    )

    assert actual.converged
    assert np.all(np.isfinite(actual.coef))
    assert actual.step_size <= 1.0


@pytest.mark.parametrize("penalty", ["none", "l1", "l2"])
@pytest.mark.parametrize("loss", ["least_squares", "huber", "logistic"])
def test_weighted_erm_routes_every_supported_objective_through_core_solver(
    monkeypatch, loss, penalty
):
    calls = []

    def fake_fit(X, y, sample_weight, **kwargs):
        calls.append(kwargs)
        return type(
            "Result", (),
            {"coef": np.zeros(X.shape[1]), "converged": True}
        )()

    monkeypatch.setattr(estimators, "fit_weighted", fake_fit)
    monkeypatch.setattr(
        estimators,
        "_exact_partition_search",
        lambda *args, **kwargs: (np.array([], dtype=int), 0.0),
    )
    monkeypatch.setattr(
        estimators.scipy.optimize,
        "minimize",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("legacy scipy coefficient optimizer was called")
        ),
    )
    y = np.array([0, 1, 0, 1, 0, 1], dtype=float) if loss == "logistic" else np.arange(6.0)
    M = {
        "least_squares": lambda prediction, truth: (prediction - truth) ** 2,
        "huber": _huber,
        "logistic": _logistic,
    }[loss]

    estimators.weighted_erm(
        X=np.arange(18.0).reshape(6, 3),
        y=y,
        M=M,
        L=2,
        Delta_n=1,
        gamma=0.0,
        coefficient_loss=loss,
        coefficient_penalty=penalty,
        coefficient_lambda=0.0 if penalty == "none" else 0.2,
        huber_delta=0.9 if loss == "huber" else 1.35,
    )

    assert len(calls) == 2
    assert all(call["loss"] == loss for call in calls)
    assert all(call["penalty"] == penalty for call in calls)
