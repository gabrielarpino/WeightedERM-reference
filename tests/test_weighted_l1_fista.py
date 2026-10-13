import numpy as np
import pytest
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

from WERM.proximal import (
    fit_weighted_l1,
    soft_threshold,
    weighted_smooth_value_and_gradient,
)


def _finite_difference_gradient(function, point, step=1e-6):
    gradient = np.empty_like(point)
    for index in range(point.size):
        direction = np.zeros_like(point)
        direction[index] = step
        gradient[index] = (
            function(point + direction) - function(point - direction)
        ) / (2.0 * step)
    return gradient


def _slsqp_reference(X, y, weights, loss, l1, huber_delta=1.35):
    """Independent epigraph formulation of the small convex test problems."""
    dimension = X.shape[1]

    def objective(parameters):
        beta, epigraph = parameters[:dimension], parameters[dimension:]
        value, _ = weighted_smooth_value_and_gradient(
            beta, X, y, weights, loss=loss, huber_delta=huber_delta
        )
        return value + l1 * np.sum(epigraph)

    def jacobian(parameters):
        beta = parameters[:dimension]
        _, gradient = weighted_smooth_value_and_gradient(
            beta, X, y, weights, loss=loss, huber_delta=huber_delta
        )
        return np.r_[gradient, np.full(dimension, l1)]

    # t_j >= |beta_j| makes the L1 term smooth in the augmented variables.
    constraints = (
        {"type": "ineq", "fun": lambda z: z[dimension:] - z[:dimension]},
        {"type": "ineq", "fun": lambda z: z[dimension:] + z[:dimension]},
    )
    result = minimize(
        objective,
        np.zeros(2 * dimension),
        jac=jacobian,
        constraints=constraints,
        bounds=[(None, None)] * dimension + [(0.0, None)] * dimension,
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 5000},
    )
    assert result.success, result.message
    return result.x[:dimension], result.fun


def test_soft_threshold_can_leave_selected_coordinates_unpenalized():
    values = np.array([-3.0, -0.5, 0.25, 4.0])
    actual = soft_threshold(
        values, 1.0, penalized_mask=np.array([False, True, True, False])
    )
    np.testing.assert_allclose(actual, [-3.0, 0.0, 0.0, 4.0])


@pytest.mark.parametrize("loss", ["logistic", "huber"])
def test_weighted_smooth_gradient_matches_finite_differences(loss):
    rng = np.random.default_rng(20260901)
    X = rng.normal(size=(17, 6))
    y = rng.integers(0, 2, size=17) if loss == "logistic" else rng.normal(size=17)
    weights = rng.uniform(0.05, 3.0, size=17)
    beta = rng.normal(size=6)

    value, gradient = weighted_smooth_value_and_gradient(
        beta, X, y, weights, loss=loss, huber_delta=0.8
    )
    numerical = _finite_difference_gradient(
        lambda candidate: weighted_smooth_value_and_gradient(
            candidate, X, y, weights, loss=loss, huber_delta=0.8
        )[0],
        beta,
    )

    assert np.isfinite(value)
    np.testing.assert_allclose(gradient, numerical, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("loss", ["logistic", "huber"])
def test_fista_matches_independent_slsqp_reference_and_kkt_conditions(loss):
    rng = np.random.default_rng(17 if loss == "logistic" else 23)
    X = rng.normal(size=(80, 8))
    true_beta = np.array([1.2, -0.8, 0.5, 0.0, 0.0, -0.3, 0.0, 0.0])
    linear = X @ true_beta
    if loss == "logistic":
        probabilities = 1.0 / (1.0 + np.exp(-linear))
        y = rng.binomial(1, probabilities)
    else:
        y = linear + rng.standard_t(4, size=X.shape[0]) * 0.25
    weights = rng.uniform(0.2, 2.5, size=X.shape[0])
    l1 = 0.075

    result = fit_weighted_l1(
        X,
        y,
        weights,
        loss=loss,
        l1=l1,
        huber_delta=0.9,
        tol=1e-8,
        max_iter=5000,
    )
    reference_beta, reference_objective = _slsqp_reference(
        X, y, weights, loss, l1, huber_delta=0.9
    )

    assert result.converged
    assert result.kkt_violation <= 2e-7
    assert result.objective <= reference_objective + 2e-8
    np.testing.assert_allclose(result.coef, reference_beta, atol=2e-4, rtol=2e-4)


def test_logistic_matches_sklearn_saga_objective_scaling():
    rng = np.random.default_rng(83)
    X = rng.normal(size=(250, 12))
    beta = np.r_[1.0, -0.7, 0.4, np.zeros(9)]
    probabilities = 1.0 / (1.0 + np.exp(-(X @ beta)))
    y = rng.binomial(1, probabilities)
    weights = rng.uniform(0.25, 2.0, size=len(y))
    l1 = 0.04

    ours = fit_weighted_l1(
        X, y, weights, loss="logistic", l1=l1, tol=1e-8, max_iter=5000
    )
    # sklearn's objective is sum_i w_i loss_i + ||beta||_1 / C.
    # Dividing by sum(weights) gives our convention when C=1/(lambda*sum(w)).
    sklearn_model = LogisticRegression(
        penalty="l1",
        solver="saga",
        C=1.0 / (l1 * weights.sum()),
        fit_intercept=False,
        tol=1e-10,
        max_iter=30000,
        random_state=0,
    ).fit(X, y, sample_weight=weights)
    sklearn_beta = sklearn_model.coef_.reshape(-1)

    def objective(candidate):
        smooth, _ = weighted_smooth_value_and_gradient(
            candidate, X, y, weights, loss="logistic"
        )
        return smooth + l1 * np.sum(np.abs(candidate))

    assert ours.objective <= objective(sklearn_beta) + 2e-6
    np.testing.assert_allclose(ours.coef, sklearn_beta, atol=8e-4, rtol=8e-4)


def test_weight_rescaling_is_invariant_and_intercept_is_not_penalized():
    rng = np.random.default_rng(91)
    X_features = rng.normal(size=(180, 5))
    X = np.c_[np.ones(len(X_features)), X_features]
    probabilities = 1.0 / (1.0 + np.exp(-1.25 * np.ones(len(X))))
    y = rng.binomial(1, probabilities)
    weights = rng.uniform(0.1, 4.0, size=len(y))
    kwargs = dict(
        loss="logistic",
        l1=10.0,
        unpenalized_indices=(0,),
        tol=1e-9,
        max_iter=5000,
    )

    original = fit_weighted_l1(X, y, weights, **kwargs)
    rescaled = fit_weighted_l1(X, y, 37.0 * weights, **kwargs)

    np.testing.assert_allclose(original.coef, rescaled.coef, atol=2e-8, rtol=2e-8)
    assert abs(original.coef[0]) > 0.5
    np.testing.assert_allclose(original.coef[1:], 0.0, atol=1e-10)


def test_backtracking_handles_badly_scaled_design():
    rng = np.random.default_rng(101)
    X = rng.normal(size=(120, 6)) * np.array([1.0, 10.0, 100.0, 0.1, 3.0, 30.0])
    y = rng.normal(size=120)
    result = fit_weighted_l1(
        X,
        y,
        np.ones(len(y)),
        loss="huber",
        l1=0.1,
        initial_step=1e6,
        tol=1e-7,
        max_iter=5000,
    )

    assert result.converged
    assert result.backtracking_steps > 0
    assert np.isfinite(result.objective)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"loss": "unknown", "l1": 0.1}, "loss"),
        ({"loss": "huber", "l1": -1.0}, "l1"),
        ({"loss": "huber", "l1": 0.1, "huber_delta": 0.0}, "huber_delta"),
        ({"loss": "huber", "l1": 0.1, "unpenalized_indices": (4,)}, "unpenalized"),
    ],
)
def test_solver_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        fit_weighted_l1(
            np.ones((5, 4)), np.zeros(5), np.ones(5), **kwargs
        )


def test_solver_rejects_invalid_or_zero_total_weights():
    X = np.ones((5, 2))
    y = np.zeros(5)
    for weights in (np.zeros(5), np.array([1.0, 1.0, -1.0, 1.0, 1.0])):
        with pytest.raises(ValueError, match="sample_weight"):
            fit_weighted_l1(X, y, weights, loss="huber", l1=0.1)
