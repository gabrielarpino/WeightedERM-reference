"""Fast weighted convex solvers for the losses used by WERM.

The objective convention is deliberately explicit::

    sum_i sample_weight[i] * loss_i(beta) / sum(sample_weight)
        + l1 * sum_j penalized_mask[j] * abs(beta[j]).

Uniformly rescaling all sample weights therefore leaves the solution unchanged.
Weighted least squares with no penalty or an L2 penalty is solved exactly.
Other supported combinations use monotone FISTA with backtracking and report
the final KKT violation as a convergence certificate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
from scipy.special import expit


_LOSS_ALIASES = {
    "least_squares": "least_squares",
    "least-squares": "least_squares",
    "squared": "least_squares",
    "squared_loss": "least_squares",
    "huber": "huber",
    "logistic": "logistic",
}
_SUPPORTED_PENALTIES = frozenset(("none", "l1", "l2"))


@dataclass(frozen=True)
class LossSpec:
    name: str
    scale: float = 1.0
    huber_delta: float | None = None


@dataclass(frozen=True)
class PenaltySpec:
    name: str
    strength: float
    penalized_mask: np.ndarray


@dataclass(frozen=True)
class WeightedL1Result:
    """Result returned by :func:`fit_weighted_l1`."""

    coef: np.ndarray
    objective: float
    converged: bool
    n_iter: int
    kkt_violation: float
    step_size: float
    backtracking_steps: int
    solver: str = "fista-backtracking"


WeightedProximalResult = WeightedL1Result


def _canonical_loss_name(loss):
    try:
        return _LOSS_ALIASES[str(loss).lower()]
    except KeyError as error:
        raise ValueError(
            f"loss must be one of {sorted(set(_LOSS_ALIASES.values()))}"
        ) from error


def _validated_problem(
    X, y, sample_weight, loss, huber_delta, loss_scale=1.0
):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    sample_weight = np.asarray(sample_weight, dtype=float).reshape(-1)
    if X.ndim != 2 or X.shape[0] != y.size:
        raise ValueError("X and y must have matching sample dimensions")
    if X.shape[1] < 1:
        raise ValueError("X must contain at least one feature")
    if sample_weight.size != y.size:
        raise ValueError("sample_weight must have one value per observation")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("X and y must contain only finite values")
    if (
        not np.all(np.isfinite(sample_weight))
        or np.any(sample_weight < 0.0)
        or not float(np.sum(sample_weight)) > 0.0
    ):
        raise ValueError("sample_weight must be finite, nonnegative, and have positive sum")
    loss = _canonical_loss_name(loss)
    if loss == "logistic" and not np.all(np.isin(y, (0.0, 1.0))):
        raise ValueError("logistic outcomes must be encoded as zero or one")
    if not np.isfinite(huber_delta) or huber_delta <= 0.0:
        raise ValueError("huber_delta must be finite and positive")
    loss_scale = float(loss_scale)
    if not np.isfinite(loss_scale) or loss_scale <= 0.0:
        raise ValueError("loss_scale must be finite and positive")
    return X, y, sample_weight / np.sum(sample_weight), loss, loss_scale


def _smooth_value_and_gradient(
    beta, X, y, normalized_weight, loss, huber_delta, loss_scale=1.0
):
    linear_predictor = X @ beta
    if loss == "logistic":
        losses = np.logaddexp(0.0, linear_predictor) - y * linear_predictor
        gradient = X.T @ (normalized_weight * (expit(linear_predictor) - y))
    elif loss == "huber":
        residual = y - linear_predictor
        absolute_residual = np.abs(residual)
        losses = np.where(
            absolute_residual <= huber_delta,
            0.5 * residual**2,
            huber_delta * (absolute_residual - 0.5 * huber_delta),
        )
        score = np.clip(residual, -huber_delta, huber_delta)
        gradient = -(X.T @ (normalized_weight * score))
    else:
        residual = linear_predictor - y
        losses = 0.5 * residual**2
        gradient = X.T @ (normalized_weight * residual)
    return (
        float(loss_scale * (normalized_weight @ losses)),
        np.asarray(loss_scale * gradient, dtype=float),
    )


def weighted_smooth_value_and_gradient(
    beta,
    X,
    y,
    sample_weight,
    *,
    loss,
    huber_delta=1.35,
    loss_scale=1.0,
):
    """Evaluate the normalized weighted smooth loss and its gradient."""
    X, y, normalized_weight, loss, loss_scale = _validated_problem(
        X, y, sample_weight, loss, huber_delta, loss_scale
    )
    beta = np.asarray(beta, dtype=float).reshape(-1)
    if beta.size != X.shape[1] or not np.all(np.isfinite(beta)):
        raise ValueError("beta must be a finite vector with one value per feature")
    return _smooth_value_and_gradient(
        beta, X, y, normalized_weight, loss, float(huber_delta), loss_scale
    )


def soft_threshold(values, threshold, penalized_mask=None):
    """Apply the L1 proximal operator only on selected coordinates."""
    values = np.asarray(values, dtype=float)
    threshold = float(threshold)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("threshold must be finite and nonnegative")
    if penalized_mask is None:
        penalized_mask = np.ones(values.shape, dtype=bool)
    else:
        penalized_mask = np.asarray(penalized_mask, dtype=bool)
        if penalized_mask.shape != values.shape:
            raise ValueError("penalized_mask must have the same shape as values")
    result = values.copy()
    selected = result[penalized_mask]
    result[penalized_mask] = np.sign(selected) * np.maximum(
        np.abs(selected) - threshold, 0.0
    )
    return result


def _penalized_mask(dimension, unpenalized_indices):
    try:
        indices = tuple(unpenalized_indices)
    except TypeError as error:
        raise ValueError("unpenalized_indices must contain valid feature indices") from error
    if any(
        isinstance(index, (bool, np.bool_))
        or not isinstance(index, (int, np.integer))
        or int(index) < 0
        or int(index) >= dimension
        for index in indices
    ) or len(set(map(int, indices))) != len(indices):
        raise ValueError("unpenalized_indices must contain unique valid feature indices")
    mask = np.ones(dimension, dtype=bool)
    mask[list(map(int, indices))] = False
    return mask


def _evaluate_loss_callable(M, prediction, truth):
    try:
        value = np.asarray(M(np.asarray(prediction), np.asarray(truth)), dtype=float)
    except Exception as error:
        raise ValueError(
            "could not evaluate M while identifying its WERM loss contract"
        ) from error
    if value.shape != np.asarray(prediction).shape or not np.all(np.isfinite(value)):
        raise ValueError("M must return one finite loss per observation")
    return value


def infer_loss(M):
    """Strictly identify a standard least-squares, Huber, or logistic loss."""
    prediction = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    at_zero = _evaluate_loss_callable(M, prediction, np.zeros_like(prediction))
    at_one = _evaluate_loss_callable(M, prediction, np.ones_like(prediction))
    nonzero = prediction != 0.0
    scales = -(at_one[nonzero] - at_zero[nonzero]) / prediction[nonzero]
    logistic_scale = float(np.median(scales))
    if logistic_scale > 0.0:
        expected_zero = logistic_scale * np.logaddexp(0.0, prediction)
        if np.allclose(at_zero, expected_zero, rtol=2e-6, atol=2e-8) and np.allclose(
            at_one, expected_zero - logistic_scale * prediction,
            rtol=2e-6, atol=2e-8,
        ):
            return LossSpec("logistic", logistic_scale, None)

    residual = np.array([-3.0, -1.1, -0.2, 0.0, 0.2, 1.1, 3.0])
    values = _evaluate_loss_callable(M, residual, np.zeros_like(residual))
    if abs(values[3]) > 1e-9 or not np.allclose(
        values[:3], values[:3:-1], rtol=2e-6, atol=2e-8
    ):
        raise ValueError(
            "M is not a supported least-squares, Huber, or logistic loss"
        )
    small = 1e-3
    small_value = _evaluate_loss_callable(
        M, np.array([small]), np.array([0.0])
    )[0]
    scale = float(small_value / (0.5 * small**2))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            "M is not a supported least-squares, Huber, or logistic loss"
        )
    if np.allclose(
        values, scale * 0.5 * residual**2, rtol=2e-5, atol=2e-8
    ):
        return LossSpec("least_squares", scale, None)

    large, difference = 100.0, 1e-2
    large_values = _evaluate_loss_callable(
        M, np.array([large - difference, large + difference]), np.zeros(2)
    )
    slope = float((large_values[1] - large_values[0]) / (2.0 * difference))
    delta = slope / scale
    absolute = np.abs(residual)
    expected = scale * np.where(
        absolute <= delta,
        0.5 * residual**2,
        delta * (absolute - 0.5 * delta),
    )
    if (
        not np.isfinite(delta)
        or delta <= 0.0
        or delta >= large
        or not np.allclose(values, expected, rtol=3e-5, atol=3e-8)
    ):
        raise ValueError(
            "M is not a supported least-squares, Huber, or logistic loss"
        )
    return LossSpec("huber", scale, float(delta))


def _evaluate_penalty_callable(P_func, beta):
    try:
        value = np.asarray(P_func(np.asarray(beta, dtype=float)), dtype=float)
    except Exception as error:
        raise ValueError(
            "could not evaluate P_func while identifying its penalty contract"
        ) from error
    if value.size != 1 or not np.isfinite(float(value.reshape(-1)[0])):
        raise ValueError("P_func must return one finite scalar")
    return float(value.reshape(-1)[0])


def infer_penalty(P_func, dimension):
    """Strictly identify a zero, uniform separable L1, or L2 penalty."""
    if isinstance(dimension, (bool, np.bool_)) or int(dimension) < 1:
        raise ValueError("dimension must be a positive integer")
    dimension = int(dimension)
    zero = np.zeros(dimension)
    baseline = _evaluate_penalty_callable(P_func, zero)
    unit = np.empty(dimension)
    doubled = np.empty(dimension)
    for index in range(dimension):
        basis = np.zeros(dimension)
        basis[index] = 1.0
        unit[index] = _evaluate_penalty_callable(P_func, basis) - baseline
        doubled[index] = _evaluate_penalty_callable(P_func, 2.0 * basis) - baseline
    tolerance = 1e-9 * max(1.0, np.max(np.abs(unit), initial=0.0))
    if np.any(unit < -tolerance):
        raise ValueError("P_func is not a nonnegative supported coefficient penalty")
    mask = unit > tolerance
    if not np.any(mask):
        return PenaltySpec("none", 0.0, np.zeros(dimension, dtype=bool))
    strengths = unit[mask]
    strength = float(np.mean(strengths))
    if not np.allclose(strengths, strength, rtol=2e-6, atol=2e-9):
        raise ValueError("P_func must use one common penalty strength")
    ratios = doubled[mask] / strengths
    if np.allclose(ratios, 2.0, rtol=2e-6, atol=2e-8):
        name = "l1"
    elif np.allclose(ratios, 4.0, rtol=2e-6, atol=2e-8):
        name = "l2"
    else:
        raise ValueError("P_func is not a supported L1 or L2 penalty")
    if not np.isclose(
        _evaluate_penalty_callable(P_func, mask.astype(float)) - baseline,
        strength * int(np.sum(mask)), rtol=2e-6, atol=2e-8,
    ):
        raise ValueError("P_func must be a separable L1 or L2 penalty")
    return PenaltySpec(name, strength, mask)


def _penalty_value(beta, penalty, strength, mask):
    selected = beta[mask]
    if penalty == "none":
        return 0.0
    if penalty == "l1":
        return float(strength * np.sum(np.abs(selected)))
    return float(strength * np.sum(selected**2))


def _proximal_map(values, step, penalty, strength, mask):
    if penalty == "none" or strength == 0.0:
        return np.asarray(values, dtype=float).copy()
    if penalty == "l1":
        return soft_threshold(values, step * strength, mask)
    result = np.asarray(values, dtype=float).copy()
    result[mask] /= 1.0 + 2.0 * step * strength
    return result


def _kkt_violation(gradient, beta, penalty, strength, mask):
    residual = np.asarray(gradient, dtype=float).copy()
    if penalty == "l2":
        residual[mask] += 2.0 * strength * beta[mask]
    elif penalty == "l1":
        penalized = np.flatnonzero(mask)
        nonzero = penalized[np.abs(beta[penalized]) > 1e-10]
        zero = penalized[np.abs(beta[penalized]) <= 1e-10]
        residual[nonzero] += strength * np.sign(beta[nonzero])
        residual[zero] = np.sign(residual[zero]) * np.maximum(
            np.abs(residual[zero]) - strength, 0.0
        )
    return float(np.max(np.abs(residual), initial=0.0))


def fit_weighted(
    X,
    y,
    sample_weight=None,
    *,
    loss,
    penalty="none",
    penalty_strength=0.0,
    huber_delta=1.35,
    loss_scale=1.0,
    unpenalized_indices=(),
    initial=None,
    initial_step=1.0,
    max_iter=10000,
    tol=1e-7,
    backtracking_factor=0.5,
    step_growth=1.05,
    max_backtracking=100,
):
    """Fit a supported normalized weighted convex objective.

    Least-squares scale one means ``0.5 * residual**2``; a package callable
    returning ``residual**2`` has scale two.  L2 means
    ``penalty_strength * sum(beta_j**2)``.  Least-squares none/L2 cases use an
    exact linear solve; all other cases use monotone FISTA with backtracking.
    """
    X_array = np.asarray(X)
    if X_array.ndim != 2:
        raise ValueError("X and y must have matching sample dimensions")
    if sample_weight is None:
        sample_weight = np.ones(X_array.shape[0], dtype=float)
    X, y, normalized_weight, loss, loss_scale = _validated_problem(
        X_array, y, sample_weight, loss, huber_delta, loss_scale
    )
    dimension = X.shape[1]
    mask = _penalized_mask(dimension, unpenalized_indices)
    penalty = str(penalty).lower()
    if penalty not in _SUPPORTED_PENALTIES:
        raise ValueError(f"penalty must be one of {sorted(_SUPPORTED_PENALTIES)}")
    strength = float(penalty_strength)
    if not np.isfinite(strength) or strength < 0.0:
        raise ValueError("penalty_strength must be finite and nonnegative")
    if penalty == "none" and strength != 0.0:
        raise ValueError("penalty_strength must be zero when penalty='none'")
    if isinstance(max_iter, (bool, np.bool_)) or int(max_iter) < 1:
        raise ValueError("max_iter must be a positive integer")
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be finite and positive")
    if not np.isfinite(initial_step) or initial_step <= 0.0:
        raise ValueError("initial_step must be finite and positive")
    if not 0.0 < float(backtracking_factor) < 1.0:
        raise ValueError("backtracking_factor must lie strictly between zero and one")
    if not np.isfinite(step_growth) or step_growth < 1.0:
        raise ValueError("step_growth must be finite and at least one")
    if isinstance(max_backtracking, (bool, np.bool_)) or int(max_backtracking) < 1:
        raise ValueError("max_backtracking must be a positive integer")

    if loss == "least_squares" and penalty in {"none", "l2"}:
        weighted_X = np.sqrt(normalized_weight)[:, None] * X
        weighted_y = np.sqrt(normalized_weight) * y
        design = np.sqrt(loss_scale) * weighted_X
        response = np.sqrt(loss_scale) * weighted_y
        gram = design.T @ design
        right_hand_side = design.T @ response
        if penalty == "l2" and strength > 0.0 and np.any(mask):
            gram[np.diag_indices(dimension)] += 2.0 * strength * mask
        try:
            factor = scipy.linalg.cho_factor(
                gram, lower=True, check_finite=False
            )
            coefficients = scipy.linalg.cho_solve(
                factor, right_hand_side, check_finite=False
            )
        except np.linalg.LinAlgError:
            # Preserve the minimum-norm solution on rank-deficient designs.
            if penalty == "l2" and strength > 0.0 and np.any(mask):
                regularizer = np.zeros((int(np.sum(mask)), dimension))
                regularizer[
                    np.arange(int(np.sum(mask))), np.flatnonzero(mask)
                ] = np.sqrt(2.0 * strength)
                design = np.vstack((design, regularizer))
                response = np.r_[response, np.zeros(regularizer.shape[0])]
            coefficients = np.linalg.lstsq(design, response, rcond=None)[0]
        smooth, gradient = _smooth_value_and_gradient(
            coefficients, X, y, normalized_weight, loss,
            float(huber_delta), loss_scale,
        )
        objective = smooth + _penalty_value(coefficients, penalty, strength, mask)
        kkt = _kkt_violation(gradient, coefficients, penalty, strength, mask)
        return WeightedProximalResult(
            coef=coefficients,
            objective=objective,
            converged=kkt <= max(tol, 1e-10),
            n_iter=0,
            kkt_violation=kkt,
            step_size=np.nan,
            backtracking_steps=0,
            solver=(
                "weighted-least-squares-exact"
                if penalty == "none" or strength == 0.0
                else "weighted-ridge-exact"
            ),
        )

    if initial is None:
        coefficients = np.zeros(dimension, dtype=float)
    else:
        coefficients = np.asarray(initial, dtype=float).reshape(-1).copy()
        if coefficients.size != dimension or not np.all(np.isfinite(coefficients)):
            raise ValueError("initial must be a finite vector with one value per feature")

    extrapolated = coefficients.copy()
    momentum = 1.0
    step_size = float(initial_step)
    # On separable logistic data the unpenalized MLE has no finite minimizer.
    # Local curvature then vanishes along separating directions, so an
    # unconstrained step-growth heuristic can reach floating-point overflow
    # even though ordinary unit-step gradient iterations remain stable.  The
    # initial step is therefore also a ceiling for this one objective class;
    # backtracking can still reduce it whenever the global curvature requires.
    step_ceiling = (
        float(initial_step)
        if loss == "logistic" and penalty == "none"
        else np.inf
    )
    total_backtracking = 0
    smooth, gradient = _smooth_value_and_gradient(
        coefficients, X, y, normalized_weight, loss,
        float(huber_delta), loss_scale,
    )
    objective = smooth + _penalty_value(coefficients, penalty, strength, mask)
    converged = False

    def proximal_backtracking(base, starting_step):
        nonlocal total_backtracking
        base_smooth, base_gradient = _smooth_value_and_gradient(
            base, X, y, normalized_weight, loss,
            float(huber_delta), loss_scale,
        )
        trial_step = starting_step
        for _ in range(int(max_backtracking)):
            candidate = _proximal_map(
                base - trial_step * base_gradient, trial_step,
                penalty, strength, mask,
            )
            candidate_smooth, _ = _smooth_value_and_gradient(
                candidate, X, y, normalized_weight, loss,
                float(huber_delta), loss_scale,
            )
            difference = candidate - base
            quadratic_bound = (
                base_smooth
                + float(base_gradient @ difference)
                + 0.5 * float(difference @ difference) / trial_step
            )
            numerical_slack = 1e-12 * max(1.0, abs(base_smooth))
            if candidate_smooth <= quadratic_bound + numerical_slack:
                return candidate, candidate_smooth, trial_step
            trial_step *= float(backtracking_factor)
            total_backtracking += 1
        raise RuntimeError("proximal-gradient backtracking failed to find a valid step")

    for iteration in range(1, int(max_iter) + 1):
        proposed_step = min(step_size * float(step_growth), step_ceiling)
        candidate, candidate_smooth, accepted_step = proximal_backtracking(
            extrapolated, proposed_step
        )
        candidate_objective = candidate_smooth + _penalty_value(
            candidate, penalty, strength, mask
        )

        # Monotone restart: discard momentum whenever extrapolation raises the
        # full objective, then take a guaranteed proximal-gradient step from x.
        if candidate_objective > objective + 1e-13 * max(1.0, abs(objective)):
            momentum = 1.0
            extrapolated = coefficients.copy()
            candidate, candidate_smooth, accepted_step = proximal_backtracking(
                extrapolated, min(proposed_step, step_size)
            )
            candidate_objective = candidate_smooth + _penalty_value(
                candidate, penalty, strength, mask
            )

        relative_step = np.linalg.norm(candidate - coefficients, ord=np.inf) / max(
            1.0, np.linalg.norm(coefficients, ord=np.inf)
        )
        new_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum**2))
        new_extrapolated = candidate + (
            (momentum - 1.0) / new_momentum
        ) * (candidate - coefficients)

        # Gradient-based adaptive restart prevents unproductive oscillation.
        # Use the point from which this step was taken. Comparing against the
        # newly formed extrapolate would always restart because that vector is
        # a positive multiple of (candidate - coefficients).
        if float((extrapolated - candidate) @ (candidate - coefficients)) > 0.0:
            new_momentum = 1.0
            new_extrapolated = candidate.copy()

        coefficients = candidate
        extrapolated = new_extrapolated
        momentum = new_momentum
        step_size = accepted_step
        smooth = candidate_smooth
        objective = candidate_objective

        if relative_step <= tol:
            _, gradient = _smooth_value_and_gradient(
                coefficients, X, y, normalized_weight, loss,
                float(huber_delta), loss_scale,
            )
            if _kkt_violation(
                gradient, coefficients, penalty, strength, mask
            ) <= tol:
                converged = True
                break

    smooth, gradient = _smooth_value_and_gradient(
        coefficients, X, y, normalized_weight, loss,
        float(huber_delta), loss_scale,
    )
    objective = smooth + _penalty_value(coefficients, penalty, strength, mask)
    kkt_violation = _kkt_violation(
        gradient, coefficients, penalty, strength, mask
    )
    if kkt_violation <= tol:
        converged = True

    return WeightedL1Result(
        coef=coefficients,
        objective=objective,
        converged=converged,
        n_iter=iteration,
        kkt_violation=kkt_violation,
        step_size=step_size,
        backtracking_steps=total_backtracking,
        solver="fista-backtracking",
    )


def fit_weighted_l1(
    X,
    y,
    sample_weight=None,
    *,
    loss,
    l1,
    huber_delta=1.35,
    unpenalized_indices=(),
    initial=None,
    initial_step=1.0,
    max_iter=10000,
    tol=1e-7,
    backtracking_factor=0.5,
    step_growth=1.05,
    max_backtracking=100,
):
    """Backward-compatible wrapper for the original weighted L1 API."""
    l1 = float(l1)
    if not np.isfinite(l1) or l1 < 0.0:
        raise ValueError("l1 must be finite and nonnegative")
    return fit_weighted(
        X, y, sample_weight,
        loss=loss,
        penalty="l1",
        penalty_strength=l1,
        huber_delta=huber_delta,
        unpenalized_indices=unpenalized_indices,
        initial=initial,
        initial_step=initial_step,
        max_iter=max_iter,
        tol=tol,
        backtracking_factor=backtracking_factor,
        step_growth=step_growth,
        max_backtracking=max_backtracking,
    )
