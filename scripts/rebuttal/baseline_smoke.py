"""Small deterministic checks for the Phase 1 reproducibility baseline."""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

# Keep numerical-library inner parallelism bounded before importing NumPy/JAX.
for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(variable, "1")

import jax.numpy as jnp
import numpy as np

from WERM.estimators import weighted_erm


ROOT = Path(__file__).resolve().parents[2]


def effective_workers(requested: int) -> tuple[int, int]:
    if requested < 1:
        raise ValueError("requested workers must be positive")
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:
        available = os.cpu_count() or 1
    return min(requested, 32, available), available


def run_weighted_erm_case(
    *,
    seed: int,
    n: int,
    p: int,
    beta: np.ndarray,
    changepoints: tuple[int, ...],
    loss,
    binary: bool = False,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    L = beta.shape[1]
    X = rng.normal(size=(n, p)) / np.sqrt(n)
    segment = np.searchsorted(np.asarray(changepoints), np.arange(n), side="right")
    linear_predictor = np.asarray(
        [X[i] @ beta[:, segment[i]] for i in range(n)]
    )
    if binary:
        probability = 1.0 / (1.0 + np.exp(-linear_predictor))
        y = rng.binomial(1, probability).astype(float)
        penalty = lambda coefficients: 0.01 * jnp.sum(coefficients**2)
    else:
        y = linear_predictor + 0.01 * rng.normal(size=n)
        penalty = lambda coefficients: 0.0

    estimate, objective = weighted_erm(
        X=X,
        y=y,
        M=loss,
        L=L,
        Delta_n=1,
        gamma=0.0,
        brute_force=True,
        return_gof=True,
        P_func=penalty,
    )
    estimate = np.asarray(estimate, dtype=int)
    if estimate.shape != (L - 1,):
        raise AssertionError(f"expected {L - 1} changepoints, got {estimate}")
    if np.any(np.diff(estimate) <= 0) or np.any((estimate <= 0) | (estimate >= n)):
        raise AssertionError(f"invalid changepoints: {estimate}")
    if not np.isfinite(float(objective)):
        raise AssertionError(f"non-finite objective: {objective}")
    return estimate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requested-workers", type=int, default=1)
    args = parser.parse_args()
    workers, available = effective_workers(args.requested_workers)
    print(
        "EFFECTIVE WORKERS: "
        f"{workers} (requested={args.requested_workers}, affinity={available}, cap=32)"
    )

    squared_loss = lambda prediction, outcome: jnp.square(prediction - outcome)
    logistic_loss = (
        lambda prediction, outcome: jnp.logaddexp(0.0, prediction)
        - outcome * prediction
    )

    m1 = run_weighted_erm_case(
        seed=101,
        n=18,
        p=4,
        beta=np.asarray(
            [[1.5, -1.0], [0.0, 0.8], [0.5, 0.0], [0.0, 0.0]]
        ),
        changepoints=(8,),
        loss=squared_loss,
    )
    print(f"M1: SMOKE (reduced sparse-linear call, changepoints={m1.tolist()})")

    m2 = run_weighted_erm_case(
        seed=202,
        n=20,
        p=5,
        beta=np.asarray(
            [
                [1.2, 1.2, -0.8],
                [0.0, 0.9, 0.9],
                [0.7, 0.0, 0.0],
                [0.0, -0.6, 0.0],
                [0.0, 0.0, 0.5],
            ]
        ),
        changepoints=(6, 13),
        loss=squared_loss,
    )
    print(f"M2: SMOKE (reduced differential-sparsity call, changepoints={m2.tolist()})")

    m3 = run_weighted_erm_case(
        seed=303,
        n=18,
        p=4,
        beta=np.asarray(
            [[0.8, -0.8], [0.0, 0.6], [0.4, 0.0], [0.0, -0.3]]
        ),
        changepoints=(9,),
        loss=logistic_loss,
        binary=True,
    )
    print(f"M3: SMOKE (reduced logistic call, changepoints={m3.tolist()})")

    myocardial_result = (
        ROOT / "results/myocardial_infarction_posterior_exact_one_chgpt.pkl"
    )
    if myocardial_result.exists():
        print("Myocardial: SMOKE (archived posterior result is locally readable)")
    else:
        print(
            "Myocardial: BLOCKED "
            "(UCI data are remote and the expected posterior result pickle is absent)"
        )

    fred_pickle = ROOT / "FRED-MD data/fredmd_data.pkl"
    with fred_pickle.open("rb") as handle:
        fred = pickle.load(handle)
    X_fred = np.asarray(fred["X"])
    y_fred = np.asarray(fred["y"])
    if X_fred.shape[0] != y_fred.shape[0]:
        raise AssertionError("FRED-MD X/y row mismatch")
    print(
        "FRED-MD: SMOKE "
        f"(committed pickle loaded, X={X_fred.shape}, y={y_fred.shape}; "
        "R competitors not run)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
