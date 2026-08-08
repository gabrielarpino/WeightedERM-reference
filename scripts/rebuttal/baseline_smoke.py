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
HISTORICAL_ROOT = Path(
    os.environ.get("WERM_HISTORICAL_ROOT", "/scratch/ga442/Weighted-ERM")
)


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
        brute_force=False,
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
    print("PARTITION SEARCH: exact DP production route")

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

    myocardial_candidates = (
        ROOT / "results/myocardial_infarction_posterior_exact_one_chgpt.pkl",
        HISTORICAL_ROOT
        / "results/myocardial_infarction_posterior_exact_one_chgpt.pkl",
    )
    myocardial_result = next(
        (path for path in myocardial_candidates if path.exists()), None
    )
    if myocardial_result is None:
        print(
            "Myocardial: BLOCKED "
            "(UCI data are remote and no archived posterior result was found in "
            "the reference or historical tree)"
        )
    else:
        with myocardial_result.open("rb") as handle:
            myocardial = pickle.load(handle)
        required = {"posterior", "η_arr_subsampled", "chgpt_est", "L"}
        missing = required.difference(myocardial)
        if missing:
            raise AssertionError(f"myocardial archive missing keys: {sorted(missing)}")
        posterior = np.asarray(myocardial["posterior"], dtype=float)
        eta = np.asarray(myocardial["η_arr_subsampled"], dtype=int)
        if posterior.ndim != 1 or eta.shape != (posterior.size, 1):
            raise AssertionError("myocardial posterior/configuration shape mismatch")
        if not np.all(np.isfinite(posterior)):
            raise AssertionError("myocardial posterior contains non-finite values")
        print(
            "Myocardial: SMOKE "
            f"(read archived result from {myocardial_result}; "
            f"posterior={posterior.shape}, eta={eta.shape}, "
            f"chgpt_est={np.asarray(myocardial['chgpt_est']).tolist()}; "
            "not a fresh reproduction)"
        )

    fred_pickle = ROOT / "FRED-MD data/fredmd_data.pkl"
    with fred_pickle.open("rb") as handle:
        fred = pickle.load(handle)
    X_fred = np.asarray(fred["X"])
    y_fred = np.asarray(fred["y"])
    if X_fred.shape[0] != y_fred.shape[0]:
        raise AssertionError("FRED-MD X/y row mismatch")
    fred_subset_n = 24
    fred_subset_p = 6
    fred_estimate, fred_objective = weighted_erm(
        X=X_fred[:fred_subset_n, :fred_subset_p],
        y=y_fred[:fred_subset_n],
        M=squared_loss,
        L=2,
        Delta_n=1,
        gamma=0.0,
        return_gof=True,
    )
    fred_estimate = np.asarray(fred_estimate, dtype=int)
    if fred_estimate.shape != (1,) or not np.isfinite(float(fred_objective)):
        raise AssertionError(
            "FRED-MD promoted-path subset returned an invalid estimate/objective"
        )

    historical_fred_available = all(
        (HISTORICAL_ROOT / relative_path).exists()
        for relative_path in (
            "FRED-MD data/fredmd.csv",
            "FRED-MD data/fredmd_data.RData",
        )
    )
    print(
        "FRED-MD: SMOKE "
        f"(committed pickle loaded, X={X_fred.shape}, y={y_fred.shape}; "
        f"historical raw/RData available={historical_fred_available}; "
        f"exact-DP subset n={fred_subset_n}, p={fred_subset_p}, "
        f"changepoints={fred_estimate.tolist()}; "
        "R competitors not run)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
