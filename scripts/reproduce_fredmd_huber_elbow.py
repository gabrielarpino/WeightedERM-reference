"""Reproduce the original FRED-MD Huber elbow experiments.

The unpenalized curve fits each fixed segment count directly.  The L1 curve
implements the intended two-level procedure: for every fixed segment count,
CV selects the coefficient-penalty constant; the full-data fits at those
selected constants then form the curve used to select the elbow.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd

import WERM.estimators
from WERM.experiment_utils import weighted_erm_joint_cv


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "FRED-MD_original-Huber-elbow-reproduction"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
CONSTANTS = (0.00375, 0.075, 0.15, 0.3, 0.6, 1.2, 2.4)
SEGMENT_COUNTS = (1, 2, 3, 4, 5)
DELTA_FRACTION = 1 / 40
NUM_FOLDS = 5
N_JOBS = min(36, os.cpu_count() or 1)


def huber_loss(prediction, truth, epsilon=1.35):
    residual = prediction - truth
    absolute = jnp.abs(residual)
    return jnp.where(
        absolute <= epsilon,
        0.5 * jnp.square(residual),
        epsilon * (absolute - 0.5 * epsilon),
    )


def absolute_loss(prediction, truth):
    return jnp.abs(prediction - truth)


def original_slope_elbow(gofs):
    """Exact ``find_elbow`` rule from FRED-MD Real Data Experiment.ipynb."""
    values = np.asarray(gofs, dtype=float)
    if values.size < 3:
        raise ValueError("Need at least three points to compute an elbow")
    average_slopes = [
        ((values[i] - values[i - 1]) + (values[i + 1] - values[i])) / 2
        for i in range(1, values.size - 1)
    ]
    for j in range(1, len(average_slopes)):
        if abs(average_slopes[j]) < abs(average_slopes[j - 1]):
            return j + 2, average_slopes
    return int(values.size), average_slopes


def penalty(constant, p, n):
    coefficient_lambda = float(constant) * np.sqrt(2 * np.log(p) / n)
    return lambda beta: coefficient_lambda * jnp.sum(jnp.abs(beta))


def full_curve(X, y, dates, minimum_length, constants_by_segments):
    rows = []
    for segments in SEGMENT_COUNTS:
        constant = constants_by_segments[segments]
        penalty_function = (
            (lambda beta: jnp.array(0.0))
            if constant is None
            else penalty(constant, X.shape[1], len(y))
        )
        change_points, gof = WERM.estimators.weighted_erm(
            X=X,
            y=y,
            M=huber_loss,
            L=segments,
            Delta_n=minimum_length,
            gamma=0.0,
            brute_force=False,
            return_gof=True,
            P_func=penalty_function,
        )
        change_points = np.asarray(change_points, dtype=int).reshape(-1)
        rows.append(
            {
                "segments": segments,
                "change_points_count": segments - 1,
                "constant": constant,
                "change_point_indices": change_points.tolist(),
                "change_point_dates": dates[change_points].tolist(),
                "full_data_gof": float(gof),
            }
        )
    elbow_segments, average_slopes = original_slope_elbow(
        [row["full_data_gof"] for row in rows]
    )
    return rows, {
        "selected_segments": elbow_segments,
        "selected_change_points_count": elbow_segments - 1,
        "selected_change_point_indices": rows[elbow_segments - 1][
            "change_point_indices"
        ],
        "selected_change_point_dates": rows[elbow_segments - 1][
            "change_point_dates"
        ],
        "average_slopes": [float(value) for value in average_slopes],
    }


with (ROOT / "FRED-MD data" / "fredmd_data.pkl").open("rb") as handle:
    data = pickle.load(handle)
X = np.asarray(data["X"], dtype=float)
y = np.asarray(data["y"], dtype=float).reshape(-1)
dates = np.asarray(data["date"]).astype(str)
assert X.shape == (773, 114)
assert dates[[0, -1]].tolist() == ["1960-01", "2024-05"]
minimum_length = max(int(np.ceil(len(y) / 50)), int(len(y) * DELTA_FRACTION), 1)

print("Fitting fresh unpenalized Huber curve", flush=True)
unpenalized_curve, unpenalized_elbow = full_curve(
    X,
    y,
    dates,
    minimum_length,
    {segments: None for segments in SEGMENT_COUNTS},
)

print("Selecting one L1 constant by CV within each fixed L", flush=True)
_, _, diagnostics = weighted_erm_joint_cv(
    X=X,
    y=y,
    M=huber_loss,
    M_scorer=absolute_loss,
    Lmax=max(SEGMENT_COUNTS),
    delta_fraction=DELTA_FRACTION,
    l1_constants=CONSTANTS,
    num_folds=NUM_FOLDS,
    n_jobs=N_JOBS,
    standardize_y=False,
    selection_strategy="exhaustive",
    return_diagnostics=True,
    penalty_norm="l1",
    segment_counts=SEGMENT_COUNTS,
)
candidates = [tuple(candidate) for candidate in diagnostics["candidates"]]
scores = np.asarray(diagnostics["scores"], dtype=float)
score_lookup = dict(zip(candidates, scores))
constants_by_segments = {
    segments: min(
        CONSTANTS,
        key=lambda constant: score_lookup[(segments, float(constant))],
    )
    for segments in SEGMENT_COUNTS
}
penalized_curve, penalized_elbow = full_curve(
    X, y, dates, minimum_length, constants_by_segments
)
for row in penalized_curve:
    row["selected_constant_cv_mae"] = float(
        score_lookup[(row["segments"], float(row["constant"]))] / len(y)
    )

summary = {
    "data": {
        "shape": list(X.shape),
        "start": dates[0],
        "end": dates[-1],
    },
    "settings": {
        "fit_loss": "Huber epsilon=1.35",
        "heldout_score_loss": "absolute",
        "intercept": False,
        "delta_fraction": DELTA_FRACTION,
        "minimum_segment_length": minimum_length,
        "num_folds_for_l1_constant": NUM_FOLDS,
        "constants": list(CONSTANTS),
        "n_jobs": N_JOBS,
        "elbow_rule": "original average-of-two-connecting-slopes rule",
    },
    "unpenalized": {
        "curve": unpenalized_curve,
        "elbow": unpenalized_elbow,
    },
    "l1_penalty_constant_cv_within_fixed_L": {
        "curve": penalized_curve,
        "elbow": penalized_elbow,
        "constants_by_segments": {
            str(key): value for key, value in constants_by_segments.items()
        },
    },
}
with (RESULT_DIR / "summary.json").open("w") as handle:
    json.dump(summary, handle, indent=2)
pd.DataFrame(unpenalized_curve).to_csv(
    RESULT_DIR / "unpenalized_curve.csv", index=False
)
pd.DataFrame(penalized_curve).to_csv(RESULT_DIR / "l1_per_L_cv_curve.csv", index=False)
print(json.dumps({
    "unpenalized_elbow": unpenalized_elbow,
    "l1_per_L_cv_elbow": penalized_elbow,
    "constants_by_segments": constants_by_segments,
}, indent=2), flush=True)
