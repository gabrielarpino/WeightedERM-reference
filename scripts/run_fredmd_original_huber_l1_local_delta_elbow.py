"""Local Delta sensitivity around n/30 for the original Huber-L1 result."""

from pathlib import Path
import json
import os
import pickle

import jax.numpy as jnp
import numpy as np
import pandas as pd

import WERM.estimators
from WERM.experiment_utils import weighted_erm_joint_cv


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "FRED-MD_original-Huber-L1-local-delta-elbow"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
N_JOBS = min(36, os.cpu_count() or 1)
CONSTANTS = (0.15, 0.3, 0.5, 0.75, 1.0, 2.0)
SEGMENT_COUNTS = (1, 2, 3, 4, 5)
DELTAS = (1 / 30, 1 / 28, 1 / 25, 1 / 22, 1 / 20)
FIXED_CONSTANT = 0.3


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


def endpoint_distance_elbow(gofs):
    values = np.asarray(gofs, dtype=float)
    valid = np.flatnonzero(np.isfinite(values))
    if valid.size < 3:
        raise RuntimeError("At least three feasible GOF values are required")
    x = valid.astype(float)
    x = (x - x[0]) / (x[-1] - x[0])
    y = values[valid]
    y = (y - y[-1]) / (np.ptp(y) or 1.0)
    dx = x[-1] - x[0]
    dy = y[-1] - y[0]
    distances = np.abs(
        dy * x - dx * y + x[-1] * y[0] - y[-1] * x[0]
    ) / (np.hypot(dx, dy) or 1.0)
    distances[[0, -1]] = -np.inf
    return int(valid[np.argmax(distances)]) + 1, distances.tolist()


def coefficient_penalty(constant, p, n):
    coefficient_lambda = float(constant) * np.sqrt(2.0 * np.log(p) / n)
    return lambda beta: coefficient_lambda * jnp.sum(jnp.abs(beta))


def full_solution_curve(X, y, dates, delta_fraction, constants_by_segments):
    minimum_length = max(
        int(np.ceil(len(y) / 50.0)),
        int(delta_fraction * len(y)),
        1,
    )
    rows = []
    for segments in SEGMENT_COUNTS:
        constant = float(constants_by_segments[segments])
        cp, gof = WERM.estimators.weighted_erm(
            X=X,
            y=y,
            M=huber_loss,
            L=segments,
            Delta_n=minimum_length,
            gamma=0.0,
            L_is_upper_bound=False,
            brute_force=False,
            return_gof=True,
            P_func=coefficient_penalty(constant, X.shape[1], len(y)),
        )
        cp = np.asarray(cp, dtype=int).reshape(-1)
        rows.append(
            {
                "segments": segments,
                "constant": constant,
                "change_points": cp.tolist(),
                "dates": dates[cp].tolist(),
                "gof": float(gof),
            }
        )
    elbow_segments, distances = endpoint_distance_elbow(
        [row["gof"] for row in rows]
    )
    selected = rows[elbow_segments - 1]
    return {
        "selected_segments": elbow_segments,
        "selected_change_points": elbow_segments - 1,
        "selected_constant": selected["constant"],
        "change_points": selected["change_points"],
        "dates": selected["dates"],
        "gofs": [row["gof"] for row in rows],
        "distances": distances,
        "curve": rows,
    }


with (ROOT / "FRED-MD data" / "fredmd_data.pkl").open("rb") as handle:
    data = pickle.load(handle)
X = np.asarray(data["X"], dtype=float)
y = np.asarray(data["y"], dtype=float).reshape(-1)
dates = np.asarray(data["date"]).astype(str)
assert X.shape == (773, 114)
assert dates[0] == "1960-01" and dates[-1] == "2024-05"

records = []
for index, delta_fraction in enumerate(DELTAS, start=1):
    print(f"[{index}/{len(DELTAS)}] delta=1/{round(1/delta_fraction)}", flush=True)
    selected, cp, diagnostics = weighted_erm_joint_cv(
        X=X,
        y=y,
        M=huber_loss,
        M_scorer=absolute_loss,
        Lmax=5,
        delta_fraction=delta_fraction,
        l1_constants=CONSTANTS,
        num_folds=2,
        n_jobs=N_JOBS,
        standardize_y=False,
        selection_strategy="exhaustive",
        return_diagnostics=True,
        penalty_norm="l1",
        segment_counts=SEGMENT_COUNTS,
    )
    candidates = [tuple(value) for value in diagnostics["candidates"]]
    scores = np.asarray(diagnostics["scores"], dtype=float)
    score_lookup = {
        tuple(candidate): float(score)
        for candidate, score in zip(candidates, scores)
    }
    selected_total = score_lookup[tuple(selected)]
    zero_total = min(
        score_lookup[(1, float(constant))] for constant in CONSTANTS
    )
    cp = np.asarray(cp, dtype=int).reshape(-1)

    best_constant_by_segments = {
        segments: min(
            CONSTANTS,
            key=lambda constant: score_lookup[(segments, float(constant))],
        )
        for segments in SEGMENT_COUNTS
    }
    conditional_elbow = full_solution_curve(
        X, y, dates, delta_fraction, best_constant_by_segments
    )
    fixed_elbow = full_solution_curve(
        X,
        y,
        dates,
        delta_fraction,
        {segments: FIXED_CONSTANT for segments in SEGMENT_COUNTS},
    )
    record = {
        "delta_fraction": delta_fraction,
        "delta_label": f"n/{round(1/delta_fraction)}",
        "minimum_segment_length": max(
            int(np.ceil(len(y) / 50.0)),
            int(delta_fraction * len(y)),
            1,
        ),
        "cv_selected_segments": int(selected[0]),
        "cv_selected_change_points": int(selected[0] - 1),
        "cv_selected_constant": float(selected[1]),
        "cv_change_points": cp.tolist(),
        "cv_dates": dates[cp].tolist(),
        "cv_mae": selected_total / len(y),
        "zero_cp_cv_mae": zero_total / len(y),
        "cv_improvement_vs_zero_pct": (
            (zero_total - selected_total) / zero_total * 100
        ),
        "conditional_C_elbow": conditional_elbow,
        "fixed_C_0.3_elbow": fixed_elbow,
        "candidate_scores": [
            {
                "segments": int(candidate[0]),
                "change_points": int(candidate[0] - 1),
                "constant": float(candidate[1]),
                "cv_total": float(score),
                "cv_mean": float(score / len(y)),
            }
            for candidate, score in zip(candidates, scores)
        ],
    }
    records.append(record)
    with (RESULT_DIR / "checkpoint.json").open("w") as handle:
        json.dump(records, handle, indent=2)
    print(
        {
            "cv": [record["cv_selected_change_points"], record["cv_dates"]],
            "conditional_elbow": [
                conditional_elbow["selected_change_points"],
                conditional_elbow["dates"],
            ],
            "fixed_C_elbow": [
                fixed_elbow["selected_change_points"], fixed_elbow["dates"]
            ],
        },
        flush=True,
    )

rows = []
for record in records:
    conditional = record["conditional_C_elbow"]
    fixed = record["fixed_C_0.3_elbow"]
    rows.append(
        {
            "delta": record["delta_label"],
            "minimum_segment_length": record["minimum_segment_length"],
            "cv_change_points": record["cv_selected_change_points"],
            "cv_constant": record["cv_selected_constant"],
            "cv_dates": ";".join(record["cv_dates"]),
            "cv_mae": record["cv_mae"],
            "zero_cp_cv_mae": record["zero_cp_cv_mae"],
            "cv_improvement_vs_zero_pct": record[
                "cv_improvement_vs_zero_pct"
            ],
            "conditional_C_elbow_change_points": conditional[
                "selected_change_points"
            ],
            "conditional_C_elbow_constant": conditional["selected_constant"],
            "conditional_C_elbow_dates": ";".join(conditional["dates"]),
            "fixed_C_elbow_change_points": fixed["selected_change_points"],
            "fixed_C_elbow_dates": ";".join(fixed["dates"]),
        }
    )
pd.DataFrame(rows).to_csv(RESULT_DIR / "selections.csv", index=False)
with (RESULT_DIR / "summary.json").open("w") as handle:
    json.dump(
        {
            "dataset": {
                "n": len(y),
                "p": X.shape[1],
                "start": dates[0],
                "end": dates[-1],
            },
            "setting": {
                "loss": "Huber-eps1.35",
                "penalty": "l1",
                "intercept": False,
                "folds": 2,
                "constants": list(CONSTANTS),
                "segment_counts": list(SEGMENT_COUNTS),
                "score_loss": "absolute_loss",
            },
            "elbow_rule": "maximum distance to endpoint line",
            "records": records,
        },
        handle,
        indent=2,
    )
print({"complete": len(records), "result": str(RESULT_DIR)}, flush=True)
