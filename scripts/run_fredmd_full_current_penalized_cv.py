"""Run penalized WERM-CV on the full 1960--2026 latest-vintage sample."""

from pathlib import Path
import hashlib
import json
import os
import pickle
import time

import jax.numpy as jnp
import numpy as np
import pandas as pd

import WERM.estimators
from WERM.experiment_utils import weighted_erm_joint_cv


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "FRED-MD_full-current-penalized-CV"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT = RESULT_DIR / "checkpoint.json"
N_JOBS = min(36, os.cpu_count() or 1)
CONSTANTS = (0.15, 0.3, 0.5, 0.75, 1.0, 2.0)
SEGMENT_COUNTS = (1, 2, 3, 4, 5)
SENSITIVITIES = ((2, 1 / 50), (2, 1 / 30), (5, 1 / 50), (5, 1 / 30))


def squared_loss(prediction, truth):
    return jnp.square(prediction - truth)


def absolute_loss(prediction, truth):
    return jnp.abs(prediction - truth)


def huber_loss(prediction, truth, epsilon=1.35):
    residual = prediction - truth
    absolute = jnp.abs(residual)
    return jnp.where(
        absolute <= epsilon,
        0.5 * jnp.square(residual),
        epsilon * (absolute - 0.5 * epsilon),
    )


def data_hash(X, y):
    return hashlib.blake2b(
        np.ascontiguousarray(X).tobytes()
        + np.ascontiguousarray(y).tobytes(),
        digest_size=16,
    ).hexdigest()


def endpoint_distance_elbow(gofs):
    """Existing penalized elbow: maximum distance to the endpoint line."""
    values = np.asarray(gofs, dtype=float)
    valid_indices = np.flatnonzero(np.isfinite(values))
    if valid_indices.size < 3:
        return int(valid_indices[np.argmin(values[valid_indices])]) + 1, []
    x = valid_indices.astype(float)
    x_normalized = (x - x[0]) / (x[-1] - x[0])
    valid_values = values[valid_indices]
    y_normalized = (valid_values - valid_values[-1]) / (
        np.ptp(valid_values) or 1.0
    )
    dx = x_normalized[-1] - x_normalized[0]
    dy = y_normalized[-1] - y_normalized[0]
    distances = np.abs(
        dy * x_normalized
        - dx * y_normalized
        + x_normalized[-1] * y_normalized[0]
        - y_normalized[-1] * x_normalized[0]
    ) / (np.hypot(dx, dy) or 1.0)
    distances[[0, -1]] = -np.inf
    full_distances = np.full(len(values), np.nan)
    full_distances[valid_indices] = distances
    return int(valid_indices[np.argmax(distances)]) + 1, full_distances.tolist()


def location_hausdorff(first, second):
    first = np.asarray(first, dtype=int)
    second = np.asarray(second, dtype=int)
    if first.size == 0 or second.size == 0:
        return float("nan")
    return float(
        max(
            np.max(np.min(np.abs(first[:, None] - second[None, :]), axis=1)),
            np.max(np.min(np.abs(second[:, None] - first[None, :]), axis=1)),
        )
    )


with (ROOT / "FRED-MD data" / "fredmd_2026-07_mcscan.pkl").open("rb") as handle:
    data = pickle.load(handle)

X_base = np.asarray(data["X"], dtype=float)
y = np.asarray(data["y"], dtype=float).reshape(-1)
dates = np.asarray(data["date"]).astype(str)
assert X_base.shape == (797, 92)
assert dates[0] == "1960-01" and dates[-1] == "2026-05"

if CHECKPOINT.exists():
    with CHECKPOINT.open() as handle:
        records = json.load(handle)
else:
    records = []
completed = {record["key"] for record in records}

configs = (
    ("least-squares", squared_loss, "l1"),
    ("least-squares", squared_loss, "l2"),
    ("Huber-eps1.35", huber_loss, "l1"),
    ("Huber-eps1.35", huber_loss, "l2"),
)
total = 2 * len(SENSITIVITIES) * len(configs)
surface = 0
print(
    {
        "n_jobs": N_JOBS,
        "surfaces": total,
        "resuming_complete": len(records),
        "n": len(y),
        "p": X_base.shape[1],
        "start": dates[0],
        "end": dates[-1],
    },
    flush=True,
)

for include_intercept in (False, True):
    X = (
        np.column_stack((np.ones(len(y)), X_base))
        if include_intercept
        else X_base
    )
    unpenalized_indices = (0,) if include_intercept else ()
    for num_folds, delta_fraction in SENSITIVITIES:
        for loss_label, base_loss, penalty_norm in configs:
            surface += 1
            key = "|".join(
                (
                    f"intercept={int(include_intercept)}",
                    loss_label,
                    penalty_norm,
                    f"folds={num_folds}",
                    f"delta={delta_fraction:.12g}",
                )
            )
            if key in completed:
                continue
            print(f"[{surface}/{total}] {key}", flush=True)
            started = time.perf_counter()
            selected, cp, diagnostics = weighted_erm_joint_cv(
                X=X,
                y=y,
                M=base_loss,
                M_scorer=absolute_loss,
                Lmax=5,
                delta_fraction=delta_fraction,
                l1_constants=CONSTANTS,
                num_folds=num_folds,
                n_jobs=N_JOBS,
                standardize_y=False,
                selection_strategy="exhaustive",
                return_diagnostics=True,
                penalty_norm=penalty_norm,
                segment_counts=SEGMENT_COUNTS,
                unpenalized_feature_indices=unpenalized_indices,
            )
            candidates = [tuple(value) for value in diagnostics["candidates"]]
            scores = np.asarray(diagnostics["scores"], dtype=float)
            selected_total = float(scores[candidates.index(tuple(selected))])
            zero_total = min(
                float(score)
                for candidate, score in zip(candidates, scores)
                if candidate[0] == 1 and np.isfinite(score)
            )
            cp = np.asarray(cp, dtype=int).reshape(-1)
            penalty_mask = np.ones(X.shape[1], dtype=float)
            penalty_mask[list(unpenalized_indices)] = 0.0
            score_by_candidate = {
                tuple(candidate): float(score)
                for candidate, score in zip(candidates, scores)
            }
            minimum_length = max(
                int(np.ceil(len(y) / 50.0)),
                int(delta_fraction * len(y)),
                1,
            )
            by_segment_count = {}
            for num_segments in SEGMENT_COUNTS:
                best_constant = min(
                    CONSTANTS,
                    key=lambda constant: score_by_candidate[
                        (num_segments, float(constant))
                    ],
                )
                coefficient_lambda = best_constant * np.sqrt(
                    2.0 * np.log(X.shape[1]) / len(y)
                )
                if penalty_norm == "l1":
                    penalty = lambda beta, value=coefficient_lambda: (
                        value
                        * jnp.sum(
                            jnp.abs(beta * jnp.asarray(penalty_mask))
                        )
                    )
                else:
                    penalty = lambda beta, value=coefficient_lambda: (
                        value
                        * jnp.sum(
                            jnp.square(beta * jnp.asarray(penalty_mask))
                        )
                    )
                elbow_fit_cp, elbow_gof = WERM.estimators.weighted_erm(
                    X=X,
                    y=y,
                    M=base_loss,
                    L=num_segments,
                    Delta_n=minimum_length,
                    gamma=0.0,
                    L_is_upper_bound=False,
                    brute_force=False,
                    return_gof=True,
                    P_func=penalty,
                )
                elbow_fit_cp = np.asarray(
                    elbow_fit_cp, dtype=int
                ).reshape(-1)
                by_segment_count[num_segments] = {
                    "constant": float(best_constant),
                    "cv_total": score_by_candidate[
                        (num_segments, float(best_constant))
                    ],
                    "change_points": elbow_fit_cp.tolist(),
                    "dates": dates[elbow_fit_cp].tolist(),
                    "gof": float(elbow_gof),
                }
            elbow_segments, elbow_distances = endpoint_distance_elbow(
                [
                    by_segment_count[num_segments]["gof"]
                    for num_segments in SEGMENT_COUNTS
                ]
            )
            elbow_result = by_segment_count[elbow_segments]
            elbow_cp = elbow_result["change_points"]
            record = {
                "key": key,
                "intercept": include_intercept,
                "intercept_penalized": False,
                "loss": loss_label,
                "penalty": penalty_norm,
                "folds": num_folds,
                "delta_fraction": delta_fraction,
                "selected_constant": float(selected[1]),
                "selected_change_points": int(selected[0] - 1),
                "estimated_indices": cp.tolist(),
                "estimated_dates": dates[cp].tolist(),
                "cv_mae": selected_total / len(y),
                "zero_cp_cv_mae": zero_total / len(y),
                "improvement_vs_zero_pct": (
                    (zero_total - selected_total) / zero_total * 100
                ),
                "elbow_rule": "maximum distance to endpoint line",
                "elbow_selected_segments": int(elbow_segments),
                "elbow_selected_change_points": int(elbow_segments - 1),
                "elbow_selected_constant": elbow_result["constant"],
                "elbow_change_points": elbow_cp,
                "elbow_dates": elbow_result["dates"],
                "elbow_gofs": [
                    by_segment_count[num_segments]["gof"]
                    for num_segments in SEGMENT_COUNTS
                ],
                "elbow_distances": elbow_distances,
                "by_segment_count": by_segment_count,
                "both_nonzero": bool(cp.size and len(elbow_cp)),
                "count_match": bool(cp.size == len(elbow_cp)),
                "location_hausdorff_months": location_hausdorff(
                    cp, elbow_cp
                ),
                "elapsed_seconds": time.perf_counter() - started,
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
            completed.add(key)
            with CHECKPOINT.open("w") as handle:
                json.dump(records, handle, indent=2)
            print(
                {
                    "cp": record["selected_change_points"],
                    "C": record["selected_constant"],
                    "dates": record["estimated_dates"],
                    "cv_mae": record["cv_mae"],
                    "gain_pct": record["improvement_vs_zero_pct"],
                    "elbow_cp": record["elbow_selected_change_points"],
                    "elbow_C": record["elbow_selected_constant"],
                    "elbow_dates": record["elbow_dates"],
                    "count_match": record["count_match"],
                    "seconds": record["elapsed_seconds"],
                },
                flush=True,
            )

rows = [
    {
        key: value
        for key, value in record.items()
        if key
        not in {
            "candidate_scores",
            "by_segment_count",
            "elbow_gofs",
            "elbow_distances",
        }
    }
    for record in records
]
table = pd.DataFrame(rows).sort_values(
    ["intercept", "penalty", "loss", "folds", "delta_fraction"]
)
table["estimated_dates"] = table["estimated_dates"].map(";".join)
table["elbow_dates"] = table["elbow_dates"].map(";".join)
table.to_csv(RESULT_DIR / "all_selections.csv", index=False)
table.loc[table["selected_change_points"] > 0].to_csv(
    RESULT_DIR / "nonzero_selections.csv", index=False
)
table.loc[table["selected_change_points"].isin([2, 3])].to_csv(
    RESULT_DIR / "two_or_three_change_points.csv", index=False
)
table.loc[table["both_nonzero"]].to_csv(
    RESULT_DIR / "both_nonzero_cv_and_elbow.csv", index=False
)
table.loc[table["both_nonzero"] & table["count_match"]].sort_values(
    ["location_hausdorff_months", "improvement_vs_zero_pct"],
    ascending=[True, False],
).to_csv(RESULT_DIR / "coherent_count_matches.csv", index=False)

summary = {
    "dataset": {
        "vintage": "2026-07",
        "n": len(y),
        "p": X_base.shape[1],
        "start": dates[0],
        "end": dates[-1],
        "hash": data_hash(X_base, y),
    },
    "loss_contract": {
        "segmentation": "base loss",
        "hard_refits": "base loss",
        "heldout_scoring_only": "absolute_loss",
        "intercept": "leading ones column excluded from L1/L2 penalty",
    },
    "elbow": {
        "rule": "maximum distance to endpoint line",
        "penalty_selection": "best CV constant chosen separately for each segment count",
        "full_data_refit": "same base loss and coefficient penalty",
    },
    "grid": {
        "constants": list(CONSTANTS),
        "segment_counts": list(SEGMENT_COUNTS),
        "sensitivities": [
            {"folds": folds, "delta_fraction": delta}
            for folds, delta in SENSITIVITIES
        ],
        "losses": ["least-squares", "Huber-eps1.35"],
        "penalties": ["l1", "l2"],
        "n_jobs": N_JOBS,
    },
    "records": records,
}
with (RESULT_DIR / "summary.json").open("w") as handle:
    json.dump(summary, handle, indent=2)

print(
    {
        "complete": len(records),
        "nonzero": int((table["selected_change_points"] > 0).sum()),
        "two_or_three": int(
            table["selected_change_points"].isin([2, 3]).sum()
        ),
        "both_nonzero": int(table["both_nonzero"].sum()),
        "coherent_count_matches": int(
            (table["both_nonzero"] & table["count_match"]).sum()
        ),
        "summary": str(RESULT_DIR / "summary.json"),
    },
    flush=True,
)
