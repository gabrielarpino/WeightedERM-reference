"""Compare unpenalized WERM-CV and the existing GOF elbow at high Delta.

The three samples are the period variants requested for the FRED-MD audit.
Segmentation and hard refits use the base loss; held-out scoring alone uses
absolute loss.  The elbow rule exactly reproduces the existing notebook rule.
"""

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
RESULT_DIR = ROOT / "results" / "FRED-MD_high-delta-unpenalized-CV-vs-elbow"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT = RESULT_DIR / "checkpoint.json"
N_JOBS = min(36, os.cpu_count() or 1)
SEGMENT_COUNTS = (1, 2, 3, 4, 5)
DELTAS = (1 / 20, 1 / 10, 1 / 5)
FOLDS = (2, 5)
HUBER_EPSILONS = (0.5, 0.75, 1.0, 1.35, 2.0, 3.0)


def squared_loss(prediction, truth):
    return jnp.square(prediction - truth)


def absolute_loss(prediction, truth):
    return jnp.abs(prediction - truth)


def make_huber(epsilon):
    epsilon = float(epsilon)

    def loss(prediction, truth):
        residual = prediction - truth
        absolute = jnp.abs(residual)
        return jnp.where(
            absolute <= epsilon,
            0.5 * jnp.square(residual),
            epsilon * (absolute - 0.5 * epsilon),
        )

    return loss


def restrict(data, start, end=None):
    dates = np.asarray(data["date"]).astype(str)
    keep = dates >= start
    if end is not None:
        keep &= dates <= end
    return {
        "X": np.asarray(data["X"], dtype=float)[keep],
        "y": np.asarray(data["y"], dtype=float).reshape(-1)[keep],
        "date": dates[keep],
    }


def data_hash(X, y):
    return hashlib.blake2b(
        np.ascontiguousarray(X).tobytes()
        + np.ascontiguousarray(y).tobytes(),
        digest_size=16,
    ).hexdigest()


def existing_elbow(gofs):
    """Reproduce `_fs_elbow` from FRED-MD Real Data Experiment.ipynb."""
    values = np.asarray(gofs, dtype=float)
    valid_indices = np.flatnonzero(np.isfinite(values))
    if len(valid_indices) < 3:
        return None, float("nan")
    valid_values = values[valid_indices]
    average_adjacent_differences = []
    for index in range(1, len(valid_values) - 1):
        average_adjacent_differences.append(
            (
                (valid_values[index] - valid_values[index - 1])
                + (valid_values[index + 1] - valid_values[index])
            )
            / 2.0
        )
    for index in range(1, len(average_adjacent_differences)):
        if abs(average_adjacent_differences[index]) < abs(
            average_adjacent_differences[index - 1]
        ):
            position = index + 1
            return (
                int(valid_indices[position] + 1),
                float(values[valid_indices[position]]),
            )
    return int(valid_indices[-1] + 1), float(values[valid_indices[-1]])


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


with (ROOT / "FRED-MD data" / "fredmd_data.pkl").open("rb") as handle:
    historical = pickle.load(handle)
with (ROOT / "FRED-MD data" / "fredmd_2026-07_mcscan.pkl").open("rb") as handle:
    current = pickle.load(handle)

DATASETS = {
    "DPDU_2000-01_to_2022-12": restrict(historical, "2000-01", "2022-12"),
    "McScan_requested_1970-01_to_2024-01": restrict(
        historical, "1970-01", "2024-01"
    ),
    "Our_latest_1970-01_to_2026-05": restrict(current, "1970-01", None),
}
LOSSES = [("least-squares", None, squared_loss)] + [
    (f"Huber-eps{epsilon:g}", epsilon, make_huber(epsilon))
    for epsilon in HUBER_EPSILONS
]

if CHECKPOINT.exists():
    with CHECKPOINT.open() as handle:
        records = json.load(handle)
else:
    records = []
completed = {record["key"] for record in records}
elbow_cache = {
    (
        record["dataset"],
        record["intercept"],
        record["loss"],
        record["delta_fraction"],
    ): {
        "elbow_selected_segments": record["elbow_selected_segments"],
        "elbow_change_points": record["elbow_change_points"],
        "elbow_dates": record["elbow_dates"],
        "elbow_gofs": record["elbow_gofs"],
    }
    for record in records
}

total = len(DATASETS) * 2 * len(DELTAS) * len(LOSSES) * len(FOLDS)
surface = 0
print(
    {
        "n_jobs": N_JOBS,
        "surfaces": total,
        "resuming_complete": len(records),
        "datasets": {
            name: {
                "n": len(data["y"]),
                "p": data["X"].shape[1],
                "start": data["date"][0],
                "end": data["date"][-1],
            }
            for name, data in DATASETS.items()
        },
    },
    flush=True,
)

for dataset_name, data in DATASETS.items():
    X_base = data["X"]
    y = data["y"]
    dates = data["date"]
    for include_intercept in (False, True):
        X = (
            np.column_stack((np.ones(len(y)), X_base))
            if include_intercept
            else X_base
        )
        unpenalized_indices = (0,) if include_intercept else ()
        for delta_fraction in DELTAS:
            for loss_label, epsilon, base_loss in LOSSES:
                elbow_key = (
                    dataset_name,
                    include_intercept,
                    loss_label,
                    delta_fraction,
                )
                if elbow_key not in elbow_cache:
                    elbow_gofs = []
                    elbow_cps = []
                    minimum_length = max(
                        int(np.ceil(len(y) / 50.0)),
                        int(delta_fraction * len(y)),
                        1,
                    )
                    for num_segments in SEGMENT_COUNTS:
                        try:
                            cp, gof = WERM.estimators.weighted_erm(
                                X=X,
                                y=y,
                                M=base_loss,
                                L=num_segments,
                                Delta_n=minimum_length,
                                gamma=0.0,
                                L_is_upper_bound=False,
                                brute_force=False,
                                return_gof=True,
                                P_func=lambda beta: jnp.array(0.0),
                            )
                            elbow_gofs.append(float(gof))
                            elbow_cps.append(
                                np.asarray(cp, dtype=int).reshape(-1).tolist()
                            )
                        except AssertionError:
                            elbow_gofs.append(float("nan"))
                            elbow_cps.append([])
                    elbow_segments, _ = existing_elbow(elbow_gofs)
                    elbow_cp = (
                        []
                        if elbow_segments is None
                        else elbow_cps[elbow_segments - 1]
                    )
                    elbow_cache[elbow_key] = {
                        "elbow_selected_segments": elbow_segments,
                        "elbow_change_points": elbow_cp,
                        "elbow_dates": dates[elbow_cp].tolist(),
                        "elbow_gofs": elbow_gofs,
                    }
                elbow = elbow_cache[elbow_key]

                for num_folds in FOLDS:
                    surface += 1
                    key = "|".join(
                        (
                            dataset_name,
                            f"intercept={int(include_intercept)}",
                            loss_label,
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
                        l1_constants=(1.0,),
                        num_folds=num_folds,
                        n_jobs=N_JOBS,
                        standardize_y=False,
                        selection_strategy="exhaustive",
                        return_diagnostics=True,
                        penalty_norm="none",
                        segment_counts=SEGMENT_COUNTS,
                        unpenalized_feature_indices=unpenalized_indices,
                    )
                    candidates = [
                        tuple(value) for value in diagnostics["candidates"]
                    ]
                    scores = np.asarray(diagnostics["scores"], dtype=float)
                    selected_total = float(
                        scores[candidates.index(tuple(selected))]
                    )
                    zero_total = float(scores[candidates.index((1, 0.0))])
                    cp = np.asarray(cp, dtype=int).reshape(-1)
                    elbow_cp = elbow["elbow_change_points"]
                    record = {
                        "key": key,
                        "dataset": dataset_name,
                        "n": len(y),
                        "p_without_intercept": X_base.shape[1],
                        "sample_start": dates[0],
                        "sample_end": dates[-1],
                        "data_hash": data_hash(X_base, y),
                        "intercept": include_intercept,
                        "loss": loss_label,
                        "epsilon": epsilon,
                        "folds": num_folds,
                        "delta_fraction": delta_fraction,
                        "cv_selected_segments": int(selected[0]),
                        "cv_selected_change_points": int(selected[0] - 1),
                        "cv_change_points": cp.tolist(),
                        "cv_dates": dates[cp].tolist(),
                        "cv_mae": selected_total / len(y),
                        "zero_cp_cv_mae": zero_total / len(y),
                        "cv_improvement_vs_zero_pct": (
                            (zero_total - selected_total) / zero_total * 100
                        ),
                        **elbow,
                        "elbow_selected_change_points": (
                            None
                            if elbow["elbow_selected_segments"] is None
                            else elbow["elbow_selected_segments"] - 1
                        ),
                        "both_nonzero": bool(cp.size and len(elbow_cp)),
                        "count_match": bool(cp.size == len(elbow_cp)),
                        "location_hausdorff_months": location_hausdorff(
                            cp, elbow_cp
                        ),
                        "candidate_scores": [
                            {
                                "segments": int(candidate[0]),
                                "change_points": int(candidate[0] - 1),
                                "cv_total": float(score),
                                "cv_mean": float(score / len(y)),
                            }
                            for candidate, score in zip(candidates, scores)
                        ],
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                    records.append(record)
                    completed.add(key)
                    with CHECKPOINT.open("w") as handle:
                        json.dump(records, handle, indent=2)
                    print(
                        {
                            "cv_cp": record["cv_selected_change_points"],
                            "cv_dates": record["cv_dates"],
                            "elbow_cp": record[
                                "elbow_selected_change_points"
                            ],
                            "elbow_dates": record["elbow_dates"],
                            "count_match": record["count_match"],
                            "hausdorff_months": record[
                                "location_hausdorff_months"
                            ],
                            "seconds": record["elapsed_seconds"],
                        },
                        flush=True,
                    )

rows = [
    {key: value for key, value in record.items() if key not in {"candidate_scores", "elbow_gofs"}}
    for record in records
]
table = pd.DataFrame(rows).sort_values(
    ["dataset", "intercept", "loss", "delta_fraction", "folds"]
)
for column in ("cv_dates", "elbow_dates"):
    table[column] = table[column].map(";".join)
table.to_csv(RESULT_DIR / "all_selections.csv", index=False)
both = table.loc[table["both_nonzero"]].copy()
both.to_csv(RESULT_DIR / "both_nonzero.csv", index=False)
coherent = both.loc[both["count_match"]].sort_values(
    [
        "location_hausdorff_months",
        "cv_improvement_vs_zero_pct",
    ],
    ascending=[True, False],
)
coherent.to_csv(RESULT_DIR / "coherent_count_matches.csv", index=False)

joint_rows = []
for group_key, group in table.loc[table["epsilon"].notna()].groupby(
    ["dataset", "intercept", "folds", "delta_fraction"]
):
    best = group.loc[group["cv_mae"].idxmin()]
    joint_rows.append(
        {
            "dataset": group_key[0],
            "intercept": bool(group_key[1]),
            "folds": int(group_key[2]),
            "delta_fraction": float(group_key[3]),
            "selected_epsilon": float(best["epsilon"]),
            "cv_selected_change_points": int(
                best["cv_selected_change_points"]
            ),
            "cv_dates": best["cv_dates"],
            "elbow_selected_change_points": int(
                best["elbow_selected_change_points"]
            ),
            "elbow_dates": best["elbow_dates"],
            "count_match": bool(best["count_match"]),
            "location_hausdorff_months": float(
                best["location_hausdorff_months"]
            ),
            "cv_mae": float(best["cv_mae"]),
            "cv_improvement_vs_zero_pct": float(
                best["cv_improvement_vs_zero_pct"]
            ),
        }
    )
pd.DataFrame(joint_rows).to_csv(
    RESULT_DIR / "joint_huber_epsilon_selections.csv", index=False
)

summary = {
    "criterion": (
        "unpenalized exhaustive CV versus the existing fold-independent GOF elbow"
    ),
    "loss_contract": {
        "segmentation": "base loss",
        "hard_refits": "base loss",
        "heldout_scoring_only": "absolute_loss",
    },
    "elbow_rule": "existing notebook `_fs_elbow` rule",
    "grid": {
        "deltas": list(DELTAS),
        "folds": list(FOLDS),
        "segment_counts": list(SEGMENT_COUNTS),
        "Huber_epsilons": list(HUBER_EPSILONS),
        "intercept": [False, True],
        "n_jobs": N_JOBS,
    },
    "records": records,
    "joint_huber_epsilon_selections": joint_rows,
}
with (RESULT_DIR / "summary.json").open("w") as handle:
    json.dump(summary, handle, indent=2)

print(
    {
        "complete": len(records),
        "both_nonzero": len(both),
        "coherent_count_matches": len(coherent),
        "summary": str(RESULT_DIR / "summary.json"),
    },
    flush=True,
)
