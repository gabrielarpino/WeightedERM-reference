"""Run FRED-MD period and intercept sensitivity experiments.

The search holds the corrected CV loss contract fixed: segmentation and hard
segment refits use the base loss, while absolute loss is held-out scoring only.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import itertools
import json
import os
import pickle
import time

import jax.numpy as jnp
import numpy as np
import pandas as pd

from scripts.fredmd_preprocessing import mcscan_design_from_csv
from WERM.experiment_utils import weighted_erm_joint_cv


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "FRED-MD_period-intercept-CV"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT = RESULT_DIR / "checkpoint.json"
SUMMARY_PATH = RESULT_DIR / "summary.json"
CURRENT_PICKLE = ROOT / "FRED-MD data" / "fredmd_2026-07_mcscan.pkl"
N_JOBS = min(36, os.cpu_count() or 1)

CONSTANTS = (0.15, 0.3, 0.5, 0.75, 1.0, 2.0)
SEGMENT_COUNTS = (1, 2, 3, 4, 5)
SENSITIVITIES = (
    {"num_folds": 2, "delta_fraction": 1 / 50},
    {"num_folds": 2, "delta_fraction": 1 / 30},
    {"num_folds": 5, "delta_fraction": 1 / 50},
    {"num_folds": 5, "delta_fraction": 1 / 30},
)
UNPENALIZED_HUBER_EPSILONS = (0.5, 0.75, 1.0, 1.35, 2.0, 3.0)


def squared_loss(prediction, truth):
    return jnp.square(prediction - truth)


def absolute_loss(prediction, truth):
    return jnp.abs(prediction - truth)


def make_huber(epsilon):
    epsilon = float(epsilon)

    def huber_loss(prediction, truth):
        residual = prediction - truth
        absolute = jnp.abs(residual)
        return jnp.where(
            absolute <= epsilon,
            0.5 * jnp.square(residual),
            epsilon * (absolute - 0.5 * epsilon),
        )

    return huber_loss


def data_hash(X, y):
    return hashlib.blake2b(
        np.ascontiguousarray(X).tobytes() + np.ascontiguousarray(y).tobytes(),
        digest_size=16,
    ).hexdigest()


def restrict(data, start, end=None):
    dates = np.asarray(data["date"]).astype(str)
    mask = dates >= start
    if end is not None:
        mask &= dates <= end
    return {
        "X": np.asarray(data["X"], dtype=float)[mask],
        "y": np.asarray(data["y"], dtype=float).reshape(-1)[mask],
        "date": dates[mask],
    }


with (ROOT / "FRED-MD data" / "fredmd_data.pkl").open("rb") as handle:
    historical = pickle.load(handle)

if CURRENT_PICKLE.exists():
    with CURRENT_PICKLE.open("rb") as handle:
        current = pickle.load(handle)
else:
    current = mcscan_design_from_csv(ROOT / "FRED-MD data" / "2026-07-md.csv")
    with CURRENT_PICKLE.open("wb") as handle:
        pickle.dump(current, handle)

DATASETS = {
    "DPDU_2000-01_to_2022-12": restrict(historical, "2000-01", "2022-12"),
    "McScan_requested_1970-01_to_2024-01": restrict(
        historical, "1970-01", "2024-01"
    ),
    "Our_latest_1970-01_to_2026-05": restrict(current, "1970-01", None),
    "Prior_full_sample_1960-01_to_2024-05": restrict(
        historical, "1960-01", "2024-05"
    ),
}
assert DATASETS["Our_latest_1970-01_to_2026-05"]["date"][-1] == "2026-05"

CONFIGS = []
for sensitivity in SENSITIVITIES:
    CONFIGS.append(
        {
            **sensitivity,
            "loss": "least-squares",
            "epsilon": None,
            "penalty_norm": "none",
        }
    )
    for epsilon in UNPENALIZED_HUBER_EPSILONS:
        CONFIGS.append(
            {
                **sensitivity,
                "loss": "Huber",
                "epsilon": epsilon,
                "penalty_norm": "none",
            }
        )
    for loss, penalty_norm in itertools.product(
        ("least-squares", "Huber"), ("l1", "l2")
    ):
        CONFIGS.append(
            {
                **sensitivity,
                "loss": loss,
                "epsilon": 1.35 if loss == "Huber" else None,
                "penalty_norm": penalty_norm,
            }
        )

assert len(CONFIGS) == 44


def record_key(dataset_name, include_intercept, config):
    epsilon = "none" if config["epsilon"] is None else f'{config["epsilon"]:g}'
    return "|".join(
        [
            dataset_name,
            f"intercept={int(include_intercept)}",
            config["loss"],
            f"eps={epsilon}",
            config["penalty_norm"],
            f'folds={config["num_folds"]}',
            f'delta={config["delta_fraction"]:.12g}',
        ]
    )


if CHECKPOINT.exists():
    with CHECKPOINT.open() as handle:
        records = json.load(handle)
else:
    records = []
completed = {record["record_key"] for record in records}
total = len(DATASETS) * 2 * len(CONFIGS)
print(
    {
        "n_jobs": N_JOBS,
        "surfaces": total,
        "resuming_complete": len(records),
        "datasets": {
            name: {
                "shape": list(data["X"].shape),
                "first": data["date"][0],
                "last": data["date"][-1],
                "hash": data_hash(data["X"], data["y"]),
            }
            for name, data in DATASETS.items()
        },
    },
    flush=True,
)

surface_number = 0
for dataset_name, data in DATASETS.items():
    for include_intercept in (False, True):
        X_base = data["X"]
        X = (
            np.column_stack((np.ones(X_base.shape[0]), X_base))
            if include_intercept
            else X_base
        )
        y = data["y"]
        dates = data["date"]
        unpenalized_indices = (0,) if include_intercept else ()
        for config in CONFIGS:
            surface_number += 1
            key = record_key(dataset_name, include_intercept, config)
            if key in completed:
                continue
            print(f"[{surface_number}/{total}] {key}", flush=True)
            started = time.perf_counter()
            if config["loss"] == "least-squares":
                base_loss = squared_loss
                loss_label = "least-squares"
            else:
                base_loss = make_huber(config["epsilon"])
                loss_label = f'Huber-eps{config["epsilon"]:g}'

            selected, cp, diagnostics = weighted_erm_joint_cv(
                X=X,
                y=y,
                M=base_loss,
                M_scorer=absolute_loss,
                Lmax=max(SEGMENT_COUNTS),
                delta_fraction=config["delta_fraction"],
                l1_constants=CONSTANTS,
                num_folds=config["num_folds"],
                n_jobs=N_JOBS,
                standardize_y=False,
                selection_strategy="exhaustive",
                return_diagnostics=True,
                penalty_norm=config["penalty_norm"],
                segment_counts=SEGMENT_COUNTS,
                unpenalized_feature_indices=unpenalized_indices,
            )
            candidates = [tuple(candidate) for candidate in diagnostics["candidates"]]
            scores = np.asarray(diagnostics["scores"], dtype=float)
            chosen_index = candidates.index(tuple(selected))
            selected_total = float(scores[chosen_index])
            baseline_total = min(
                float(score)
                for candidate, score in zip(candidates, scores)
                if candidate[0] == 1 and np.isfinite(score)
            )
            cp = np.asarray(cp, dtype=int).reshape(-1)
            record = {
                **config,
                "record_key": key,
                "dataset": dataset_name,
                "include_intercept": include_intercept,
                "intercept_penalized": False,
                "loss_label": loss_label,
                "n": int(len(y)),
                "p_without_intercept": int(X_base.shape[1]),
                "p_fit": int(X.shape[1]),
                "sample_start": str(dates[0]),
                "sample_end": str(dates[-1]),
                "data_hash": data_hash(X, y),
                "constants": list(CONSTANTS)
                if config["penalty_norm"] != "none"
                else [0.0],
                "segment_counts": list(SEGMENT_COUNTS),
                "score_loss": "absolute_loss",
                "refit_loss": loss_label,
                "selected_segments": int(selected[0]),
                "selected_change_points": int(selected[0] - 1),
                "selected_constant": float(selected[1]),
                "estimated_change_points": cp.tolist(),
                "estimated_dates": dates[cp].tolist(),
                "selected_cv_total": selected_total,
                "selected_cv_mean": selected_total / len(y),
                "best_zero_cp_cv_total": baseline_total,
                "best_zero_cp_cv_mean": baseline_total / len(y),
                "relative_improvement_vs_zero_pct": (
                    (baseline_total - selected_total) / baseline_total * 100
                ),
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
                "elapsed_seconds": time.perf_counter() - started,
            }
            records.append(record)
            completed.add(key)
            with CHECKPOINT.open("w") as handle:
                json.dump(records, handle, indent=2)
            print(
                {
                    "selected_cp": record["selected_change_points"],
                    "constant": record["selected_constant"],
                    "dates": record["estimated_dates"],
                    "cv_mean": record["selected_cv_mean"],
                    "improvement_pct": record[
                        "relative_improvement_vs_zero_pct"
                    ],
                    "seconds": record["elapsed_seconds"],
                },
                flush=True,
            )

selection_rows = [
    {
        "dataset": record["dataset"],
        "intercept": record["include_intercept"],
        "loss": record["loss_label"],
        "penalty": record["penalty_norm"],
        "folds": record["num_folds"],
        "delta_fraction": record["delta_fraction"],
        "selected_constant": record["selected_constant"],
        "selected_change_points": record["selected_change_points"],
        "estimated_dates": ";".join(record["estimated_dates"]),
        "cv_mae": record["selected_cv_mean"],
        "zero_cp_cv_mae": record["best_zero_cp_cv_mean"],
        "improvement_vs_zero_pct": record[
            "relative_improvement_vs_zero_pct"
        ],
    }
    for record in records
]
selection_table = pd.DataFrame(selection_rows).sort_values(
    ["dataset", "intercept", "penalty", "loss", "folds", "delta_fraction"]
)
selection_table.to_csv(RESULT_DIR / "all_selections.csv", index=False)
selection_table.loc[selection_table["selected_change_points"] > 0].to_csv(
    RESULT_DIR / "nonzero_selections.csv", index=False
)
selection_table.loc[
    selection_table["selected_change_points"].isin([2, 3])
].to_csv(RESULT_DIR / "two_or_three_change_points.csv", index=False)

joint_huber_rows = []
for (dataset_name, include_intercept, folds, delta), group in itertools.groupby(
    sorted(
        [
            record
            for record in records
            if record["penalty_norm"] == "none"
            and record["loss"] == "Huber"
        ],
        key=lambda record: (
            record["dataset"],
            record["include_intercept"],
            record["num_folds"],
            record["delta_fraction"],
        ),
    ),
    key=lambda record: (
        record["dataset"],
        record["include_intercept"],
        record["num_folds"],
        record["delta_fraction"],
    ),
):
    group = list(group)
    best = min(group, key=lambda record: record["selected_cv_total"])
    joint_huber_rows.append(
        {
            "dataset": dataset_name,
            "intercept": include_intercept,
            "folds": folds,
            "delta_fraction": delta,
            "selected_epsilon": best["epsilon"],
            "selected_change_points": best["selected_change_points"],
            "estimated_dates": ";".join(best["estimated_dates"]),
            "cv_mae": best["selected_cv_mean"],
            "improvement_vs_epsilon_specific_zero_pct": best[
                "relative_improvement_vs_zero_pct"
            ],
        }
    )
pd.DataFrame(joint_huber_rows).to_csv(
    RESULT_DIR / "joint_unpenalized_huber_epsilon_selections.csv", index=False
)

summary = {
    "notebook": "FRED-MD Period and Intercept CV.ipynb",
    "official_vintage": {
        "label": "2026-07",
        "source_url": "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/2026-07-md.csv",
        "raw_latest_row": current["raw_balanced_end"],
        "one_step_design_latest_date": str(current["date"][-1]),
    },
    "loss_contract": {
        "segmentation": "base loss",
        "hard_segment_refits": "same base loss",
        "heldout_scoring_only": "absolute_loss",
        "coefficient_penalty": "none, L1, or L2 as labeled",
        "intercept": "leading ones column when enabled; excluded from L1/L2 penalty",
    },
    "grid": {
        "constants": list(CONSTANTS),
        "segment_counts": list(SEGMENT_COUNTS),
        "sensitivities": list(SENSITIVITIES),
        "unpenalized_huber_epsilons": list(UNPENALIZED_HUBER_EPSILONS),
        "n_jobs": N_JOBS,
    },
    "datasets": {
        name: {
            "n": int(len(data["y"])),
            "p": int(data["X"].shape[1]),
            "start": str(data["date"][0]),
            "end": str(data["date"][-1]),
            "hash": data_hash(data["X"], data["y"]),
        }
        for name, data in DATASETS.items()
    },
    "records": records,
    "joint_unpenalized_huber_epsilon_selections": joint_huber_rows,
}
with SUMMARY_PATH.open("w") as handle:
    json.dump(summary, handle, indent=2)

print(
    {
        "complete": len(records),
        "nonzero": int((selection_table["selected_change_points"] > 0).sum()),
        "two_or_three": int(
            selection_table["selected_change_points"].isin([2, 3]).sum()
        ),
        "summary": str(SUMMARY_PATH),
    },
    flush=True,
)

