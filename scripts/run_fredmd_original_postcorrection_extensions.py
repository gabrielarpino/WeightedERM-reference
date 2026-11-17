"""Queue the missing corrected-CV extensions on the original FRED-MD sample.

Priority order:
1. Unpenalized higher-Delta CV.
2. Unpenalized higher-fold CV.
3. Penalized higher-Delta CV.
4. Penalized higher-fold CV.

Every surface uses the corrected loss contract: WERM segmentation and hard
segment fits use the base loss; absolute loss is held-out scoring only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

# Keep each worker single-threaded so at most N_JOBS CPU threads are requested.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import jax.numpy as jnp
import numpy as np

from WERM.experiment_utils import weighted_erm_joint_cv


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "FRED-MD_original-postcorrection-extensions"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT = RESULT_DIR / "checkpoint.json"
SUMMARY = RESULT_DIR / "summary.json"
ALL_CSV = RESULT_DIR / "all_selections.csv"
TARGET_CSV = RESULT_DIR / "target_2_or_3_change_points.csv"
STATUS = RESULT_DIR / "status.json"

N_JOBS = min(36, os.cpu_count() or 1)
SEGMENT_COUNTS = (1, 2, 3, 4, 5)  # zero through four change points
CONSTANTS = (0.15, 0.3, 0.5, 0.75, 1.0, 2.0)
LOSSES = ("least-squares", "Huber-eps1.35")
INTERCEPTS = (False, True)


def squared_loss(prediction, truth):
    return jnp.square(prediction - truth)


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


def data_hash(X, y):
    return hashlib.blake2b(
        np.ascontiguousarray(X).tobytes() + np.ascontiguousarray(y).tobytes(),
        digest_size=16,
    ).hexdigest()


def make_configurations():
    configs = []

    # Most important: unpenalized higher Delta, including moderate higher folds.
    for folds in (2, 5, 10, 20):
        for delta in (1 / 20, 1 / 10, 1 / 5):
            for loss in LOSSES:
                for intercept in INTERCEPTS:
                    configs.append(("unpenalized_higher_delta", loss, intercept,
                                    "none", folds, delta))

    # Unpenalized high-fold coverage at the two established Delta values.
    for folds in (10, 20, 50, 100):
        for delta in (1 / 50, 1 / 30):
            for loss in LOSSES:
                for intercept in INTERCEPTS:
                    configs.append(("unpenalized_higher_folds", loss, intercept,
                                    "none", folds, delta))

    # Higher Delta with L1 and L2; include 10 folds as a useful extension.
    for folds in (2, 5, 10):
        for delta in (1 / 20, 1 / 10, 1 / 5):
            for loss in LOSSES:
                for intercept in INTERCEPTS:
                    for norm in ("l1", "l2"):
                        configs.append(("penalized_higher_delta", loss, intercept,
                                        norm, folds, delta))

    # Higher-fold penalized coverage; 50/100-fold penalized surfaces are omitted
    # because 10/20 folds already diagnose the fold-direction effect at far lower
    # computational cost. Unpenalized fits retain the requested 50/100 coverage.
    for folds in (10, 20):
        for delta in (1 / 50, 1 / 30):
            for loss in LOSSES:
                for intercept in INTERCEPTS:
                    for norm in ("l1", "l2"):
                        configs.append(("penalized_higher_folds", loss, intercept,
                                        norm, folds, delta))
    return configs


def configuration_key(config):
    stage, loss, intercept, norm, folds, delta = config
    return "|".join(
        (stage, loss, f"intercept={int(intercept)}", norm, f"folds={folds}",
         f"delta={delta:.12g}")
    )


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2)
    temporary.replace(path)


def csv_value(value):
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"))
    return value


def write_csv(path, rows):
    if not rows:
        return
    columns = [
        "stage", "loss", "intercept", "penalty_norm", "folds",
        "delta_fraction", "delta_label", "minimum_segment_length",
        "selected_constant", "selected_change_points", "change_point_indices",
        "change_point_dates", "cv_mae", "zero_cp_cv_mae",
        "improvement_vs_zero_pct", "mean_distance_to_1982_2008_months",
        "nearest_distance_to_2020_months", "elapsed_seconds", "key",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: csv_value(row.get(column)) for column in columns})
    temporary.replace(path)


def month_number(date):
    year, month = map(int, str(date).split("-")[:2])
    return 12 * year + month - 1


def target_distances(change_dates):
    if not change_dates:
        return float("inf"), float("inf")
    estimates = np.asarray([month_number(value) for value in change_dates])
    reference = np.asarray([month_number("1982-01"), month_number("2008-09")])
    primary = float(np.mean([np.min(np.abs(estimates - value)) for value in reference]))
    bonus = float(np.min(np.abs(estimates - month_number("2020-01"))))
    return primary, bonus


with (ROOT / "FRED-MD data" / "fredmd_data.pkl").open("rb") as handle:
    data = pickle.load(handle)
X_base = np.asarray(data["X"], dtype=float)
y = np.asarray(data["y"], dtype=float).reshape(-1)
dates = np.asarray(data["date"]).astype(str)
assert X_base.shape == (773, 114)
assert dates[[0, -1]].tolist() == ["1960-01", "2024-05"]

if CHECKPOINT.exists():
    with CHECKPOINT.open() as handle:
        records = json.load(handle)
else:
    records = []
completed = {record["key"] for record in records}
configurations = make_configurations()
started_run = time.time()

print(json.dumps({
    "event": "start",
    "n_jobs": N_JOBS,
    "surfaces": len(configurations),
    "already_complete": len(completed),
    "data_shape": list(X_base.shape),
    "date_range": [dates[0], dates[-1]],
    "data_hash": data_hash(X_base, y),
}), flush=True)

for position, config in enumerate(configurations, start=1):
    key = configuration_key(config)
    if key in completed:
        continue
    stage, loss_label, intercept, norm, folds, delta = config
    X = np.column_stack((np.ones(len(y)), X_base)) if intercept else X_base
    unpenalized_indices = (0,) if intercept else ()
    base_loss = squared_loss if loss_label == "least-squares" else huber_loss
    started_surface = time.perf_counter()
    print(f"[{position}/{len(configurations)}] {key}", flush=True)
    try:
        selected, change_points, diagnostics = weighted_erm_joint_cv(
            X=X,
            y=y,
            M=base_loss,
            M_scorer=absolute_loss,
            Lmax=5,
            delta_fraction=delta,
            l1_constants=CONSTANTS,
            num_folds=folds,
            n_jobs=N_JOBS,
            standardize_y=False,
            selection_strategy="exhaustive",
            return_diagnostics=True,
            penalty_norm=norm,
            segment_counts=SEGMENT_COUNTS,
            unpenalized_feature_indices=unpenalized_indices,
        )
        candidates = [tuple(candidate) for candidate in diagnostics["candidates"]]
        scores = np.asarray(diagnostics["scores"], dtype=float)
        selected_index = candidates.index(tuple(selected))
        selected_total = float(scores[selected_index])
        zero_total = min(
            float(score)
            for candidate, score in zip(candidates, scores)
            if candidate[0] == 1 and np.isfinite(score)
        )
        change_points = np.asarray(change_points, dtype=int).reshape(-1)
        change_dates = dates[change_points].tolist()
        primary_distance, bonus_distance = target_distances(change_dates)
        record = {
            "key": key,
            "stage": stage,
            "loss": loss_label,
            "intercept": intercept,
            "penalty_norm": norm,
            "folds": folds,
            "delta_fraction": delta,
            "delta_label": f"n/{round(1 / delta)}",
            "minimum_segment_length": max(
                int(np.ceil(len(y) / 50)), int(delta * len(y)), 1
            ),
            "selected_constant": float(selected[1]),
            "selected_change_points": int(selected[0] - 1),
            "change_point_indices": change_points.tolist(),
            "change_point_dates": change_dates,
            "cv_mae": selected_total / len(y),
            "zero_cp_cv_mae": zero_total / len(y),
            "improvement_vs_zero_pct": (zero_total - selected_total) / zero_total * 100,
            "mean_distance_to_1982_2008_months": primary_distance,
            "nearest_distance_to_2020_months": bonus_distance,
            "elapsed_seconds": time.perf_counter() - started_surface,
            "candidate_scores": [
                {
                    "segments": int(candidate[0]),
                    "change_points": int(candidate[0] - 1),
                    "constant": float(candidate[1]),
                    "cv_total": float(score),
                    "cv_mae": float(score / len(y)),
                }
                for candidate, score in zip(candidates, scores)
            ],
        }
    except Exception as error:  # checkpoint the failure and continue the queue
        record = {
            "key": key,
            "stage": stage,
            "loss": loss_label,
            "intercept": intercept,
            "penalty_norm": norm,
            "folds": folds,
            "delta_fraction": delta,
            "delta_label": f"n/{round(1 / delta)}",
            "error": f"{type(error).__name__}: {error}",
            "elapsed_seconds": time.perf_counter() - started_surface,
        }
        print(json.dumps({"event": "surface_error", "key": key,
                          "error": record["error"]}), flush=True)
    records.append(record)
    completed.add(key)
    write_json(CHECKPOINT, records)
    successful = [record for record in records if "error" not in record]
    targets = sorted(
        [record for record in successful if record["selected_change_points"] in (2, 3)],
        key=lambda record: (
            record["penalty_norm"] != "none",
            record["mean_distance_to_1982_2008_months"],
            record["nearest_distance_to_2020_months"],
            -record["improvement_vs_zero_pct"],
        ),
    )
    write_csv(ALL_CSV, successful)
    write_csv(TARGET_CSV, targets)
    write_json(STATUS, {
        "state": "running",
        "completed_surfaces": len(records),
        "total_surfaces": len(configurations),
        "successful_surfaces": len(successful),
        "failed_surfaces": len(records) - len(successful),
        "target_2_or_3_cp_surfaces": len(targets),
        "last_key": key,
        "updated_unix_time": time.time(),
    })
    print(json.dumps({
        "event": "surface_complete",
        "completed": len(records),
        "selected_change_points": record.get("selected_change_points"),
        "dates": record.get("change_point_dates"),
        "targets_so_far": len(targets),
    }), flush=True)

successful = [record for record in records if "error" not in record]
targets = sorted(
    [record for record in successful if record["selected_change_points"] in (2, 3)],
    key=lambda record: (
        record["penalty_norm"] != "none",
        record["mean_distance_to_1982_2008_months"],
        record["nearest_distance_to_2020_months"],
        -record["improvement_vs_zero_pct"],
    ),
)
summary = {
    "state": "complete",
    "loss_contract": "base M for segmentation and hard fits; absolute heldout score only",
    "data": {"shape": list(X_base.shape), "start": dates[0], "end": dates[-1]},
    "grid": {
        "Huber_epsilon": 1.35,
        "intercepts": list(INTERCEPTS),
        "segment_counts": list(SEGMENT_COUNTS),
        "constants": list(CONSTANTS),
        "n_jobs": N_JOBS,
        "surfaces": len(configurations),
    },
    "records": records,
    "target_2_or_3_change_point_records": targets,
    "elapsed_seconds_this_invocation": time.time() - started_run,
}
write_json(SUMMARY, summary)
write_json(STATUS, {
    "state": "complete",
    "completed_surfaces": len(records),
    "total_surfaces": len(configurations),
    "successful_surfaces": len(successful),
    "failed_surfaces": len(records) - len(successful),
    "target_2_or_3_cp_surfaces": len(targets),
    "updated_unix_time": time.time(),
})
print(json.dumps({"event": "complete", "surfaces": len(records),
                  "targets": len(targets)}), flush=True)
