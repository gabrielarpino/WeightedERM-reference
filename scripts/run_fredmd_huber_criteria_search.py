"""Search original-sample Huber settings satisfying the requested criteria."""

from __future__ import annotations

import json
import os
import pickle
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import jax.numpy as jnp
import numpy as np

import WERM.estimators
from WERM.experiment_utils import weighted_erm_joint_cv


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "FRED-MD_original-Huber-criteria-search"
OUT.mkdir(parents=True, exist_ok=True)
CHECKPOINT = OUT / "checkpoint.json"
STATUS = OUT / "status.json"
SUMMARY = OUT / "summary.json"
N_JOBS = min(36, os.cpu_count() or 1)
CONSTANTS = (0.15, 0.3, 0.5, 0.75, 1.0, 2.0)
SEGMENTS = (1, 2, 3, 4, 5)
DENOMINATORS = (30, 20, 10, 5, 40)
FOLD_STAGES = (5, 2, 10)
PRIMARY_DATES = ("1982-01", "2008-09")
LATE_DATE = "2015-01"
CLOSE_MONTHS = 36.0


def huber(prediction, truth, epsilon=1.35):
    residual = prediction - truth
    absolute = jnp.abs(residual)
    return jnp.where(
        absolute <= epsilon,
        0.5 * jnp.square(residual),
        epsilon * (absolute - 0.5 * epsilon),
    )


def absolute(prediction, truth):
    return jnp.abs(prediction - truth)


def month_number(value):
    year, month = map(int, str(value).split("-")[:2])
    return year * 12 + month - 1


def primary_distance(date_values):
    if not date_values:
        return float("inf")
    estimates = np.asarray([month_number(value) for value in date_values])
    references = np.asarray([month_number(value) for value in PRIMARY_DATES])
    return float(np.mean([np.min(np.abs(estimates - value)) for value in references]))


def original_slope_elbow(gofs):
    values = np.asarray(gofs, dtype=float)
    average_slopes = [
        ((values[i] - values[i - 1]) + (values[i + 1] - values[i])) / 2
        for i in range(1, len(values) - 1)
    ]
    for j in range(1, len(average_slopes)):
        if abs(average_slopes[j]) < abs(average_slopes[j - 1]):
            return j + 2, average_slopes
    return len(values), average_slopes


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2)
    temporary.replace(path)


def full_unpenalized_curve(X, y, dates, denominator):
    delta = 1 / denominator
    minimum_length = max(int(np.ceil(len(y) / 50)), int(delta * len(y)), 1)
    curve = []
    for segments in SEGMENTS:
        try:
            change_points, gof = WERM.estimators.weighted_erm(
                X=X,
                y=y,
                M=huber,
                L=segments,
                Delta_n=minimum_length,
                gamma=0.0,
                brute_force=False,
                return_gof=True,
                P_func=lambda beta: jnp.array(0.0),
            )
        except AssertionError:
            curve.append({
                "segments": segments,
                "change_points": segments - 1,
                "feasible": False,
                "indices": [],
                "dates": [],
                "gof": None,
            })
            continue
        change_points = np.asarray(change_points, dtype=int).reshape(-1)
        curve.append({
            "segments": segments,
            "change_points": segments - 1,
            "feasible": True,
            "indices": change_points.tolist(),
            "dates": dates[change_points].tolist(),
            "gof": float(gof),
        })
    feasible_curve = [row for row in curve if row["feasible"]]
    elbow_position, slopes = original_slope_elbow([row["gof"] for row in feasible_curve])
    elbow = feasible_curve[elbow_position - 1]
    elbow_segments = elbow["segments"]
    return {
        "minimum_length": minimum_length,
        "curve": curve,
        "elbow_segments": elbow_segments,
        "elbow_change_points": elbow_segments - 1,
        "elbow_dates": elbow["dates"],
        "elbow_primary_distance_months": primary_distance(elbow["dates"]),
        "average_slopes": [float(value) for value in slopes],
        "fixed_2_change_points": curve[2],
        "fixed_3_change_points": curve[3],
    }


def run_cv(X, y, dates, folds, denominator, penalty_norm):
    selected, change_points, diagnostics = weighted_erm_joint_cv(
        X=X,
        y=y,
        M=huber,
        M_scorer=absolute,
        Lmax=5,
        delta_fraction=1 / denominator,
        l1_constants=CONSTANTS,
        num_folds=folds,
        n_jobs=N_JOBS,
        standardize_y=False,
        selection_strategy="exhaustive",
        return_diagnostics=True,
        penalty_norm=penalty_norm,
        segment_counts=SEGMENTS,
        unpenalized_feature_indices=(0,) if X.shape[1] == 115 else (),
    )
    candidates = [tuple(candidate) for candidate in diagnostics["candidates"]]
    scores = np.asarray(diagnostics["scores"], dtype=float)
    selected_index = candidates.index(tuple(selected))
    change_points = np.asarray(change_points, dtype=int).reshape(-1)
    return {
        "segments": int(selected[0]),
        "change_points": int(selected[0] - 1),
        "constant": float(selected[1]),
        "indices": change_points.tolist(),
        "dates": dates[change_points].tolist(),
        "cv_mae": float(scores[selected_index] / len(y)),
        "scores": [
            {
                "segments": int(candidate[0]),
                "change_points": int(candidate[0] - 1),
                "constant": float(candidate[1]),
                "cv_mae": float(score / len(y)),
            }
            for candidate, score in zip(candidates, scores)
        ],
    }


with (ROOT / "FRED-MD data" / "fredmd_data.pkl").open("rb") as handle:
    data = pickle.load(handle)
X_base = np.asarray(data["X"], dtype=float)
y = np.asarray(data["y"], dtype=float).reshape(-1)
dates = np.asarray(data["date"]).astype(str)
assert X_base.shape == (773, 114)
assert dates[[0, -1]].tolist() == ["1960-01", "2024-05"]

if CHECKPOINT.exists():
    with CHECKPOINT.open() as handle:
        state = json.load(handle)
else:
    state = {"curves": {}, "records": []}
completed = {record["key"] for record in state["records"]}

stage_found_stable_candidate = False
for folds in FOLD_STAGES:
    stage_found_candidate = False
    for denominator in DENOMINATORS:
        for intercept in (False, True):
            key = f"folds={folds}|denominator={denominator}|intercept={int(intercept)}"
            if key in completed:
                previous = next(record for record in state["records"] if record["key"] == key)
                stage_found_candidate |= previous.get("all_requested_criteria", False)
                stage_found_stable_candidate |= previous.get("stable_l1_ge_2", False)
                continue
            X = np.column_stack((np.ones(len(y)), X_base)) if intercept else X_base
            curve_key = f"denominator={denominator}|intercept={int(intercept)}"
            print(f"running {key}", flush=True)
            started = time.perf_counter()
            if curve_key not in state["curves"]:
                state["curves"][curve_key] = full_unpenalized_curve(
                    X, y, dates, denominator
                )
                write_json(CHECKPOINT, state)
            curve = state["curves"][curve_key]
            l1 = run_cv(X, y, dates, folds, denominator, "l1")
            unpenalized_cv = run_cv(X, y, dates, folds, denominator, "none")
            fixed_2 = curve["fixed_2_change_points"]
            fixed_3 = curve["fixed_3_change_points"]
            fixed_2_distance = primary_distance(fixed_2["dates"])
            fixed_3_distance = primary_distance(fixed_3["dates"])
            fixed_3_has_late = any(str(value) >= LATE_DATE for value in fixed_3["dates"])
            base_criteria = (
                l1["change_points"] >= 2
                and curve["elbow_primary_distance_months"] <= CLOSE_MONTHS
                and fixed_2_distance <= CLOSE_MONTHS
                and fixed_3_distance <= CLOSE_MONTHS
                and fixed_3_has_late
            )
            repeats = []
            if base_criteria:
                for _ in range(2):
                    repeats.append(run_cv(X, y, dates, folds, denominator, "l1"))
            stable_l1_ge_2 = bool(base_criteria and all(
                repeat["change_points"] >= 2 for repeat in repeats
            ))
            record = {
                "key": key,
                "folds": folds,
                "denominator": denominator,
                "delta_fraction": 1 / denominator,
                "intercept": intercept,
                "l1_full_cv": l1,
                "l1_repeat_results": repeats,
                "unpenalized_full_cv": unpenalized_cv,
                "unpenalized_elbow_change_points": curve["elbow_change_points"],
                "unpenalized_elbow_dates": curve["elbow_dates"],
                "unpenalized_elbow_primary_distance_months": curve[
                    "elbow_primary_distance_months"
                ],
                "unpenalized_fixed_2": fixed_2,
                "unpenalized_fixed_2_primary_distance_months": fixed_2_distance,
                "unpenalized_fixed_3": fixed_3,
                "unpenalized_fixed_3_primary_distance_months": fixed_3_distance,
                "unpenalized_fixed_3_has_post_2015": fixed_3_has_late,
                "all_requested_criteria": base_criteria,
                "stable_l1_ge_2": stable_l1_ge_2,
                "elapsed_seconds": time.perf_counter() - started,
            }
            state["records"].append(record)
            completed.add(key)
            stage_found_candidate |= base_criteria
            stage_found_stable_candidate |= stable_l1_ge_2
            write_json(CHECKPOINT, state)
            write_json(STATUS, {
                "state": "running",
                "fold_stage": folds,
                "completed_records": len(state["records"]),
                "last_key": key,
                "candidates": sum(
                    record.get("all_requested_criteria", False)
                    for record in state["records"]
                ),
                "stable_candidates": sum(
                    record.get("stable_l1_ge_2", False)
                    for record in state["records"]
                ),
                "updated_unix_time": time.time(),
            })
            print(json.dumps({
                "key": key,
                "l1_change_points": l1["change_points"],
                "l1_dates": l1["dates"],
                "unpenalized_cv_change_points": unpenalized_cv["change_points"],
                "elbow_dates": curve["elbow_dates"],
                "fixed_2_dates": fixed_2["dates"],
                "fixed_3_dates": fixed_3["dates"],
                "all_criteria": base_criteria,
                "stable": stable_l1_ge_2,
            }), flush=True)
    # The user asked to try 5 folds first, using other fold counts only if it fails.
    if stage_found_stable_candidate:
        break

qualifying = [
    record for record in state["records"] if record["all_requested_criteria"]
]
stable = [record for record in state["records"] if record["stable_l1_ge_2"]]
summary = {
    "state": "complete",
    "data": {"shape": list(X_base.shape), "start": dates[0], "end": dates[-1]},
    "settings": {
        "fit_loss": "Huber epsilon=1.35",
        "score_loss": "absolute heldout only",
        "denominators": list(DENOMINATORS),
        "fold_stages": list(FOLD_STAGES),
        "close_threshold_months": CLOSE_MONTHS,
        "late_threshold": LATE_DATE,
        "constants": list(CONSTANTS),
        "n_jobs": N_JOBS,
    },
    "qualifying_records": qualifying,
    "stable_qualifying_records": stable,
    "records": state["records"],
    "curves": state["curves"],
}
write_json(SUMMARY, summary)
write_json(STATUS, {
    "state": "complete",
    "completed_records": len(state["records"]),
    "candidates": len(qualifying),
    "stable_candidates": len(stable),
    "updated_unix_time": time.time(),
})
print(json.dumps({"event": "complete", "records": len(state["records"]),
                  "candidates": len(qualifying), "stable": len(stable)}), flush=True)
