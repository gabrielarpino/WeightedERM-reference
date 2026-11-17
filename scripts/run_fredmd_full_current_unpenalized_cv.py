"""Run the missing full-sample latest-vintage unpenalized WERM-CV grid."""

from pathlib import Path
import hashlib
import json
import os
import pickle
import time

import jax.numpy as jnp
import numpy as np
import pandas as pd

from WERM.experiment_utils import weighted_erm_joint_cv


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "FRED-MD_full-current-unpenalized-CV"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT = RESULT_DIR / "checkpoint.json"
N_JOBS = min(36, os.cpu_count() or 1)
SEGMENT_COUNTS = (1, 2, 3, 4, 5)
EPSILONS = (0.5, 0.75, 1.0, 1.35, 2.0, 3.0)
SENSITIVITIES = (
    (2, 1 / 50),
    (2, 1 / 30),
    (5, 1 / 50),
    (5, 1 / 30),
)


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


def hash_data(X, y):
    return hashlib.blake2b(
        np.ascontiguousarray(X).tobytes()
        + np.ascontiguousarray(y).tobytes(),
        digest_size=16,
    ).hexdigest()


with (ROOT / "FRED-MD data" / "fredmd_2026-07_mcscan.pkl").open("rb") as handle:
    data = pickle.load(handle)

X_base = np.asarray(data["X"], dtype=float)
y = np.asarray(data["y"], dtype=float).reshape(-1)
dates = np.asarray(data["date"]).astype(str)
assert dates[0] == "1960-01" and dates[-1] == "2026-05"
assert X_base.shape == (797, 92)

if CHECKPOINT.exists():
    with CHECKPOINT.open() as handle:
        records = json.load(handle)
else:
    records = []
completed = {record["key"] for record in records}

configs = [("least-squares", None)] + [("Huber", value) for value in EPSILONS]
total = 2 * len(SENSITIVITIES) * len(configs)
surface = 0

for include_intercept in (False, True):
    X = (
        np.column_stack((np.ones(len(y)), X_base))
        if include_intercept
        else X_base
    )
    unpenalized_indices = (0,) if include_intercept else ()
    for num_folds, delta_fraction in SENSITIVITIES:
        for loss_name, epsilon in configs:
            surface += 1
            label = (
                "least-squares"
                if epsilon is None
                else f"Huber-eps{epsilon:g}"
            )
            key = "|".join(
                (
                    f"intercept={int(include_intercept)}",
                    label,
                    f"folds={num_folds}",
                    f"delta={delta_fraction:.12g}",
                )
            )
            if key in completed:
                continue
            print(f"[{surface}/{total}] {key}", flush=True)
            started = time.perf_counter()
            base_loss = squared_loss if epsilon is None else make_huber(epsilon)
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
            candidates = [tuple(value) for value in diagnostics["candidates"]]
            scores = np.asarray(diagnostics["scores"], dtype=float)
            chosen = candidates.index(tuple(selected))
            selected_total = float(scores[chosen])
            zero_total = float(scores[candidates.index((1, 0.0))])
            cp = np.asarray(cp, dtype=int).reshape(-1)
            record = {
                "key": key,
                "intercept": include_intercept,
                "loss": label,
                "epsilon": epsilon,
                "folds": num_folds,
                "delta_fraction": delta_fraction,
                "selected_change_points": int(selected[0] - 1),
                "estimated_indices": cp.tolist(),
                "estimated_dates": dates[cp].tolist(),
                "cv_mae": selected_total / len(y),
                "zero_cp_cv_mae": zero_total / len(y),
                "improvement_vs_zero_pct": (
                    (zero_total - selected_total) / zero_total * 100
                ),
                "elapsed_seconds": time.perf_counter() - started,
                "candidate_scores": [
                    {
                        "segments": int(candidate[0]),
                        "change_points": int(candidate[0] - 1),
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
                    "dates": record["estimated_dates"],
                    "cv_mae": record["cv_mae"],
                    "gain_pct": record["improvement_vs_zero_pct"],
                    "seconds": record["elapsed_seconds"],
                },
                flush=True,
            )

table = pd.DataFrame(
    [
        {key: value for key, value in record.items() if key != "candidate_scores"}
        for record in records
    ]
).sort_values(["intercept", "folds", "delta_fraction", "loss"])
table["estimated_dates"] = table["estimated_dates"].map(";".join)
table.to_csv(RESULT_DIR / "all_selections.csv", index=False)

joint_rows = []
for (intercept, folds, delta), group in table.loc[
    table["epsilon"].notna()
].groupby(["intercept", "folds", "delta_fraction"]):
    best = group.loc[group["cv_mae"].idxmin()]
    joint_rows.append(
        {
            "intercept": bool(intercept),
            "folds": int(folds),
            "delta_fraction": float(delta),
            "selected_epsilon": float(best["epsilon"]),
            "selected_change_points": int(best["selected_change_points"]),
            "estimated_dates": best["estimated_dates"],
            "cv_mae": float(best["cv_mae"]),
            "improvement_vs_epsilon_specific_zero_pct": float(
                best["improvement_vs_zero_pct"]
            ),
        }
    )
joint = pd.DataFrame(joint_rows)
joint.to_csv(RESULT_DIR / "joint_huber_epsilon_selections.csv", index=False)

summary = {
    "dataset": {
        "vintage": "2026-07",
        "n": len(y),
        "p": X_base.shape[1],
        "start": dates[0],
        "end": dates[-1],
        "hash": hash_data(X_base, y),
    },
    "loss_contract": {
        "segmentation": "base loss",
        "hard_refits": "base loss",
        "heldout_scoring_only": "absolute_loss",
    },
    "n_jobs": N_JOBS,
    "records": records,
    "joint_huber_epsilon_selections": joint_rows,
}
with (RESULT_DIR / "summary.json").open("w") as handle:
    json.dump(summary, handle, indent=2)

print(
    {
        "complete": len(records),
        "nonzero": int((table["selected_change_points"] > 0).sum()),
        "summary": str(RESULT_DIR / "summary.json"),
    },
    flush=True,
)
