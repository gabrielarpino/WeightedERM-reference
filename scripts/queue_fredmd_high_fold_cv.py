"""Queue the focused FRED-MD high-fold unpenalized CV experiment."""

import os
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ACTIVE_PID = 462535
PREREQUISITE = Path(
    "results/FRED-MD_no-outlier-truncation-CV-comparison.pkl"
)
NOTEBOOK = Path("FRED-MD Real Data Experiment.ipynb")
MARKER = "# FRED-MD very-high-fold unpenalized CV experiment."


def process_exists(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


print(
    f"QUEUED high-fold CV experiment: waiting for PID {ACTIVE_PID}",
    flush=True,
)
while process_exists(ACTIVE_PID):
    time.sleep(60)

if not PREREQUISITE.exists():
    raise RuntimeError(
        f"Preceding experiment exited without final artifact: {PREREQUISITE}"
    )

nb = nbformat.read(NOTEBOOK, as_version=4)
if any(
    MARKER in cell.source for cell in nb.cells if cell.cell_type == "code"
):
    raise RuntimeError("The high-fold experiment cell already exists")

markdown = """# Very-high-fold unpenalized CV: 20, 50, and 100 folds

This focused experiment checks whether using very many interlaced folds changes
the number of change points selected by unpenalized CV. It uses the original
preprocessed FRED-MD design without an added bias column, endpoint fraction
`1/40`, candidate change-point counts 0--4, and absolute validation loss.
Least-squares and Huber (`epsilon=1.35`) fitting losses are compared. The run
uses 36 jobs, exhaustive CV, checkpoints after every configuration, unique
output names, and no execution timeout.
"""

code = r'''# FRED-MD very-high-fold unpenalized CV experiment.
import csv
import json
import pickle
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from WERM.experiment_utils import weighted_erm_joint_cv

_hf_folds = (20, 50, 100)
_hf_delta = 1 / 40
_hf_abs = lambda u, v: jnp.abs(u - v)

def _hf_ls(u, v):
    return jnp.square(u - v)

def _hf_huber(u, v, epsilon=1.35):
    residual = u - v
    absolute = jnp.abs(residual)
    return jnp.where(
        absolute <= epsilon,
        0.5 * jnp.square(residual),
        epsilon * (absolute - 0.5 * epsilon),
    )

_hf_losses = {
    "least-squares": _hf_ls,
    "Huber-eps1.35": _hf_huber,
}
_hf_out = Path("results/FRED-MD_unpenalized-high-fold-20-50-100-CV.pkl")
_hf_checkpoint = Path(
    "results/FRED-MD_unpenalized-high-fold-20-50-100-CV_checkpoint.pkl"
)
_hf_json = Path(
    "results/FRED-MD_unpenalized-high-fold-20-50-100-CV_summary.json"
)
_hf_csv = Path(
    "results/FRED-MD_unpenalized-high-fold-20-50-100-CV.csv"
)
_hf_figure = Path(
    "figures/FRED-MD_unpenalized-high-fold-20-50-100-CV.png"
)
_hf_out.parent.mkdir(parents=True, exist_ok=True)
_hf_figure.parent.mkdir(parents=True, exist_ok=True)
if _hf_out.exists():
    raise FileExistsError(f"Refusing to overwrite {_hf_out}")

_hf_records = []
if _hf_checkpoint.exists():
    with open(_hf_checkpoint, "rb") as handle:
        _hf_records = list(pickle.load(handle).get("records", []))
_hf_done = {(r["loss"], r["num_folds"]) for r in _hf_records}

for _hf_loss_name, _hf_M in _hf_losses.items():
    for _hf_k in _hf_folds:
        _hf_key = (_hf_loss_name, _hf_k)
        if _hf_key in _hf_done:
            continue
        _hf_selected, _hf_cp, _hf_diag = weighted_erm_joint_cv(
            X=X,
            y=y,
            M=_hf_M,
            M_scorer=_hf_abs,
            Lmax=5,
            delta_fraction=_hf_delta,
            l1_constants=(0.0,),
            num_folds=_hf_k,
            n_jobs=36,
            selection_strategy="exhaustive",
            return_diagnostics=True,
            penalty_norm="none",
        )
        _hf_curve = [
            {
                "segments": int(candidate[0]),
                "change_points": int(candidate[0]) - 1,
                "cv_total_absolute_error": float(score),
                "cv_mean_absolute_error": float(score) / len(y),
            }
            for candidate, score in zip(
                _hf_diag["candidates"], _hf_diag["scores"]
            )
        ]
        _hf_cp_list = np.asarray(_hf_cp, dtype=int).reshape(-1).tolist()
        _hf_dates = [
            str(np.asarray(date).reshape(-1)[idx])
            for idx in _hf_cp_list
            if 0 <= idx < len(np.asarray(date).reshape(-1))
        ]
        _hf_record = {
            "loss": _hf_loss_name,
            "num_folds": int(_hf_k),
            "bias": "no-bias",
            "delta_fraction": float(_hf_delta),
            "selected_segments": int(_hf_selected[0]),
            "selected_change_points": int(_hf_selected[0]) - 1,
            "estimated_change_points": _hf_cp_list,
            "estimated_dates": _hf_dates,
            "score_curve": _hf_curve,
        }
        _hf_records.append(_hf_record)
        _hf_done.add(_hf_key)
        with open(_hf_checkpoint, "wb") as handle:
            pickle.dump({"records": _hf_records}, handle)
        print(
            f"completed {_hf_loss_name}, folds={_hf_k}: "
            f"selected {_hf_record['selected_change_points']} CP "
            f"{_hf_cp_list} {_hf_dates}",
            flush=True,
        )

with open(_hf_out, "wb") as handle:
    pickle.dump(
        {
            "records": _hf_records,
            "configuration": {
                "folds": _hf_folds,
                "delta_fraction": _hf_delta,
                "bias": "no-bias",
                "penalty": "none",
                "M_scorer": "absolute",
                "n_jobs": 36,
            },
        },
        handle,
    )
with open(_hf_json, "w") as handle:
    json.dump({"records": _hf_records}, handle, indent=2)
with open(_hf_csv, "w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(
        [
            "loss",
            "num_folds",
            "candidate_change_points",
            "cv_mean_absolute_error",
            "selected_change_points",
            "estimated_change_points",
            "estimated_dates",
        ]
    )
    for record in _hf_records:
        for point in record["score_curve"]:
            writer.writerow(
                [
                    record["loss"],
                    record["num_folds"],
                    point["change_points"],
                    point["cv_mean_absolute_error"],
                    record["selected_change_points"],
                    record["estimated_change_points"],
                    record["estimated_dates"],
                ]
            )

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
for axis, loss_name in zip(axes, _hf_losses):
    for fold_count in _hf_folds:
        record = next(
            r
            for r in _hf_records
            if r["loss"] == loss_name and r["num_folds"] == fold_count
        )
        axis.plot(
            [p["change_points"] for p in record["score_curve"]],
            [p["cv_mean_absolute_error"] for p in record["score_curve"]],
            marker="o",
            linewidth=2,
            label=f"{fold_count} folds",
        )
        selected = record["selected_change_points"]
        selected_score = record["score_curve"][selected]["cv_mean_absolute_error"]
        axis.scatter([selected], [selected_score], s=90, marker="*", zorder=5)
    axis.set_title(loss_name)
    axis.set_xlabel("Candidate number of change points")
    axis.set_xticks(range(5))
    axis.grid(alpha=0.25)
    axis.legend()
axes[0].set_ylabel("Interlaced CV mean absolute error")
fig.suptitle(
    "FRED-MD unpenalized CV with 20, 50, and 100 folds "
    "(no bias, delta fraction 1/40)"
)
fig.tight_layout()
fig.savefig(_hf_figure, dpi=200, bbox_inches="tight")
display(fig)

print("\nFinal high-fold CV selections:")
for record in _hf_records:
    print(
        f"{record['loss']}, {record['num_folds']} folds: "
        f"{record['selected_change_points']} CP; "
        f"indices={record['estimated_change_points']}; "
        f"dates={record['estimated_dates']}"
    )
print(f"COMPLETE: {len(_hf_records)}/6 configurations saved to {_hf_out}")
'''

compile(code, "fredmd-high-fold-cell", "exec")
nb.cells.append(nbformat.v4.new_markdown_cell(markdown))
nb.cells.append(nbformat.v4.new_code_cell(code))
nbformat.write(nb, NOTEBOOK)
print("HIGH-FOLD CELL APPENDED: launching with timeout=None", flush=True)

setup = '''import pickle, numpy as np, jax.numpy as jnp
with open("FRED-MD data/fredmd_data.pkl", "rb") as handle:
    loaded_data = pickle.load(handle)
X = loaded_data["X"]
y = loaded_data["y"]
date = loaded_data["date"]
'''
run_nb = nbformat.v4.new_notebook(metadata=nb.metadata)
run_nb.cells = [
    nbformat.v4.new_code_cell(setup),
    nb.cells[-1],
]
client = NotebookClient(
    run_nb,
    timeout=None,
    kernel_name="weightederm",
    resources={"metadata": {"path": str(NOTEBOOK.parent)}},
)
try:
    client.execute()
finally:
    latest = nbformat.read(NOTEBOOK, as_version=4)
    latest.cells[-1].outputs = run_nb.cells[-1].get("outputs", [])
    latest.cells[-1].execution_count = run_nb.cells[-1].get("execution_count")
    nbformat.write(latest, NOTEBOOK)
    print("HIGH-FOLD NOTEBOOK OUTPUTS SAVED", flush=True)
print("HIGH-FOLD EXPERIMENT FINISHED", flush=True)
