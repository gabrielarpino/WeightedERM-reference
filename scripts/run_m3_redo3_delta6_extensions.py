"""Execute only the appended redo3 delta=6 alpha extensions and embed outputs."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "M3_upsample_redo3.ipynb"
SCENARIOS = ((0.1, "01"), (0.3, "03"))
KINDS = ("experiment", "full_plot", "two_method_plot")


def _extension_cells(notebook, alpha):
    by_kind = {}
    for cell in notebook.cells:
        metadata = cell.metadata
        if metadata.get("m3_delta6_extension") and metadata.get("alpha") == alpha:
            kind = metadata.get("kind")
            if kind in KINDS:
                by_kind[kind] = cell
    if tuple(by_kind) != KINDS:
        raise RuntimeError(f"delta=6 alpha={alpha:g}: expected cells {KINDS}, found {tuple(by_kind)}")
    return [by_kind[kind] for kind in KINDS]


def _execute(cells, notebook_metadata):
    run_notebook = nbformat.v4.new_notebook(
        cells=[copy.deepcopy(cell) for cell in cells],
        metadata=copy.deepcopy(notebook_metadata),
    )
    return NotebookClient(
        run_notebook,
        timeout=None,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    ).execute()


def _merge_outputs(alpha, executed):
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    targets = {cell.id: cell for cell in _extension_cells(notebook, alpha)}
    for completed in executed.cells:
        target = targets[completed.id]
        target.outputs = completed.outputs
        target.execution_count = completed.execution_count
    temporary = NOTEBOOK_PATH.with_suffix(NOTEBOOK_PATH.suffix + ".tmp")
    nbformat.write(notebook, temporary)
    os.replace(temporary, NOTEBOOK_PATH)


def _verify(alpha, token, executed, smoke):
    expected_suffix = "_smoke.pkl" if smoke else "_100trials.pkl"
    artifact = ROOT / (
        "results/M3d_redo_delta6_alpha"
        f"{token}_glmbsa2-5-10_thirteen_methods_folds2-5-10_"
        "widegrid_c0018-018-18_C003-03-3_seed42"
        f"{expected_suffix}"
    )
    if not artifact.is_file() or artifact.stat().st_size == 0:
        raise RuntimeError(f"Missing result artifact: {artifact}")
    for cell in executed.cells[1:]:
        display_count = sum(output.output_type == "display_data" for output in cell.outputs)
        if display_count != 7:
            raise RuntimeError(f"{cell.id}: expected 7 displayed plots, found {display_count}")


def main():
    smoke = os.environ.get("WERM_SMOKE") == "1"
    for alpha, token in SCENARIOS:
        print(f"Starting delta=6 alpha={alpha:g} ({'smoke' if smoke else '100 trials'})", flush=True)
        notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
        cells = _extension_cells(notebook, alpha)
        executed = _execute(cells, notebook.metadata)
        _verify(alpha, token, executed, smoke)
        if not smoke:
            _merge_outputs(alpha, executed)
            print(f"Embedded delta=6 alpha={alpha:g} results and 14 plots into redo3", flush=True)
        else:
            print(f"Validated delta=6 alpha={alpha:g} smoke run and plots", flush=True)


if __name__ == "__main__":
    main()
