"""Add and execute two-method, two-fold replots without rerunning experiments."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = {
    "M3_upsample_redo2.ipynb": 3,
    "M3_upsample_redo3.ipynb": 4,
}
SELECTED_METHODS = ("GLMBSA-CV", "WERM-L1-fullCV-2fold")
METHOD_BLOCK = """methods_to_evaluate = [
    {'name': 'GLMBSA-CV', 'label': r'GLMBSA full CV (2-fold)', 'color': '#E69F00', 'marker': 'p', 'linestyle': ':'},
    {'name': 'WERM-L1-fullCV-2fold', 'label': r'WERM Logistic+L1 full CV (2-fold)', 'color': '#EE3377', 'marker': '^', 'linestyle': '-.'},
]"""


def _two_method_source(source: str) -> str:
    block_start = source.index("methods_to_evaluate = [")
    block_end_marker = "\n\nmethod_colors_map ="
    block_end = source.index(block_end_marker, block_start)
    source = source[:block_start] + METHOD_BLOCK + source[block_end:]

    old_assertion = "assert [method['name'] for method in params['methods_evaluated']] == method_names_list"
    new_assertion = (
        "available_method_names = {method['name'] for method in params['methods_evaluated']}\n"
        "    assert set(method_names_list).issubset(available_method_names)"
    )
    if old_assertion not in source:
        raise ValueError("Expected loaded-method assertion was not found")
    source = source.replace(old_assertion, new_assertion, 1)

    filename_line = (
        "filename_base_for_plots = 'final_M3_upsample_format_' + "
        "os.path.basename(filename_to_load).replace('.pkl', '')"
    )
    if filename_line not in source:
        raise ValueError("Expected plot filename definition was not found")
    source = source.replace(
        filename_line,
        filename_line + " + '_two_method_2fold'",
        1,
    )
    return source


def _is_original_plot(cell) -> bool:
    source = cell.source if isinstance(cell.source, str) else "".join(cell.source)
    return (
        cell.cell_type == "code"
        and source.startswith("from sklearn.neighbors import KernelDensity")
        and not cell.metadata.get("m3_two_method_replot")
    )


def _prepare_notebook(path: Path, expected_count: int):
    notebook = nbformat.read(path, as_version=4)
    notebook.cells = [
        cell
        for cell in notebook.cells
        if not cell.metadata.get("m3_two_method_replot")
        and not cell.metadata.get("m3_two_method_replot_heading")
    ]
    original_plot_indices = [index for index, cell in enumerate(notebook.cells) if _is_original_plot(cell)]
    if len(original_plot_indices) != expected_count:
        raise ValueError(f"{path.name}: expected {expected_count} plot cells, found {len(original_plot_indices)}")

    new_cells = []
    for experiment_index, original_index in enumerate(original_plot_indices, start=1):
        original = notebook.cells[original_index]
        source = _two_method_source(original.source)
        cell = nbformat.v4.new_code_cell(source=source)
        cell.id = f"m3-two-method-{experiment_index:02d}-plot"
        cell.metadata["m3_two_method_replot"] = True
        cell.metadata["selected_methods"] = list(SELECTED_METHODS)
        new_cells.append(cell)

    execution_notebook = nbformat.v4.new_notebook(
        cells=[copy.deepcopy(cell) for cell in new_cells],
        metadata=copy.deepcopy(notebook.metadata),
    )
    executed = NotebookClient(
        execution_notebook,
        timeout=None,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    ).execute()

    executed_by_id = {cell.id: cell for cell in executed.cells}
    for cell in new_cells:
        completed = executed_by_id[cell.id]
        cell.outputs = completed.outputs
        cell.execution_count = completed.execution_count

    for original_index, new_cell, experiment_index in reversed(
        list(zip(original_plot_indices, new_cells, range(1, expected_count + 1)))
    ):
        heading = nbformat.v4.new_markdown_cell(
            source="### Two-method comparison: 2-fold full CV only"
        )
        heading.id = f"m3-two-method-{experiment_index:02d}-heading"
        heading.metadata["m3_two_method_replot_heading"] = True
        notebook.cells[original_index + 1:original_index + 1] = [heading, new_cell]

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    nbformat.write(notebook, temporary_path)
    os.replace(temporary_path, path)


def main():
    for notebook_name, expected_count in NOTEBOOKS.items():
        print(f"Adding and executing {expected_count} two-method replots in {notebook_name}", flush=True)
        _prepare_notebook(ROOT / notebook_name, expected_count)


if __name__ == "__main__":
    main()
