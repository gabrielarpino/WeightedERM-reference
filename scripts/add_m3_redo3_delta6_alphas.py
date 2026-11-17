"""Append delta=6, alpha=0.1/0.3 experiment and plot cells to redo3."""

from pathlib import Path
import os

import nbformat


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "M3_upsample_redo3.ipynb"
SCENARIOS = ((0.1, "01"), (0.3, "03"))


def _source(cell):
    return cell.source if isinstance(cell.source, str) else "".join(cell.source)


def _find_one(cells, predicate, description):
    matches = [cell for cell in cells if predicate(cell)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {description}, found {len(matches)}")
    return matches[0]


def _experiment_source(source, alpha, token):
    old_marker = f"# redo3 scenario: delta=3, alpha={alpha:g}"
    new_marker = f"# redo3 scenario: delta=6, alpha={alpha:g}"
    if source.count(old_marker) != 1 or source.count("fixed_delta = 3.0") != 1:
        raise RuntimeError(f"Unexpected experiment template for alpha={alpha:g}")
    source = source.replace(old_marker, new_marker, 1)
    source = source.replace("fixed_delta = 3.0", "fixed_delta = 6.0", 1)
    source = source.replace(
        f"M3d_redo_delta3_alpha{token}_",
        f"M3d_redo_delta6_alpha{token}_",
    )
    source = source.replace(
        f"M3d-redo-alpha{token}-",
        f"M3d-redo-delta6-alpha{token}-",
    )
    return source


def _plot_source(source, token):
    old = f"M3d_redo_delta3_alpha{token}_"
    new = f"M3d_redo_delta6_alpha{token}_"
    if old not in source:
        raise RuntimeError(f"Plot template does not contain {old}")
    return source.replace(old, new)


def main():
    notebook = nbformat.read(PATH, as_version=4)
    notebook.cells = [
        cell for cell in notebook.cells if not cell.metadata.get("m3_delta6_extension")
    ]
    template_cells = list(notebook.cells)
    appended = []

    for alpha, token in SCENARIOS:
        artifact_fragment = f"M3d_redo_delta3_alpha{token}_"
        experiment_template = _find_one(
            template_cells,
            lambda cell: cell.cell_type == "code"
            and _source(cell).startswith(f"# redo3 scenario: delta=3, alpha={alpha:g}"),
            f"alpha={alpha:g} experiment template",
        )
        full_plot_template = _find_one(
            template_cells,
            lambda cell: cell.cell_type == "code"
            and _source(cell).startswith("from sklearn.neighbors import KernelDensity")
            and artifact_fragment in _source(cell)
            and not cell.metadata.get("m3_two_method_replot"),
            f"alpha={alpha:g} full plot template",
        )
        two_plot_template = _find_one(
            template_cells,
            lambda cell: cell.cell_type == "code"
            and artifact_fragment in _source(cell)
            and cell.metadata.get("m3_two_method_replot"),
            f"alpha={alpha:g} two-method plot template",
        )

        heading = nbformat.v4.new_markdown_cell(
            f"## M3d redo3 at $\\delta=6$, $\\alpha={alpha:g}$\n\n"
            "The 100-trial experiment below uses 36 parallel jobs. The complete plot suite and the "
            "two-method, 2-fold full-CV comparison follow immediately afterward."
        )
        heading.id = f"m3redo3-d6-a{token}-heading"
        heading.metadata.update(m3_delta6_extension=True, alpha=alpha, kind="heading")

        experiment = nbformat.v4.new_code_cell(_experiment_source(_source(experiment_template), alpha, token))
        experiment.id = f"m3redo3-d6-a{token}-experiment"
        experiment.metadata.update(m3_delta6_extension=True, alpha=alpha, kind="experiment")

        full_plot = nbformat.v4.new_code_cell(_plot_source(_source(full_plot_template), token))
        full_plot.id = f"m3redo3-d6-a{token}-full-plot"
        full_plot.metadata.update(m3_delta6_extension=True, alpha=alpha, kind="full_plot")

        two_plot = nbformat.v4.new_code_cell(_plot_source(_source(two_plot_template), token))
        two_plot.id = f"m3redo3-d6-a{token}-two-method-plot"
        two_plot.metadata.update(
            m3_delta6_extension=True,
            m3_two_method_replot=True,
            alpha=alpha,
            kind="two_method_plot",
            selected_methods=["GLMBSA-CV", "WERM-L1-fullCV-2fold"],
        )
        appended.extend((heading, experiment, full_plot, two_plot))

    notebook.cells.extend(appended)
    temporary = PATH.with_suffix(PATH.suffix + ".tmp")
    nbformat.write(notebook, temporary)
    os.replace(temporary, PATH)


if __name__ == "__main__":
    main()
