"""Append self-contained alpha=0.3 and alpha=0.1 M3d notebook cells."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import nbformat


ALPHA_CONFIGS = ((0.3, "03"), (0.1, "01"))


def _replace_once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"expected exactly one source anchor, found {count}: {old[:100]!r}"
        )
    return source.replace(old, new, 1)


def _experiment_source(base_source: str, alpha: float, tag: str) -> str:
    source = base_source
    source = _replace_once(source, "    α = 0.5", "    α = signal_alpha")
    source = _replace_once(
        source,
        "fixed_delta = 3.0 # Fixed delta value",
        f"fixed_delta = 3.0 # Fixed delta value\nsignal_alpha = {alpha}",
    )
    source = _replace_once(source, "'schema_version': 9,", "'schema_version': 10,")
    source = _replace_once(
        source,
        "'experiment': 'M3d-redo-eleven-method-logistic-fold-comparison',",
        f"'experiment': 'M3d-redo-alpha{tag}-eleven-method-logistic-fold-comparison',",
    )
    source = _replace_once(
        source,
        "    'fixed_delta': float(fixed_delta),",
        "    'fixed_delta': float(fixed_delta),\n"
        "    'alpha': float(signal_alpha),",
    )
    source = _replace_once(
        source,
        "f'M3_upsample_d_redo_full_cv_delta{fixed_delta:g}_{checkpoint_mode}_seed{random_seed}_p{p}_L{L}_schema{checkpoint_signature[\"schema_version\"]}',",
        "f'M3_upsample_d_redo_full_cv_delta{fixed_delta:g}_alpha{signal_alpha:g}_{checkpoint_mode}_seed{random_seed}_p{p}_L{L}_schema{checkpoint_signature[\"schema_version\"]}',",
    )
    source = _replace_once(
        source,
        "artifact_stem = 'results/M3d_redo_delta3_eleven_methods_folds2-5-10_seed42'",
        f"artifact_stem = 'results/M3d_redo_delta3_alpha{tag}_eleven_methods_folds2-5-10_seed42'",
    )
    source = _replace_once(
        source,
        "        'fixed_delta': fixed_delta,",
        "        'fixed_delta': fixed_delta,\n"
        "        'signal_alpha': signal_alpha,",
    )
    return source


def _plot_source(base_source: str, alpha: float, tag: str) -> str:
    source = base_source
    source = _replace_once(
        source,
        "filename_to_load = 'results/M3d_redo_delta3_eleven_methods_folds2-5-10_seed42_100trials.pkl'",
        f"filename_to_load = 'results/M3d_redo_delta3_alpha{tag}_eleven_methods_folds2-5-10_seed42_100trials.pkl'",
    )
    source = _replace_once(
        source,
        "    fixed_delta = params['fixed_delta']",
        "    fixed_delta = params['fixed_delta']\n"
        f"    signal_alpha = params.get('signal_alpha', {alpha})",
    )
    source = _replace_once(
        source,
        'print(f"Using palette: {PAL[:4]} (Paul Tol Vibrant)")',
        f'print("Plotting M3d delta=3, alpha={alpha}")\n'
        'print(f"Using palette: {PAL[:4]} (Paul Tol Vibrant)")',
    )
    return source


def append_alpha_extensions(notebook):
    """Return a copy with the two experiment/plot pairs appended exactly once."""
    result = copy.deepcopy(notebook)
    existing = {
        cell.metadata.get("m3_alpha_experiment")
        for cell in result.cells
        if cell.cell_type == "code"
    }
    missing = [config for config in ALPHA_CONFIGS if config[0] not in existing]
    if not missing:
        return result

    base_experiment = result.cells[2].source
    base_plot = result.cells[4].source
    for alpha, tag in missing:
        heading = nbformat.v4.new_markdown_cell(
            f"## M3d redo at $\\delta=3$, $\\alpha={alpha}$\n\n"
            "The following experiment and plot cells are self-contained and use "
            "the same eleven-method comparison as the $\\alpha=0.5$ experiment."
        )
        heading.metadata["m3_alpha_heading"] = alpha

        experiment = nbformat.v4.new_code_cell(
            _experiment_source(base_experiment, alpha, tag)
        )
        experiment.metadata["m3_alpha_experiment"] = alpha

        plot = nbformat.v4.new_code_cell(_plot_source(base_plot, alpha, tag))
        plot.metadata["m3_alpha_plot"] = alpha
        result.cells.extend((heading, experiment, plot))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="M3_upsample_redo2.ipynb")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    source_path = Path(args.source)
    output_path = Path(args.output) if args.output else source_path
    notebook = nbformat.read(source_path, as_version=4)
    nbformat.write(append_alpha_extensions(notebook), output_path)


if __name__ == "__main__":
    main()
