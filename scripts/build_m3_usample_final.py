"""Build the self-contained two-scenario, two-method final M3 notebook."""

from __future__ import annotations

import copy
import os
import pickle
import re
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from tools.final_notebook_presentation import normalize_presentation


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "M3_upsample_redo3.ipynb"
TARGET = ROOT / "M3_upsample_final.ipynb"
METHODS = ("GLMBSA-CV", "WERM-L1-fullCV-2fold")
METHOD_BLOCK = """methods_to_evaluate = [
    {'name': 'GLMBSA-CV', 'label': 'GLMBSA', 'color': '#E69F00', 'marker': 'p', 'linestyle': ':'},
    {'name': 'WERM-L1-fullCV-2fold', 'label': r'\\textbf{WERM+L1}', 'color': 'black', 'marker': 'D', 'linestyle': '-'},
]"""
SCENARIOS = (
    {
        "delta": 3.0,
        "alpha": 0.1,
        "token": "d3-a01",
        "source_fragment": "M3d_redo_delta3_alpha01_",
        "reference_artifact": "results/M3d_redo_delta3_alpha01_glmbsa2-5-10_thirteen_methods_folds2-5-10_widegrid_c0018-018-18_C003-03-3_seed42_100trials.pkl",
        "target_stem": "results/M3_usample_final_delta3_alpha01_GLMBSA_WERM-L1_fullCV2_seed42",
    },
    {
        "delta": 6.0,
        "alpha": 0.1,
        "token": "d6-a01",
        "source_fragment": "M3d_redo_delta6_alpha01_",
        "reference_artifact": "results/M3d_redo_delta6_alpha01_glmbsa2-5-10_thirteen_methods_folds2-5-10_widegrid_c0018-018-18_C003-03-3_seed42_100trials.pkl",
        "target_stem": "results/M3_usample_final_delta6_alpha01_GLMBSA_WERM-L1_fullCV2_seed42",
    },
)


def _source(cell):
    return cell.source if isinstance(cell.source, str) else "".join(cell.source)


def _replace_method_block(source):
    start = source.index("methods_to_evaluate = [")
    end = source.index("\n]\n", start) + 2
    return source[:start] + METHOD_BLOCK + source[end:]


def _experiment_source(source, scenario):
    source = _replace_method_block(source)
    source = source.replace(
        "glmbsa_cv_folds = {'GLMBSA-CV': 2, 'GLMBSA-CV-5fold': 5, 'GLMBSA-CV-10fold': 10}",
        "glmbsa_cv_folds = {'GLMBSA-CV': 2}",
        1,
    )
    source = source.replace("'schema_version': 13", "'schema_version': 14", 1)
    source = re.sub(
        r"    'experiment': '[^']+',",
        "    'experiment': 'M3-usample-final-two-methods',",
        source,
        count=1,
    )
    source = source.replace("'glmbsa_cv_folds': (2, 5, 10)", "'glmbsa_cv_folds': (2,)", 1)
    source = re.sub(
        r"f'M3_upsample_d_redo_full_cv_delta\{fixed_delta:g\}[^']+schema\{checkpoint_signature\[\"schema_version\"\]\}',",
        "f'M3_usample_final_delta{fixed_delta:g}_alpha{signal_alpha:g}_{checkpoint_mode}_seed{random_seed}_p{p}_L{L}_final_two_methods_schema{checkpoint_signature[\"schema_version\"]}',",
        source,
        count=1,
    )

    legacy_start = source.index("rerun_methods = ")
    legacy_end = source.index("completed_results = {}", legacy_start)
    source = (
        source[:legacy_start]
        + "legacy_results = {}\n"
        + "print('Final two-method run: isolated checkpoints; no external checkpoint reuse')\n\n"
        + source[legacy_end:]
    )
    source = source.replace(
        "Resuming M3d redo thirteen-method comparison",
        "Resuming M3 final two-method comparison",
        1,
    )
    source = re.sub(
        r"artifact_stem = 'results/[^']+'",
        f"artifact_stem = '{scenario['target_stem']}'",
        source,
        count=1,
    )
    return source


def _plot_source(source, scenario):
    source = _replace_method_block(source)
    # Keep the box and median marks prominent while retaining every raw trial.
    source = source.replace("point_alpha=0.38,", "point_alpha=0.16,")
    source = source.replace(
        METHOD_BLOCK,
        METHOD_BLOCK + "\nKAPPA_SQUARED_LABEL = r'$\\kappa\\textsuperscript{2}$'",
        1,
    )
    source = source.replace(
        "if method['name'] == 'WERM-CV':",
        "if method['name'] == 'WERM-L1-fullCV-2fold':",
    )
    source = source.replace(
        "method['label'] = r'\\textbf{WERM}'",
        "method['label'] = r'\\textbf{WERM+L1}'",
    )
    source = source.replace(
        r"ax.set_xlabel(r'$\kappa\textsuperscript{2}$',",
        "ax.set_xlabel(KAPPA_SQUARED_LABEL,",
    )
    source = source.replace(
        r"ax.set_xlabel(r'$\kappa^2$',",
        "ax.set_xlabel(KAPPA_SQUARED_LABEL,",
    )
    source = source.replace(
        r"x_label=r'$\kappa^2$',",
        "x_label=KAPPA_SQUARED_LABEL,",
    )
    runtime_start = source.index("def plot_runtime_asymmetric_errors(")
    runtime_end = source.index("# --- Distribution Plotting Functions", runtime_start)
    runtime_line = source[runtime_start:runtime_end]
    runtime_line = runtime_line.replace(
        "    ax.set_ylim(bottom=0)",
        "    ax.set_xscale('log')\n    ax.set_ylim(bottom=0)",
        1,
    ).replace(
        "    ax.set_xticks(current_kappa_list)",
        "    ax.set_xticks(np.array([10, 100, 1000]), minor=False)\n"
        "    ax.set_xticklabels([r'$10$', r'$100$', r'$1000$'], fontsize=18)",
        1,
    )
    source = source[:runtime_start] + runtime_line + source[runtime_end:]

    runtime_box_start = source.index("def plot_runtime_boxplots(")
    runtime_box_end = source.index(
        "# -----------------------------------------------------------------------------", runtime_box_start
    )
    runtime_box = source[runtime_box_start:runtime_box_end].replace(
        'tick_label_format=".2f"', 'tick_label_format=".0f"', 1
    )
    runtime_box = runtime_box.replace('show_legend=True,', 'show_legend=False,', 1)
    source = source[:runtime_box_start] + runtime_box + source[runtime_box_end:]
    source = source.replace(
        '    ax.set_xticks(kappa_positions)\n',
        '    tick_stride = 2 if metric_prefix in ("haus", "runtime") else 1\n'
        '    ax.set_xticks(kappa_positions[::tick_stride])\n',
        1,
    ).replace(
        '[format(float(k), tick_label_format) for k in current_kappa_list]',
        '[format(float(k), tick_label_format) for k in current_kappa_list[::tick_stride]]',
        1,
    )

    # Match the corresponding publication formatting in M2 final.  These
    # substitutions intentionally touch presentation only.
    def restyle_block(text, start_marker, end_marker, replacements):
        start = text.index(start_marker)
        end = text.index(end_marker, start)
        block = text[start:end]
        for old, new, count in replacements:
            block = block.replace(old, new, count)
        return text[:start] + block + text[end:]

    source = restyle_block(
        source,
        "def plot_location_asymmetric_errors(",
        "def plot_size_asymmetric_errors(",
        [
            ("fig, ax = plt.subplots(1, 1, figsize=(7, 5))",
             "fig, ax = plt.subplots(1, 1, figsize=(5.0, 4.0))", 1),
            ("fontsize=24", "fontsize=x_y_label_fontsize", 4),
            ("labelsize=24", "labelsize=x_y_label_fontsize", 1),
        ],
    )
    source = restyle_block(
        source,
        "def plot_size_asymmetric_errors(",
        "def plot_runtime_asymmetric_errors(",
        [
            ("fig, ax = plt.subplots(1, 1, figsize=(7, 5))",
             "fig, ax = plt.subplots(1, 1, figsize=(5.0, 4.0))", 1),
            ("prop={'size': 20}", "prop={'size': 16}", 1),
        ],
    )
    source = restyle_block(
        source,
        "def plot_runtime_asymmetric_errors(",
        "# --- Distribution Plotting Functions",
        [
            ("fig, ax = plt.subplots(1, 1, figsize=(7, 5))",
             "fig, ax = plt.subplots(1, 1, figsize=(5.0, 4.0))", 1),
            ("fontsize=18", "fontsize=x_y_label_fontsize", 3),
            ("width=1.5, length=6)",
             "width=1.5, length=6, labelsize=x_y_label_fontsize)", 1),
            ("prop={'size': 17}", "prop={'size': 16}", 1),
        ],
    )
    source = restyle_block(
        source,
        "def _plot_grouped_distributions(",
        "def plot_location_boxplots(",
        [
            ("fig, ax = plt.subplots(1, 1, figsize=(7, 5))",
             "fig, ax = plt.subplots(1, 1, figsize=(5.0, 4.0))", 1),
            ("            fontsize=14,", "            fontsize=16,", 1),
        ],
    )
    histogram_start = source.index("def plot_changepoint_histograms_publication(")
    histogram = source[histogram_start:]
    histogram = histogram.replace(
        "ax.set_xticklabels(['$0.33$', '$0.53$'], fontsize=10)",
        "ax.set_xticklabels(['$0.33$', '$0.53$'], fontsize=12)",
        1,
    ).replace(
        "fontsize=20, fontweight='bold'",
        "fontsize=x_y_label_fontsize, fontweight='bold'",
        1,
    ).replace(
        "fontsize=20)",
        "fontsize=x_y_label_fontsize)",
        2,
    )
    source = source[:histogram_start] + histogram
    source = re.sub(
        r"FIGURES_DIR = '[^']+'",
        "FIGURES_DIR = 'results/M3_usample_final_plots'",
        source,
        count=1,
    )
    source = re.sub(
        r"filename_to_load = 'results/[^']+_100trials\.pkl'",
        f"filename_to_load = '{scenario['target_stem']}_100trials.pkl'",
        source,
        count=1,
    )
    return normalize_presentation(source)


def _method_field(key):
    return any(key.endswith(f"_{method}") for method in METHODS)


def _write_two_method_artifact(scenario, method_configs):
    reference_path = ROOT / scenario["reference_artifact"]
    with reference_path.open("rb") as handle:
        payload = pickle.load(handle)
    original = payload["results"]
    results = {
        key: value
        for key, value in original.items()
        if key in {"kappa_values", "n_values", "true_chgpts"} or _method_field(key)
    }
    results["successful_trials"] = {
        method: original["successful_trials"][method] for method in METHODS
    }
    results["summaries"] = [
        summary for summary in original["summaries"] if summary.get("method") in METHODS
    ]
    results["trial_records"] = [
        (
            setting_idx,
            trial_idx,
            {
                key: value
                for key, value in record.items()
                if key == "data_hash" or _method_field(key)
            },
        )
        for setting_idx, trial_idx, record in original["trial_records"]
    ]
    params = copy.deepcopy(payload["params"])
    params["methods_evaluated"] = copy.deepcopy(method_configs)
    params["checkpoint_signature"] = copy.deepcopy(params["checkpoint_signature"])
    params["checkpoint_signature"].update(
        schema_version=14,
        experiment="M3-usample-final-two-methods",
        methods=METHODS,
        glmbsa_cv_folds=(2,),
    )
    target = ROOT / f"{scenario['target_stem']}_100trials.pkl"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        pickle.dump({"results": results, "params": params}, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    source_notebook = nbformat.read(SOURCE, as_version=4)
    cells = [
        nbformat.v4.new_markdown_cell(
            "# M3 upsample final: GLMBSA versus WERM+L1\n\n"
            "Two logistic change-point experiments with $\\alpha=0.1$. Each experiment cell runs "
            "only GLMBSA full CV (2-fold) and WERM Logistic+L1 full CV (2-fold), using 100 trials "
            "and up to 36 parallel jobs. Its following cell loads the saved result and produces the "
            "complete publication-style plot suite."
        )
    ]
    cells[0].id = "m3-final-title"

    plot_cells = []
    for scenario in SCENARIOS:
        delta = scenario["delta"]
        alpha = scenario["alpha"]
        experiment_template = next(
            cell for cell in source_notebook.cells
            if cell.cell_type == "code"
            and _source(cell).startswith(f"# redo3 scenario: delta={delta:g}, alpha={alpha:g}")
        )
        plot_template = next(
            cell for cell in source_notebook.cells
            if cell.cell_type == "code"
            and cell.metadata.get("m3_two_method_replot")
            and scenario["source_fragment"] in _source(cell)
        )

        heading = nbformat.v4.new_markdown_cell(
            f"## $\\delta={delta:g}$, $\\alpha={alpha:g}$"
        )
        heading.id = f"m3-final-{scenario['token']}-heading"
        heading.metadata.update(m3_final=True, delta=delta, alpha=alpha, kind="heading")

        experiment = nbformat.v4.new_code_cell(_experiment_source(_source(experiment_template), scenario))
        experiment.id = f"m3-final-{scenario['token']}-experiment"
        experiment.metadata.update(
            m3_final=True, delta=delta, alpha=alpha, kind="experiment", methods=list(METHODS)
        )

        plot = nbformat.v4.new_code_cell(_plot_source(_source(plot_template), scenario))
        plot.id = f"m3-final-{scenario['token']}-plot"
        plot.metadata.update(
            m3_final=True, delta=delta, alpha=alpha, kind="plot", methods=list(METHODS)
        )
        cells.extend((heading, experiment, plot))
        plot_cells.append(plot)
        _write_two_method_artifact(scenario, [
            {'name': 'GLMBSA-CV', 'label': 'GLMBSA', 'color': '#E69F00', 'marker': 'p', 'linestyle': ':'},
            {'name': 'WERM-L1-fullCV-2fold', 'label': r'\textbf{WERM+L1}', 'color': 'black', 'marker': 'D', 'linestyle': '-'},
        ])

    final_notebook = nbformat.v4.new_notebook(cells=cells, metadata=copy.deepcopy(source_notebook.metadata))
    plot_notebook = nbformat.v4.new_notebook(
        cells=[copy.deepcopy(cell) for cell in plot_cells],
        metadata=copy.deepcopy(source_notebook.metadata),
    )
    kernel_manager = KernelManager(kernel_name="python3")
    kernel_manager.kernel_spec.argv[0] = sys.executable
    executed = NotebookClient(
        plot_notebook,
        timeout=None,
        km=kernel_manager,
        resources={"metadata": {"path": str(ROOT)}},
    ).execute()
    executed_by_id = {cell.id: cell for cell in executed.cells}
    for plot in plot_cells:
        completed = executed_by_id[plot.id]
        plot.outputs = completed.outputs
        plot.execution_count = completed.execution_count

    temporary = TARGET.with_suffix(TARGET.suffix + ".tmp")
    nbformat.write(final_notebook, temporary)
    os.replace(temporary, TARGET)


if __name__ == "__main__":
    main()
