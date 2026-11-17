import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "M3_upsample_final.ipynb"
METHODS = ("GLMBSA-CV", "WERM-L1-fullCV-2fold")


def _source(cell):
    return "".join(cell.get("source", []))


def test_final_notebook_has_only_requested_scenarios_and_cell_order():
    notebook = json.loads(NOTEBOOK.read_text())
    scenario_cells = [cell for cell in notebook["cells"] if cell.get("metadata", {}).get("m3_final")]
    assert [(cell["metadata"]["delta"], cell["metadata"]["alpha"], cell["metadata"]["kind"])
            for cell in scenario_cells] == [
        (3.0, 0.1, "heading"),
        (3.0, 0.1, "experiment"),
        (3.0, 0.1, "plot"),
        (6.0, 0.1, "heading"),
        (6.0, 0.1, "experiment"),
        (6.0, 0.1, "plot"),
    ]


def test_experiment_cells_run_only_the_two_requested_two_fold_methods():
    notebook = json.loads(NOTEBOOK.read_text())
    experiments = [cell for cell in notebook["cells"] if cell.get("metadata", {}).get("kind") == "experiment"]
    assert len(experiments) == 2
    for cell, delta in zip(experiments, (3.0, 6.0)):
        source = _source(cell)
        method_block = source.split("methods_to_evaluate = [", 1)[1].split("\n]", 1)[0]
        assert tuple(cell["metadata"]["methods"]) == METHODS
        assert method_block.count("'name':") == 2
        assert "'name': 'GLMBSA-CV'" in method_block
        assert "'name': 'WERM-L1-fullCV-2fold'" in method_block
        assert "num_trials = 100" in source
        assert "os.environ.get('WERM_N_JOBS', '36')" in source
        assert f"fixed_delta = {delta:.1f}" in source
        assert "signal_alpha = 0.1" in source
        assert "glmbsa_cv_folds = {'GLMBSA-CV': 2}" in source
        assert "'glmbsa_cv_folds': (2,)" in source
        assert "final_two_methods" in source
        assert "'schema_version': 14" in source


def test_plot_cells_have_reference_colors_labels_and_complete_suite():
    notebook = json.loads(NOTEBOOK.read_text())
    plots = [cell for cell in notebook["cells"] if cell.get("metadata", {}).get("kind") == "plot"]
    assert len(plots) == 2
    for cell in plots:
        source = _source(cell)
        method_block = source.split("methods_to_evaluate = [", 1)[1].split("\n]", 1)[0]
        assert "'name': 'GLMBSA-CV'" in method_block
        assert "'label': 'GLMBSA'" in method_block
        assert "'color': '#E69F00'" in method_block
        assert "'marker': 'p'" in method_block
        assert "'name': 'WERM-L1-fullCV-2fold'" in method_block
        assert "'label': r'\\textbf{WERM+L1}'" in method_block
        assert "method['label'] = r'\\textbf{WERM+L1}'" in source
        assert "method['label'] = r'\\textbf{WERM}'" not in source
        assert "'color': 'black'" in method_block
        assert "'linestyle': '-'" in method_block
        assert "'marker': 'D'" in method_block
        assert len({'p', 'D'}) == 2
        assert "if method['name'] == 'WERM-L1-fullCV-2fold':" in source
        assert "ax.set_xlabel(KAPPA_SQUARED_LABEL" in source
        assert "x_label=KAPPA_SQUARED_LABEL" in source
        assert r"KAPPA_SQUARED_LABEL = r'$\kappa\textsuperscript{2}$'" in source
        assert method_block.count("'name':") == 2
        for function_name in (
            "plot_location_asymmetric_errors",
            "plot_size_asymmetric_errors",
            "plot_runtime_asymmetric_errors",
            "plot_location_boxplots",
            "plot_size_boxplots",
            "plot_runtime_boxplots",
            "plot_changepoint_histograms_publication",
        ):
            assert f"{function_name}(" in source
        assert cell.get("execution_count") is not None
        assert sum(output.get("output_type") == "display_data" for output in cell.get("outputs", [])) == 7


def test_runtime_axes_use_location_axis_tick_format():
    notebook = json.loads(NOTEBOOK.read_text())
    plots = [cell for cell in notebook["cells"] if cell.get("metadata", {}).get("kind") == "plot"]
    for cell in plots:
        source = _source(cell)
        runtime_line = source.split("def plot_runtime_asymmetric_errors(", 1)[1].split(
            "# --- Distribution Plotting Functions", 1
        )[0]
        runtime_box = source.split("def plot_runtime_boxplots(", 1)[1].split(
            "# -----------------------------------------------------------------------------", 1
        )[0]
        assert "ax.set_xscale('log')" in runtime_line
        assert "ax.set_xticks(np.array([10, 100, 1000]), minor=False)" in runtime_line
        assert "ax.set_xticklabels([r'$10$', r'$100$', r'$1000$']" in runtime_line
        assert 'tick_label_format=".0f"' in runtime_box


def test_boxplot_jitter_is_faint_enough_to_leave_boxes_visible():
    notebook = json.loads(NOTEBOOK.read_text())
    plots = [cell for cell in notebook["cells"] if cell.get("metadata", {}).get("kind") == "plot"]
    for cell in plots:
        source = _source(cell)
        assert "point_alpha=0.16," in source
        assert "point_alpha=0.38," not in source


def test_location_runtime_boxplots_have_sparse_ticks_and_runtime_has_no_legend():
    notebook = json.loads(NOTEBOOK.read_text())
    for cell in notebook['cells']:
        if cell.get('metadata', {}).get('kind') != 'plot':
            continue
        source = _source(cell)
        assert 'tick_stride = 2 if metric_prefix in ("haus", "runtime") else 1' in source
        assert 'ax.set_xticks(kappa_positions[::tick_stride])' in source
        assert 'for k in current_kappa_list[::tick_stride]]' in source
        runtime = source.split('def plot_runtime_boxplots(', 1)[1].split('# -----------------------------------------------------------------------------', 1)[0]
        assert 'show_legend=False,' in runtime
        assert 'show_legend=True,' not in runtime


def test_plot_presentation_matches_m2_final_canonical_format():
    notebook = json.loads(NOTEBOOK.read_text())
    plots = [cell for cell in notebook["cells"] if cell.get("metadata", {}).get("kind") == "plot"]
    for cell in plots:
        source = _source(cell)
        for setting in (
            "'font.size': 14",
            "'axes.labelsize': 18",
            "'axes.titlesize': 18",
            "'xtick.labelsize': 14",
            "'ytick.labelsize': 14",
            "'legend.fontsize': 14",
            "'figure.dpi': 72",
            "'savefig.dpi': 300",
            "x_y_label_fontsize = 18",
        ):
            assert setting in source
        assert source.count("fig, ax = plt.subplots(1, 1, figsize=(5.0, 4.0))") >= 3
        grouped = source.split("def _plot_grouped_distributions(", 1)[1].split(
            "def plot_location_boxplots(", 1
        )[0]
        assert "fig, ax = plt.subplots(1, 1, figsize=(5.0, 4.0))" in grouped
        assert "fontsize=16," in grouped
        for start, end in (
            ("def plot_location_asymmetric_errors(", "def plot_size_asymmetric_errors("),
            ("def plot_size_asymmetric_errors(", "def plot_runtime_asymmetric_errors("),
            ("def plot_runtime_asymmetric_errors(", "# --- Distribution Plotting Functions"),
        ):
            block = source.split(start, 1)[1].split(end, 1)[0]
            assert "fontsize=18" in block
            assert "labelsize=14" in block
        histogram = source.split("def plot_changepoint_histograms_publication(", 1)[1]
        assert "ax.set_xticklabels(['$0.33$', '$0.53$'], fontsize=14)" in histogram
        assert "fontsize=x_y_label_fontsize, fontweight='bold'" in histogram
        assert histogram.count("fontsize=x_y_label_fontsize)") >= 2
