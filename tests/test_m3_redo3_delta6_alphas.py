import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "M3_upsample_redo3.ipynb"


def _source(cell):
    return "".join(cell.get("source", []))


def test_delta6_alpha_extensions_are_serial_and_complete():
    notebook = json.loads(NOTEBOOK.read_text())
    extension_cells = [
        cell for cell in notebook["cells"] if cell.get("metadata", {}).get("m3_delta6_extension")
    ]

    assert [(cell["metadata"]["alpha"], cell["metadata"]["kind"]) for cell in extension_cells] == [
        (0.1, "heading"),
        (0.1, "experiment"),
        (0.1, "full_plot"),
        (0.1, "two_method_plot"),
        (0.3, "heading"),
        (0.3, "experiment"),
        (0.3, "full_plot"),
        (0.3, "two_method_plot"),
    ]

    experiments = [cell for cell in extension_cells if cell["metadata"]["kind"] == "experiment"]
    for cell, alpha, alpha_token in zip(experiments, (0.1, 0.3), ("alpha01", "alpha03")):
        source = _source(cell)
        assert "fixed_delta = 6.0" in source
        assert f"signal_alpha = {alpha}" in source
        assert "num_trials = 100" in source
        assert "os.environ.get('WERM_N_JOBS', '36')" in source
        assert "glmbsa_c1_candidates = (0.018, 0.18, 1.8)" in source
        assert "joint_l1_constants = (0.03, 0.3, 3.0)" in source
        assert f"M3d_redo_delta6_{alpha_token}_" in source
        assert "schema13" not in source  # schema is generated from the explicit signature field
        assert "'schema_version': 13" in source

    plots = [cell for cell in extension_cells if "plot" in cell["metadata"]["kind"]]
    for cell in plots:
        source = _source(cell)
        assert "plot_location_asymmetric_errors(" in source
        assert "plot_size_asymmetric_errors(" in source
        assert "plot_runtime_asymmetric_errors(" in source
        assert "plot_location_boxplots(" in source
        assert "plot_size_boxplots(" in source
        assert "plot_runtime_boxplots(" in source
        assert "plot_changepoint_histograms_publication(" in source

    two_method_plots = [cell for cell in extension_cells if cell["metadata"]["kind"] == "two_method_plot"]
    for cell in two_method_plots:
        source = _source(cell)
        assert cell["metadata"]["selected_methods"] == ["GLMBSA-CV", "WERM-L1-fullCV-2fold"]
        assert source.count("'name': 'GLMBSA-CV'") == 1
        assert source.count("'name': 'WERM-L1-fullCV-2fold'") == 1
        assert "GLMBSA-CV-5fold" not in source
        assert "WERM-L1-fullCV-10fold" not in source
