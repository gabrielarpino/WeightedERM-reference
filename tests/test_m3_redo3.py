from pathlib import Path

import nbformat


NOTEBOOK = Path("M3_upsample_redo3.ipynb")

AFFECTED_METHODS = (
    "GLMBSA-CV",
    "GLMBSA-CV-5fold",
    "GLMBSA-CV-10fold",
    "WERM-L1-fullCV",
    "WERM-L1-fullCV-2fold",
    "WERM-L1-fullCV-10fold",
)


def _notebook():
    return nbformat.read(NOTEBOOK, as_version=4)


def test_redo3_contains_four_serial_self_contained_experiments_and_plots():
    notebook = _notebook()
    experiments = [c for c in notebook.cells if c.cell_type == "code" and "def run_single_trial(" in c.source]
    plots = [c for c in notebook.cells if c.cell_type == "code" and "filename_to_load = 'results/M3d_redo" in c.source]
    assert len(experiments) == len(plots) == 4
    assert len({cell.id for cell in notebook.cells}) == len(notebook.cells)
    assert "delta=3, alpha=0.5" in experiments[0].source
    assert "signal_alpha = 0.3" in experiments[1].source
    assert "signal_alpha = 0.1" in experiments[2].source
    assert "fixed_delta = 6.0" in experiments[3].source
    assert not any("subprocess" in c.source or "run_m3" in c.source for c in notebook.cells if c.cell_type == "code")
    assert all(not c.outputs and c.execution_count is None for c in notebook.cells if c.cell_type == "code")


def test_redo3_uses_requested_grids_and_truth_independent_glmbsa_budget():
    for cell in (c for c in _notebook().cells if c.cell_type == "code" and "def run_single_trial(" in c.source):
        source = cell.source
        assert "glmbsa_c1_candidates = (0.018, 0.18, 1.8)" in source
        assert "joint_l1_constants = (0.03, 0.3, 3.0)" in source
        assert "glmbsa_kmax = L - 2" in source
        assert "kmax=len(fractional_chgpt_locations)" not in source
        assert "'schema_version': 13" in source
        assert "widegrid_c0018-018-18_C003-03-3" in source


def test_redo3_reuses_only_methods_unchanged_from_schema12():
    for cell in (c for c in _notebook().cells if c.cell_type == "code" and "def run_single_trial(" in c.source):
        source = cell.source
        assert f"rerun_methods = {AFFECTED_METHODS!r}" in source
        assert "prior_checkpoint_signature['schema_version'] = 12" in source
        assert "prior_checkpoint_signature['joint_l1_constants'] = (0.15, 0.3, 0.6)" in source
        assert "prior_checkpoint_signature['glmbsa_c1_candidates'] = (0.09, 0.18, 0.36)" in source
        assert "prior_checkpoint_dir = checkpoint_dir.rsplit('_schema', 1)[0] + '_schema12'" in source
        assert "schema-12 checkpoints" in source


def test_redo3_keeps_100_trials_36_jobs_logistic_loss_and_full_plots():
    notebook = _notebook()
    for cell in (c for c in notebook.cells if c.cell_type == "code" and "def run_single_trial(" in c.source):
        assert "num_trials = 100" in cell.source
        assert "WERM_N_JOBS', '36'" in cell.source
        assert "M = lambda u, v: jnp.logaddexp(0.0, u)" in cell.source
    for cell in (c for c in notebook.cells if c.cell_type == "code" and "filename_to_load = 'results/M3d_redo" in c.source):
        for call in (
            "plot_location_asymmetric_errors(",
            "plot_size_asymmetric_errors(",
            "plot_runtime_asymmetric_errors(",
            "plot_location_boxplots(",
            "plot_size_boxplots(",
            "plot_runtime_boxplots(",
            "plot_changepoint_histograms_publication(",
        ):
            assert call in cell.source
