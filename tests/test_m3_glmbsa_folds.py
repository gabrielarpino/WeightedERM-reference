import nbformat

from scripts.update_m3_glmbsa_folds import update_notebook


GLMBSA_METHODS = (
    "GLMBSA-CV",
    "GLMBSA-CV-5fold",
    "GLMBSA-CV-10fold",
)


def test_all_m3_experiments_include_glmbsa_2_5_10_and_reuse_schema11():
    for path in ("M3_upsample_redo2.ipynb", "M3_upsample_redo_delta6.ipynb"):
        notebook = update_notebook(nbformat.read(path, as_version=4))
        experiments = [
            cell
            for cell in notebook.cells
            if cell.cell_type == "code" and "def run_single_trial(" in cell.source
        ]
        assert experiments
        for cell in experiments:
            source = cell.source
            assert "glmbsa_kmax = L - 2  # original M3d protocol; independent of ground truth" in source
            assert "kmax=glmbsa_kmax" in source
            assert "kmax=len(fractional_chgpt_locations)" not in source
            for method in GLMBSA_METHODS:
                assert f"'name': '{method}'" in source
            assert "glmbsa_cv_folds = {'GLMBSA-CV': 2, 'GLMBSA-CV-5fold': 5, 'GLMBSA-CV-10fold': 10}" in source
            assert "num_folds=glmbsa_cv_folds[method_name]" in source
            assert "'schema_version': 12" in source
            assert "'glmbsa_cv_folds': (2, 5, 10)" in source
            assert "prior_checkpoint_signature['schema_version'] = 11" in source
            assert "rerun_methods = ('GLMBSA-CV-5fold', 'GLMBSA-CV-10fold')" in source
            assert "glmbsa2-5-10_thirteen_methods" in source


def test_all_m3_plot_cells_use_new_artifact_and_keep_full_plot_suite():
    for path in ("M3_upsample_redo2.ipynb", "M3_upsample_redo_delta6.ipynb"):
        notebook = update_notebook(nbformat.read(path, as_version=4))
        plots = [
            cell
            for cell in notebook.cells
            if cell.cell_type == "code" and "filename_to_load = 'results/M3d_redo" in cell.source
        ]
        assert plots
        for cell in plots:
            assert "glmbsa2-5-10_thirteen_methods" in cell.source
            for method in GLMBSA_METHODS:
                assert f"'name': '{method}'" in cell.source
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


def test_glmbsa_fold_update_is_idempotent():
    notebook = nbformat.read("M3_upsample_redo2.ipynb", as_version=4)
    once = update_notebook(notebook)
    twice = update_notebook(once)
    assert [cell.source for cell in once.cells] == [cell.source for cell in twice.cells]
