import nbformat

from scripts.build_m3_alpha_extensions import append_alpha_extensions


METHODS = (
    "GLMBSA-CV",
    "GLMBSA-fixed0.18-2fold",
    "WERM-L1-fullCV",
    "WERM-L1-C0.3-2fold",
    "WERM-unp-2fold",
    "WERM-unp-10fold",
    "WERM-L1-C0.18-2fold",
    "WERM-L1-C0.18-10fold",
    "WERM-L1-C0.3-10fold",
    "WERM-L1-fullCV-2fold",
    "WERM-L1-fullCV-10fold",
)


def test_alpha_extensions_are_self_contained_idempotent_and_plot_complete():
    source = nbformat.read("M3_upsample_redo2.ipynb", as_version=4)
    extended = append_alpha_extensions(source)
    repeated = append_alpha_extensions(extended)

    assert len(extended.cells) == len(source.cells)
    assert len(repeated.cells) == len(extended.cells)

    for alpha, tag in ((0.3, "03"), (0.1, "01")):
        experiment = next(
            cell for cell in extended.cells
            if cell.metadata.get("m3_alpha_experiment") == alpha
        )
        plot = next(
            cell for cell in extended.cells
            if cell.metadata.get("m3_alpha_plot") == alpha
        )
        assert f"signal_alpha = {alpha}" in experiment.source
        assert "α = signal_alpha" in experiment.source
        assert "fixed_delta = 3.0" in experiment.source
        assert "num_trials = 100" in experiment.source
        assert "WERM_N_JOBS', '36'" in experiment.source
        assert "'alpha': float(signal_alpha)" in experiment.source
        glmbsa_cv_branch = experiment.source.split(
            "elif method_name in glmbsa_cv_folds:", 1
        )[1].split("elif method_name == 'GLMBSA-fixed0.18-2fold':", 1)[0]
        assert "num_folds=glmbsa_cv_folds[method_name]" in glmbsa_cv_branch
        assert "'GLMBSA-CV': 2" in experiment.source
        assert "'GLMBSA-CV-5fold': 5" in experiment.source
        assert "'GLMBSA-CV-10fold': 10" in experiment.source
        assert "GLMBSA-CV (2-fold)" in experiment.source
        assert "delta{fixed_delta:g}_alpha{signal_alpha:g}_" in experiment.source
        assert "'schema_version': 12" in experiment.source
        assert f"M3d_redo_delta3_alpha{tag}_glmbsa2-5-10_thirteen_methods" in experiment.source
        assert "prior_checkpoint_signature" in experiment.source
        assert "rerun_methods = ('GLMBSA-CV-5fold', 'GLMBSA-CV-10fold')" in experiment.source
        assert "subprocess" not in experiment.source
        for method in METHODS:
            assert method in experiment.source

        assert f"M3d_redo_delta3_alpha{tag}_glmbsa2-5-10_thirteen_methods" in plot.source
        for call in (
            "plot_location_asymmetric_errors(",
            "plot_size_asymmetric_errors(",
            "plot_runtime_asymmetric_errors(",
            "plot_location_boxplots(",
            "plot_size_boxplots(",
            "plot_runtime_boxplots(",
            "plot_changepoint_histograms_publication(",
        ):
            assert call in plot.source
