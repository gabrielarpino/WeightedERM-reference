import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "M2_upsample_final.ipynb"


def _load():
    return json.loads(NOTEBOOK.read_text())


def test_notebook_structure_is_self_contained():
    nb = _load()
    assert len(nb["cells"]) == 9
    assert [c["cell_type"] for c in nb["cells"]] == [
        "markdown", "markdown", "code", "markdown", "code", "markdown", "code",
        "markdown", "code"
    ]
    assert nb["metadata"]["m2_final"] is True


def test_exact_method_scope_and_fold_labels():
    nb = _load()
    exp = "".join(nb["cells"][2]["source"])
    expected = ["McScan", "charcoal", "WERM-unp-10fold"]
    assert f"FINAL_METHOD_NAMES = {expected!r}" in exp
    assert "methods_to_evaluate = [method_catalog[name].copy() for name in FINAL_METHOD_NAMES]" in exp
    assert "WERM_M2_METHODS" in exp
    assert "requested_method_names" in exp
    assert "num_cv_folds = int(method_name.rsplit('-', 1)[1].replace('fold', ''))" in exp
    plot = "".join(nb["cells"][4]["source"])
    assert r"\textbf{WERM unpenalized (5-fold CV)}" not in plot
    assert r"\textbf{WERM}" in plot
    assert "WERM unpenalized" not in plot


def test_experiment_contract_and_unique_storage():
    src = "".join(_load()["cells"][2]["source"])
    assert "num_trials = 100" in src
    assert "WERM_N_JOBS', '36'" in src
    assert "M_scorer = lambda u, v: jnp.abs(u - v)" in src
    assert "CHARCOAL_MAX_DELTA = 9.0" in src
    assert "delta <= CHARCOAL_MAX_DELTA" in src
    assert "delta_list = np.array([1.5, 10.5])" in src
    assert "charcoal_thresholds" not in src
    assert "_get_charcoal_threshold" not in src
    assert "penalty_norm='none'" in src
    assert "coefficient_loss='least_squares'" in src
    assert "M2_upsample_final_alpha03_seed42_100trials" in src
    assert "M2_upsample_final_alpha03_10fold" in src
    assert '"charcoal_threshold": WERM.comparison.CHARCOAL_THRESHOLD' in src


def test_original_m2_colors_are_preserved_with_distinct_werm_variants():
    src = "".join(_load()["cells"][4]["source"])
    assert "'McScan': dict(label=r'\\textbf{McScan}', color='#AA3377'" in src
    assert "'charcoal': dict(label=r'\\textbf{charcoal}', color='#336677'" in src
    assert "'McScan': dict(label=r'\\textbf{McScan}', color='#AA3377', marker='o'" in src
    assert "'charcoal': dict(label=r'\\textbf{charcoal}', color='#336677', marker='v'" in src
    assert "WERM-unp-5fold" not in src
    assert "'WERM-unp-10fold': dict(label=r'\\textbf{WERM}', color='black'" in src
    assert "marker='D', linestyle='-'" in src
    assert len({'o', 'v', 'D'}) == 3
    assert "WERM unpenalized" not in src
    assert src.count("ls=plot_linestyles[i]") == 3


def test_complete_plot_suite_is_executed_and_embedded():
    cell = _load()["cells"][4]
    src = "".join(cell["source"])
    assert "plot_with_asymmetric_errors_publication()" in src
    assert "plot_with_confidence_bands_publication()" in src
    assert "plot_location_boxplots(" in src
    assert "plot_size_boxplots(" in src
    assert "plot_runtime_boxplots(" in src
    assert "plot_changepoint_histograms_publication(" in src
    assert "_jittered_x" in src
    assert "showfliers=False" in src
    assert len([o for o in cell.get("outputs", []) if o.get("output_type") == "display_data"]) == 6
    assert not [o for o in cell.get("outputs", []) if o.get("output_type") == "error"]


def test_appended_plot_cell_matches_last_m2_comparison_format():
    cell = _load()["cells"][6]
    src = "".join(cell["source"])
    assert cell["metadata"]["kind"] == "canonical_comparison_plot"
    assert "# 5. Publication-Ready Enhanced Plotting Results" in src
    assert "skip_factor = 2" in src
    assert "ax.set_xticks(delta_effective[::2])" in src
    assert "ax.set_xticks(delta_positions[::2])" in src
    assert "selected_deltas_idx = list(range(len(δ_list)))[4:9:2]" in src
    assert "selected_deltas = δ_list[4:9:2]" in src
    assert "np.array([2.5, 5.5, 8.5])" not in src
    assert "ax.hist(all_chgpts, bins=12, alpha=0.8" in src
    assert "GridSearchCV(KernelDensity(kernel='gaussian'), params, cv=5)" in src
    assert "M2_upsample_final_alpha03_seed42_100trials.pkl" in src
    assert "WERM-unp-10fold" in src
    assert "'label': r'\\textbf{WERM}'" in src
    assert "WERM unpenalized" not in src
    assert "'color': 'black'" in src
    assert "'linestyle': '-'" in src
    assert "'marker': 'D'" in src
    assert src.count("ls=methods_to_evaluate[i]['linestyle']") == 3
    assert src.count("marker=methods_to_evaluate[i]['marker']") == 3
    assert "WERM-unp'," not in src
    assert len([o for o in cell.get("outputs", []) if o.get("output_type") == "display_data"]) == 6
    assert not [o for o in cell.get("outputs", []) if o.get("output_type") == "error"]
