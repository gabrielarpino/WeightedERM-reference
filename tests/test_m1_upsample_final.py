import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "M1_upsample_final.ipynb"


def _load():
    return json.loads(NOTEBOOK.read_text())


def test_final_notebook_has_three_self_contained_experiments_and_plots():
    nb = _load()
    assert len(nb["cells"]) == 16
    assert [cell["cell_type"] for cell in nb["cells"]] == [
        "markdown", "markdown", "code", "code", "code", "markdown", "code", "code",
        "markdown", "code", "markdown", "code", "code", "markdown", "code", "code",
    ]
    assert nb["metadata"]["m1_final"] is True


def test_dense_gaussian_experiment_is_separate_self_contained_and_paired():
    nb = _load()
    heading = "".join(nb["cells"][13]["source"])
    experiment = nb["cells"][14]
    plot = nb["cells"][15]
    src = "".join(experiment["source"])
    plot_src = "".join(plot["source"])
    assert "Gaussian noise" in heading
    assert experiment["metadata"]["kind"] == "dense_gaussian_experiment"
    assert plot["metadata"]["kind"] == "dense_gaussian_plot"
    assert '_trial_rng("M1b-alpha-1.0-gaussian", i, j)' in src
    assert "os.environ['R_HOME'] = _active_r_home" in src
    assert src.index("os.environ['R_HOME'] = _active_r_home") < src.index("import WERM.comparison")
    assert "epsilon = rng.normal(0.0, fixed_noise_scale, size=n)" in src
    assert "standard_t" not in src
    assert "num_trials = 100" in src
    assert "WERM_N_JOBS', '36'" in src
    assert "FINAL_METHOD_NAMES = ['McScan', 'WERM-unp-10fold']" in src
    assert "M1_upsample_final_alpha10_dense_gaussian_ls_seed46_100trials.pkl" in src
    assert "M1_upsample_final_alpha10_dense_gaussian_ls_seed46_100trials.pkl" in plot_src
    assert "M = lambda u, v: jnp.power(u - v, 2)" in src
    assert "coefficient_loss='least_squares'" in src
    assert "coefficient_loss='huber'" not in src
    assert 'coefficient_loss="huber"' not in src
    assert 'def M(u, v, epsilon' not in src
    assert r"label=r'\textbf{WERM}'" in plot_src


def test_only_werm_family_labels_are_bold_in_all_final_plots():
    nb = _load()
    plot_kinds = {
        "plot", "focused_plot", "dense_m3_style_plot",
        "canonical_tighter_layout_plot", "m1a_plot", "dense_gaussian_plot",
    }
    plot_sources = [
        "".join(cell["source"])
        for cell in nb["cells"]
        if cell.get("metadata", {}).get("kind") in plot_kinds
    ]
    assert plot_sources
    for src in plot_sources:
        for competitor in ("McScan", "MOSEG", "DCDP", "DPDU"):
            assert rf"\textbf{{{competitor}}}" not in src


def test_m1a_corrected_experiment_is_self_contained_reproducible_and_fair():
    nb = _load()
    heading = "".join(nb["cells"][10]["source"])
    src = "".join(nb["cells"][11]["source"])
    assert "M1a" in heading
    assert "p = 200" in src
    assert "L = 3" in src
    assert "alpha = 0.5" in src
    assert "noise_std = np.sqrt(0.6)" in src
    assert "delta_list = np.arange(1.0, 7.0, 1.0)" in src
    assert "fractional_chgpt_locations = np.array([0.4, 0.7])" in src
    assert "num_trials = 100" in src
    assert "WERM_N_JOBS', '36'" in src
    assert "os.environ['R_HOME'] = _active_r_home" in src
    assert src.index("os.environ['R_HOME'] = _active_r_home") < src.index("import WERM.comparison")
    assert "_trial_rng(\"M1a-final\", i, j)" in src
    assert "FINAL_METHOD_NAMES = ['McScan', 'MOSEG', 'WERM-unp']" in src
    assert "run_mcscan(np.array(X_scaled), y_standardized, num_chgpts=L - 1)" in src
    assert "MOSEG(np.array(X_scaled), y_standardized, num_chgpts=L - 1, ms=False)" in src
    assert "cp = res[2]" in src
    assert "ms=True, num_chgpts = None" not in src
    assert "num_chgpts=len(fractional_chgpt_locations)" not in src
    werm_block = src.split("elif method_name == 'WERM-unp':", 1)[1].split(
        "elif method_name == 'WERM-fixed':", 1
    )[0]
    assert "X=X, y=y, M=M, L=L, Delta_n=1, gamma=0.0" in werm_block
    assert "L_is_upper_bound=False, return_B_hat=False" in werm_block
    assert "brute_force=False" in werm_block
    assert "_weighted_erm_joint_cv" not in werm_block
    assert "num_folds" not in werm_block
    assert "method_name.startswith('WERM-unp-')" not in src
    assert "M_scorer = lambda u, v: jnp.abs(u - v)" in src
    assert "coefficient_loss='least_squares'" in src
    assert "M1_upsample_final_M1a_native_fixedcount_seed42_100trials.pkl" in src


def test_m1a_plot_matches_original_suite_and_final_styles():
    cell = _load()["cells"][12]
    src = "".join(cell["source"])
    assert cell["metadata"]["kind"] == "m1a_plot"
    assert "M1_upsample_final_M1a_native_fixedcount_seed42_100trials.pkl" in src
    assert "plot_with_asymmetric_errors_publication" in src
    assert "plot_with_confidence_bands_publication" in src
    assert "plot_with_boxplots_publication()" in src
    assert "plot_changepoint_histograms_publication(" in src
    reference = json.loads((ROOT / 'M1 Comparison Experiments.ipynb').read_text())
    import ast
    original = ''.join(reference['cells'][5]['source'])
    for node in ast.parse(original).body:
        if isinstance(node, ast.FunctionDef):
            assert f'def {node.name}(' in src
    assert "'axes.labelsize': 18" in src
    assert "δ_list >= 2.0" in src
    assert "_index_map[int(i)]" in src
    assert "'color': '#AA3377'" in src
    assert "'color': '#228833'" in src
    assert "'color': 'black'" in src
    assert "method_obj['color'] = PAL[i" not in src
    assert not [
        output for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]


def test_alpha03_method_scope_and_solver_settings():
    src = "".join(_load()["cells"][2]["source"])
    expected = [
        "McScan", "WERM-L1-jointCV", "WERM-unp-10fold",
        "DCDP", "DPDU", "MOSEG",
    ]
    assert f"FINAL_METHOD_NAMES = {expected!r}" in src
    assert "methods_to_evaluate = [method_catalog[name].copy() for name in FINAL_METHOD_NAMES]" in src
    assert "l1_constants=(0.03, 0.3, 3.0), num_folds=2" in src
    assert "num_cv_folds = int(method_name.rsplit('-', 1)[1].replace('fold', ''))" in src
    assert "M_scorer = lambda u, v: jnp.abs(u - v)" in src
    assert "num_trials = 100" in src
    assert "WERM_N_JOBS', '36'" in src
    assert "M1_upsample_final_alpha03_seed46_100trials" in src


def test_dense_method_scope_and_settings():
    src = "".join(_load()["cells"][6]["source"])
    expected = ["McScan", "WERM-unp-10fold"]
    assert f"FINAL_METHOD_NAMES = {expected!r}" in src
    assert "methods_to_evaluate = [method_catalog[name].copy() for name in FINAL_METHOD_NAMES]" in src
    assert "num_trials = 100" in src
    assert "WERM_N_JOBS', '36'" in src
    assert "M1_upsample_final_alpha10_dense_seed46_100trials" in src


def test_plot_cells_load_unique_artifacts_and_render_complete_suite():
    nb = _load()
    cell = nb["cells"][3]
    src = "".join(cell["source"])
    assert "M1_upsample_final_alpha03_seed46_100trials" in src
    assert "plot_with_asymmetric_errors_publication()" in src
    assert "plot_with_confidence_bands_publication()" in src
    assert "plot_location_boxplots(" in src
    assert "plot_size_boxplots(" in src
    assert "plot_runtime_boxplots(" in src
    assert "plot_changepoint_histograms_publication(" in src
    assert "_jittered_x" in src
    assert "showfliers=False" in src
    assert len([o for o in cell.get("outputs", []) if o.get("output_type") == "display_data"]) == 6


def test_dense_plot_uses_m3_final_panel_structure_and_m1_styles():
    cell = _load()["cells"][7]
    src = "".join(cell["source"])
    assert cell["metadata"]["kind"] == "dense_m3_style_plot"
    assert "M1_upsample_final_alpha10_dense_seed46_100trials" in src
    assert "plot_location_asymmetric_errors(" in src
    assert "plot_location_boxplots(" in src
    assert "plot_size_asymmetric_errors(" in src
    assert "plot_size_boxplots(" in src
    assert "plot_runtime_asymmetric_errors(" in src
    assert "plot_runtime_boxplots(" in src
    assert "plot_with_asymmetric_errors_publication()" not in src
    assert "plot_with_confidence_bands_publication()" not in src
    assert "_jittered_x" in src
    assert "showfliers=False" in src
    assert r"label=r'\textbf{WERM}'" in src
    assert "color='#AA3377'" in src
    assert "color='#E69F00'" not in src
    assert "x_y_label_fontsize = 18" in src
    assert src.count("r'$\\delta = n/p$'") >= 2
    location_box = src[src.index("def plot_location_boxplots("):src.index("def plot_size_boxplots(")]
    assert "tick_stride=2" in location_box
    assert len([o for o in cell.get("outputs", []) if o.get("output_type") == "display_data"]) == 6


def test_alpha03_three_method_focus_cell_is_executed_and_complete():
    cell = _load()["cells"][4]
    src = "".join(cell["source"])
    assert cell["metadata"]["kind"] == "focused_plot"
    assert "FOCUS_METHOD_NAMES = ['WERM-L1-jointCV', 'DCDP', 'DPDU']" in src
    assert "if method[\"name\"] in FOCUS_METHOD_NAMES" in src
    assert "plot_with_asymmetric_errors_publication()" in src
    assert "plot_with_confidence_bands_publication()" in src
    assert "plot_location_boxplots(" in src
    assert "plot_size_boxplots(" in src
    assert "plot_runtime_boxplots(" in src
    assert "plot_changepoint_histograms_publication(" in src
    assert len([o for o in cell.get("outputs", []) if o.get("output_type") == "display_data"]) == 6


def test_requested_final_colors_and_labels():
    nb = _load()
    sparse = "".join(nb["cells"][3]["source"])
    dense = "".join(nb["cells"][7]["source"])
    assert r"'WERM-L1-jointCV': dict(label=r'\textbf{WERM+L1}', color='black'" in sparse
    assert "WERM-unp-5fold" not in sparse
    assert r"'WERM-unp-10fold': dict(label=r'\textbf{WERM unpenalized (10-fold CV)}'" in sparse
    assert r"'MOSEG': dict(label=r'MOSEG', color='#228833', marker='s'" in sparse
    assert r"'DCDP': dict(label=r'DCDP', color='#EE6677', marker='X'" in sparse
    assert r"'DPDU': dict(label=r'DPDU', color='#4477AA', marker='P'" in sparse
    assert "linestyle='--'" in sparse
    assert "WERM-unp-5fold" not in dense
    assert r"dict(name='WERM-unp-10fold', label=r'\textbf{WERM}', color='black'" in dense
    for color in ("#AA3377", "#EE6677", "#4477AA", "#228833"):
        assert color in sparse


def test_appended_plot_cell_matches_tighter_layout_reference_format():
    cell = _load()["cells"][9]
    src = "".join(cell["source"])
    assert cell["metadata"]["kind"] == "canonical_tighter_layout_plot"
    assert "# 5. Publication-Ready Enhanced Plotting Results" in src
    assert "ax.set_xticks(δ_list[::2])  # Show every other tick" in src
    assert "selected_deltas_idx = list(range(len(δ_list)))[::3]" in src
    assert "selected_deltas = δ_list[::3]" in src
    assert "ax.hist(all_chgpts, bins=12, alpha=0.8" in src
    assert "GridSearchCV(KernelDensity(kernel='gaussian'), params, cv=5)" in src
    assert "plt.tight_layout(rect=[0.03, 0.05, 0.96, 0.96])" in src
    assert "M1_upsample_final_alpha03_seed46_100trials.pkl" in src
    assert "WERM-L1-jointCV" in src
    assert "WERM-unp-10fold" not in src
    assert "_delta_keep_indices = np.flatnonzero(np.asarray(δ_list) >= 2)" in src
    assert "δ_list = np.asarray(δ_list)[_delta_keep_indices]" in src
    assert "_delta_index_map = {old_idx: new_idx" in src
    assert "'linestyle': '-'" in src
    assert "'marker': 'D'" in src
    moseg_block = src[src.index("'name': 'MOSEG'"):]
    moseg_block = moseg_block[:moseg_block.index("    },")]
    assert "'marker': 's'" in moseg_block
    expected_markers = {
        "McScan": "o", "MOSEG": "s", "DCDP": "X",
        "DPDU": "P", "WERM-L1-jointCV": "D",
    }
    observed_markers = {}
    for method_name in expected_markers:
        method_block = src[src.index(f"'name': '{method_name}'"):]
        method_block = method_block[:method_block.index("    },")]
        for marker in expected_markers.values():
            if f"'marker': '{marker}'" in method_block:
                observed_markers[method_name] = marker
                break
    assert observed_markers == expected_markers
    assert len(set(observed_markers.values())) == len(observed_markers)
    assert "'linewidth': 2.5" in src
    assert "def _errorbar_with_transparent_uncertainty(" not in src
    assert "uncertainty_alpha" not in src
    assert "Explicit marker-only overlay" not in src
    assert src.count("ls=methods_to_evaluate[i]['linestyle']") == 3
    assert src.count("marker=methods_to_evaluate[i]['marker']") == 3
    assert src.count("linewidth=3.5") >= 3
    assert src.count("capsize=4, capthick=2, markersize=8,") == 3
    assert "M1b_alpha03_cp035065McScan_WERM-unp_DCDP_DPDU_MOSEG" not in src
    assert len([
        output for output in cell.get("outputs", [])
        if output.get("output_type") == "display_data"
    ]) == 7
    assert not [
        output for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
