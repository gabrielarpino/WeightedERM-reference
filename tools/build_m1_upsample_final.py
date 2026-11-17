"""Build the self-contained, publication-facing M1 final notebook."""

from copy import deepcopy
import ast
import json
from pathlib import Path
import pickle
import re
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from tools.final_notebook_presentation import normalize_presentation


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "M1_upsample_redo2.ipynb"
M3_SOURCE = ROOT / "M3_upsample_final.ipynb"
M1_COMPARISON = ROOT / "M1 Comparison Experiments.ipynb"
TARGET = ROOT / "M1_upsample_final.ipynb"
RESULTS = ROOT / "results"

ALPHA03_SOURCE = RESULTS / (
    "M1_upsample_b_alpha03_cp035065_McScan_WERM-unp-2fold_WERM-unp-5fold_"
    "WERM-unp-10fold_WERM-L1-jointCV_WERM-L1-C0.3-CV_DCDP_DPDU_MOSEG_"
    "results_p200_L4_delta123456789_noisecale0.316_seed46.pkl"
)
ALPHA10_SOURCE = RESULTS / (
    "M1_upsample_b_alpha10_cp035065_McScan_WERM-unp-2fold_WERM-unp-5fold_"
    "WERM-unp-10fold_results_p200_L4_delta23456789_noisecale0.316_seed46.pkl"
)
ALPHA03_TARGET = RESULTS / "M1_upsample_final_alpha03_seed46_100trials.pkl"
ALPHA10_TARGET = RESULTS / "M1_upsample_final_alpha10_dense_seed46_100trials.pkl"
ALPHA10_GAUSSIAN_TARGET = RESULTS / "M1_upsample_final_alpha10_dense_gaussian_ls_seed46_100trials.pkl"
M1A_TARGET = RESULTS / "M1_upsample_final_M1a_native_fixedcount_seed42_100trials.pkl"

SPARSE_METHODS = [
    "McScan", "WERM-L1-jointCV", "WERM-unp-10fold", "DCDP", "DPDU", "MOSEG"
]
DENSE_METHODS = ["McScan", "WERM-unp-10fold"]
M1A_METHODS = ["McScan", "MOSEG", "WERM-unp"]

SPARSE_CATALOG = {
    "McScan": dict(name="McScan", label=r"McScan", color="#AA3377", marker="o", linestyle="--"),
    "WERM-L1-jointCV": dict(name="WERM-L1-jointCV", label=r"\textbf{WERM+L1}", color="black", marker="D", linestyle="-"),
    "WERM-unp-10fold": dict(name="WERM-unp-10fold", label=r"\textbf{WERM unpenalized (10-fold CV)}", color="#333333", marker="D", linestyle=":"),
    "DCDP": dict(name="DCDP", label=r"DCDP", color="#EE6677", marker="X", linestyle="--"),
    "DPDU": dict(name="DPDU", label=r"DPDU", color="#4477AA", marker="P", linestyle=":"),
    "MOSEG": dict(name="MOSEG", label=r"MOSEG", color="#228833", marker="s", linestyle="-."),
}
DENSE_CATALOG = {
    "McScan": deepcopy(SPARSE_CATALOG["McScan"]),
    "WERM-unp-10fold": dict(name="WERM-unp-10fold", label=r"\textbf{WERM}", color="black", marker="D", linestyle="-"),
}
M1A_CATALOG = {
    "McScan": dict(name="McScan", label=r"McScan", color="#AA3377", marker="o", linestyle="--"),
    "MOSEG": dict(name="MOSEG", label=r"MOSEG", color="#228833", marker="s", linestyle="-."),
    "WERM-unp": dict(name="WERM-unp", label=r"\textbf{WERM}", color="black", marker="D", linestyle="-"),
}


def subset_payload(source, target, names, catalog, folds):
    with source.open("rb") as handle:
        payload = pickle.load(handle)
    old = payload["results"]
    keep = {"true_chgpts", "n_values", "delta_values"}
    for name in names:
        keep.update({f"haus_{name}", f"size_{name}", f"runtime_{name}", f"chgpts_{name}"})
    new_results = {key: value for key, value in old.items() if key in keep}
    new_results["successful_trials"] = {
        name: old["successful_trials"][name] for name in names
    }
    params = dict(payload["params"])
    params["methods_evaluated"] = [deepcopy(catalog[name]) for name in names]
    params["successful_trials_counts"] = new_results["successful_trials"]
    params["num_folds_unpenalized"] = folds
    params["source_result_file"] = str(source.relative_to(ROOT))
    with target.open("wb") as handle:
        pickle.dump({"results": new_results, "params": params}, handle)


def replace_method_block(source, names, catalog):
    rendered = json.dumps(catalog, indent=4).replace("true", "True").replace("false", "False")
    # JSON strings are valid Python here; keep a readable explicit catalog in the cell.
    block = (
        f"FINAL_METHOD_NAMES = {names!r}\n"
        f"method_catalog = {rendered}\n"
        "methods_to_evaluate = [method_catalog[name].copy() for name in FINAL_METHOD_NAMES]\n\n"
    )
    start = source.index("methods_to_evaluate = [")
    end = source.index("# Create a dictionary for easy lookup within run_single_trial", start)
    return source[:start] + block + source[end:]


def experiment_source(cell, names, catalog, artifact):
    source = "".join(cell["source"])
    source = replace_method_block(source, names, catalog)
    save = f'''\n\n# Save this final experiment under an isolated, stable name.\nFINAL_RESULT_FILE = "results/{artifact.name}"\nif SMOKE_MODE:\n    FINAL_RESULT_FILE = FINAL_RESULT_FILE.replace(".pkl", "_smoke.pkl")\nwith open(FINAL_RESULT_FILE, "wb") as handle:\n    pickle.dump({{"results": results, "params": {{\n        "p": p, "L": L, "delta_list": delta_list,\n        "fixed_noise_scale": fixed_noise_scale, "num_trials": num_trials,\n        "fractional_chgpt_locations": fractional_chgpt_locations,\n        "n_jobs": n_jobs, "successful_trials_counts": results["successful_trials"],\n        "methods_evaluated": methods_to_evaluate, "random_seed": random_seed,\n    }}}}, handle)\nprint(f"Saved final results to {{FINAL_RESULT_FILE}}")\n'''
    return source + save


def _replace_once(source, old, new, description):
    if source.count(old) != 1:
        raise RuntimeError(
            f"Expected one {description} occurrence, found {source.count(old)}"
        )
    return source.replace(old, new, 1)


def m1a_experiment_source(cell):
    """Adapt the corrected M1 runner to the original M1a data-generating process."""
    source = experiment_source(cell, M1A_METHODS, M1A_CATALOG, M1A_TARGET)
    source = '''import os
import sys

# rpy2 must bind to the R installation paired with the active Python environment.
_active_env_bin = os.path.dirname(sys.executable)
os.environ['PATH'] = _active_env_bin + os.pathsep + os.environ.get('PATH', '')
_active_r_home = os.path.join(sys.prefix, 'lib', 'R')
if os.path.isdir(_active_r_home):
    os.environ['R_HOME'] = _active_r_home

''' + source
    replacements = [
        ("random_seed = 46", "random_seed = 42", "random seed"),
        ('_trial_rng("M1b-alpha-0.3", i, j)', '_trial_rng("M1a-final", i, j)', "trial RNG namespace"),
        ("def run_single_trial(params, p, L, fixed_noise_scale,", "def run_single_trial(params, p, L, noise_std,", "trial noise argument"),
        ("    κ = 1.0\n    α = 0.3 # Fraction of nonzeros\n    σ_l_arr = κ * np.sqrt(delta) * np.ones(L)  # Signal standard deviations\n    signal_prior = SparseGaussianSignal(α, delta, σ_l_arr)",
         "    kappa = 1.0\n    alpha = 0.5  # Original M1a fraction of nonzeros\n    sigma_l_arr = kappa * np.sqrt(delta) * np.ones(L)\n    signal_prior = SparseGaussianSignal(alpha, delta, sigma_l_arr)", "M1a signal prior"),
        ("    ε = rng.standard_t(4, size=(n, )) * fixed_noise_scale # For Huber loss, use t-distribution for noise. df > 2 has finite variance.",
         "    ε = rng.normal(0.0, noise_std, size=n)", "Gaussian noise"),
        ("    y_standardized = np.array(y / np.std(y))  # Standardize y",
         "    y_standardized = np.asarray(y / (alpha * kappa**2 + noise_std**2))", "original response scaling"),
        ("p = 200\nL = 4\nfixed_noise_scale = np.sqrt(0.1) # Noise scale is now fixed\ndelta_list = np.arange(1.0, 10.0, 1.0) # Iterate over delta\nnum_trials = 100\nfractional_chgpt_locations = np.array([0.35, 0.65])",
         "p = 200\nL = 3\nnoise_std = np.sqrt(0.6)\ndelta_list = np.arange(1.0, 7.0, 1.0)\nnum_trials = 100\nfractional_chgpt_locations = np.array([0.4, 0.7])", "M1a parameters"),
        ("r = 0.4 # Charcoal requires i.i.d covariates", "r = 0.2  # Original M1a AR(1) feature correlation", "feature correlation"),
        ("            fixed_noise_scale=fixed_noise_scale,", "            noise_std=noise_std,", "parallel noise argument"),
        ('"fixed_noise_scale": fixed_noise_scale, "num_trials": num_trials,', '"noise_std": noise_std, "num_trials": num_trials,', "saved noise parameter"),
    ]
    for old, new, description in replacements:
        source = _replace_once(source, old, new, description)

    source = source.replace("coefficient_loss='huber', huber_delta=1.35,", "coefficient_loss='least_squares',")
    source = source.replace('coefficient_loss="huber", coefficient_lambda=l1_reg_param,\n                        huber_delta=1.35,', 'coefficient_loss="least_squares", coefficient_lambda=l1_reg_param,')
    source = source.replace('coefficient_loss="huber", coefficient_lambda=best_l1_reg,\n                    huber_delta=1.35,', 'coefficient_loss="least_squares", coefficient_lambda=best_l1_reg,')

    # M1a is the fixed-L experiment.  Preserve its original direct,
    # unpenalized WERM call exactly: there is no changepoint-count CV here.
    werm_start = source.index("            elif method_name.startswith('WERM-unp-'):")
    werm_end = source.index("            elif method_name == 'WERM-fixed':", werm_start)
    original_m1a_werm = """            elif method_name == 'WERM-unp':
                # Placeholder for WERM-unp method
                # This calls WERM.estimators.weighted_erm with P_func=None (no penalty)
                cp = WERM.estimators.weighted_erm(
                    X=X, y=y, M=M, L=L, Delta_n=1, gamma=0.0, # Delta_n=1 is a placeholder, adjust if needed
                    L_is_upper_bound=False, return_B_hat=False,
                    brute_force=False, # No penalty function
                )
"""
    source = source[:werm_start] + original_m1a_werm + source[werm_end:]

    source = _replace_once(
        source,
        "run_mcscan(np.array(X_scaled), y_standardized.flatten(), num_chgpts = None, method='auto')",
        "run_mcscan(np.array(X_scaled), y_standardized, num_chgpts=L - 1)",
        "M1a original fixed-count McScan call",
    )
    source = _replace_once(
        source,
        "MOSEG(np.array(X_scaled), y_standardized, ms=True, num_chgpts = None)",
        "MOSEG(np.array(X_scaled), y_standardized, num_chgpts=L - 1, ms=False)",
        "M1a original native fixed-count MOSEG call",
    )
    source = _replace_once(
        source,
        "cp = res[1]  # Extract the refined estimate; R NULL means no change found.",
        "cp = res[2]  # Native moseg: refined.cps (original M1a return field).",
        "M1a native refined MOSEG field",
    )

    huber_block = re.compile(
        r"# Loss Function \(Least Squares\)\n# M = lambda u, v: jnp\.power\(u - v, 2\)\n\n"
        r"# Huber Loss\ndef M\(u, v, epsilon=1\.35\):\n(?:    .*\n){6}"
    )
    source, count = huber_block.subn(
        "# Loss Function (Least Squares)\nM = lambda u, v: jnp.power(u - v, 2)\n\n",
        source,
        count=1,
    )
    if count != 1:
        raise RuntimeError("Could not replace the Huber loss block for M1a")
    return source


def m1a_original_plot_source():
    reference_nb = nbformat.read(M1_COMPARISON, as_version=4)
    reference = reference_nb.cells[4].source
    plot_reference = reference_nb.cells[5].source
    # Use the final M1a Load and plot section, not the draft palette.
    tree = ast.parse(plot_reference)
    rc = next(node for node in tree.body if isinstance(node, ast.Expr)
              and isinstance(node.value, ast.Call)
              and ast.unparse(node.value.func) == 'plt.rcParams.update')
    functions = [ast.get_source_segment(plot_reference, node)
                 for node in tree.body if isinstance(node, ast.FunctionDef)]
    original_plots = ast.get_source_segment(plot_reference, rc) + '\n'
    original_plots += 'x_y_label_fontsize = 22\n' + '\n\n'.join(functions)
    original_plots += '''
for plot_type in ('all', 'location', 'size', 'runtime'):
    print('M1a:', plot_type)
    plot_with_asymmetric_errors_publication(plot_type=plot_type)
    plt.show()
plot_with_confidence_bands_publication()
plot_with_boxplots_publication()
plot_changepoint_histograms_publication(results, δ_list, fractional_chgpt_locations, method_names_list)
'''
    assignment = next(
        node for node in ast.parse(reference).body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'methods_to_evaluate' for t in node.targets)
    )
    catalog = [m for m in ast.literal_eval(assignment.value) if m['name'] in M1A_METHODS]
    for m in catalog:
        m['label'] = r'\textbf{WERM}' if m['name'] == 'WERM-unp' else m['name']
    header = f'''import os, pickle, time, logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
plt.switch_backend('module://matplotlib_inline.backend_inline')
main_logger = logging.getLogger('M1a-original-plots')
FINAL_RESULT_FILE = 'results/{M1A_TARGET.name}'
if os.environ.get('WERM_SMOKE') == '1':
    FINAL_RESULT_FILE = FINAL_RESULT_FILE.replace('.pkl', '_smoke.pkl')
with open(FINAL_RESULT_FILE, 'rb') as handle:
    payload = pickle.load(handle)
results, params = payload['results'], payload['params']
p, L = params['p'], params['L']
noise_std, random_seed = params['noise_std'], params['random_seed']
num_trials = params['num_trials']
δ_list = np.asarray(params['delta_list'])
fractional_chgpt_locations = np.asarray(params['fractional_chgpt_locations'])
methods_to_evaluate = {catalog!r}
method_names_list = [m['name'] for m in methods_to_evaluate]
PAL = [m['color'] for m in methods_to_evaluate]
method_colors_map = {{m['name']: m['color'] for m in methods_to_evaluate}}
# Filter both metric rows and indexed histogram records without changing saved data.
_keep = np.flatnonzero(δ_list >= 2.0)
_index_map = {{int(old): new for new, old in enumerate(_keep)}}
results = dict(results)
for _method in method_names_list:
    for _metric in ('haus', 'size', 'runtime'):
        _key = f'{{_metric}}_{{_method}}'
        results[_key] = np.asarray(results[_key])[_keep]
for _key in ['true_chgpts'] + [f'chgpts_{{m}}' for m in method_names_list]:
    results[_key] = [(_index_map[int(i)], cp) for i, cp in results[_key] if int(i) in _index_map]
δ_list = δ_list[_keep]
'''
    return header + '\n' + original_plots


def dense_gaussian_experiment_source(cell):
    """Dense M1b with Gaussian noise and exact unpenalized least squares."""
    source = experiment_source(
        cell, DENSE_METHODS, DENSE_CATALOG, ALPHA10_GAUSSIAN_TARGET
    )
    source = '''import os
import sys

# Bind rpy2 to the R installation paired with the active Python environment.
_active_env_bin = os.path.dirname(sys.executable)
os.environ['PATH'] = _active_env_bin + os.pathsep + os.environ.get('PATH', '')
_active_r_home = os.path.join(sys.prefix, 'lib', 'R')
if os.path.isdir(_active_r_home):
    os.environ['R_HOME'] = _active_r_home

''' + source
    source = _replace_once(
        source,
        '_trial_rng("M1b-alpha-1.0-redo", i, j)',
        '_trial_rng("M1b-alpha-1.0-gaussian", i, j)',
        "dense Gaussian RNG namespace",
    )
    source = _replace_once(
        source,
        "    ε = rng.standard_t(4, size=(n, )) * fixed_noise_scale # For Huber loss, use t-distribution for noise. df > 2 has finite variance.",
        "    epsilon = rng.normal(0.0, fixed_noise_scale, size=n)\n"
        "    ε = epsilon  # Preserve the response-generation notation used below.",
        "dense Gaussian noise draw",
    )
    source = source.replace('coefficient_loss="huber"', 'coefficient_loss="least_squares"')
    source = source.replace("coefficient_loss='huber'", "coefficient_loss='least_squares'")
    source = source.replace('huber_delta=1.35,', '')
    start = source.index('# Loss Function (Least Squares)')
    end = source.index('M_scorer =', start)
    source = source[:start] + (
        '# Least-squares fitting; retain absolute-error CV scoring.\n'
        'M = lambda u, v: jnp.power(u - v, 2)\n\n'
    ) + source[end:]
    return source


def style_block(catalog):
    lines = ["_FINAL_M1B_STYLES = {"]
    for name, spec in catalog.items():
        label = "r'" + spec["label"].replace("'", "\\'") + "'"
        lines.append(
            f"    {name!r}: dict(label={label}, color={spec['color']!r}, "
            f"marker={spec['marker']!r}, linestyle={spec['linestyle']!r}, "
            f"linewidth={2.5 if spec['color'] in ('black', '#555555') else 1.5}, "
            f"zorder={10 if spec['color'] == 'black' else 9 if spec['color'] == '#555555' else 5}),"
        )
    lines.append("}")
    return "\n".join(lines)


def distribution_block():
    """Reuse M3 final's audited box+jitter implementation, adapted to delta."""
    m3 = nbformat.read(M3_SOURCE, as_version=4)
    m3_plot = "".join(m3.cells[3]["source"])
    start = m3_plot.index("# --- Distribution Plotting Functions")
    end = m3_plot.index("# -----------------------------------------------------------------------------\n# Publication-Ready Changepoint Histograms", start)
    block = m3_plot[start:end]
    replacements = [
        ("current_kappa_list", "current_delta_list"),
        ("kappa_positions", "delta_positions"),
        ("kappa_idx", "delta_idx"),
        ("kappa_squared", "delta_value"),
    ]
    for old, new in replacements:
        block = block.replace(old, new)
    block = block.replace("KAPPA_SQUARED_LABEL", r"r'$\delta$'")
    block = re.sub(r"\bkappa\b", "delta", block)
    block = block.replace(r"$\kappa^2$", r"$\delta$")
    block = block.replace(
        '    tick_label_format=".0f",\n):\n    """Shared implementation',
        '    tick_label_format=".0f",\n    tick_stride=1,\n):\n    """Shared implementation',
        1,
    )
    block = block.replace(
        '    ax.set_xticks(delta_positions)\n'
        '    ax.set_xticklabels(\n'
        '        [format(float(k), tick_label_format) for k in current_delta_list]\n'
        '    )',
        '    tick_indices = np.arange(0, len(current_delta_list), tick_stride, dtype=int)\n'
        '    ax.set_xticks([delta_positions[i] for i in tick_indices])\n'
        '    ax.set_xticklabels(\n'
        '        [format(float(current_delta_list[i]), tick_label_format) for i in tick_indices]\n'
        '    )',
        1,
    )
    block = re.sub(r"^\s*fig\.savefig\(os\.path\.join\(FIGURES_DIR,.*$", "", block, flags=re.MULTILINE)
    return "x_y_label_fontsize = 18\n\n" + block


def distribution_calls():
    return r'''print("3. Box plots with deterministic raw-point jitter (M3 final setup)...")
plot_location_boxplots(
    delta_list, results, methods_to_evaluate, "M1_upsample_final",
    plot_style="box_jitter", collapsed_style="line_and_marker",
)
plot_size_boxplots(
    delta_list, results, methods_to_evaluate, L, fractional_chgpt_locations,
    "M1_upsample_final", plot_style="box_jitter",
    collapsed_style="line_and_marker",
)
plot_runtime_boxplots(
    delta_list, results, methods_to_evaluate, "M1_upsample_final",
    plot_style="box_jitter", collapsed_style="line_and_marker",
)'''


def plot_source(cell, artifact, catalog, selected_names=None):
    source = "".join(cell["source"])
    source = re.sub(
        r"_FINAL_M1B_STYLES = \{.*?\n\}", lambda _match: style_block(catalog), source,
        count=1, flags=re.DOTALL,
    )
    box_start = source.index("# --- Option 3: Publication-Ready Box Plots ---")
    box_end = source.index("# --- Publication-Ready Changepoint Histograms ---", box_start)
    source = source[:box_start] + distribution_block() + "\n" + source[box_end:]
    source = source.replace(
        'print("3. Box plots (publication style)...")\nplot_with_boxplots_publication()',
        distribution_calls(),
    )
    save_marker = "# Save results to a file"
    if save_marker in source:
        source = source[:source.index(save_marker)] + "print('Plots complete; experiment artifact was not modified.')\n"
    focus_block = ""
    if selected_names is not None:
        focus_block = (
            f"FOCUS_METHOD_NAMES = {selected_names!r}\n"
            "methods_to_evaluate = [\n"
            "    method for method in methods_to_evaluate\n"
            "    if method[\"name\"] in FOCUS_METHOD_NAMES\n"
            "]\n"
        )
    noise_value = '_params["noise_std"]' if artifact == M1A_TARGET else '_params["fixed_noise_scale"]'
    header = f'''import os\nimport pickle\nimport numpy as np\nimport matplotlib.pyplot as plt\n\nSMOKE_MODE = os.environ.get("WERM_SMOKE") == "1"\nFINAL_RESULT_FILE = "results/{artifact.name}"\nif SMOKE_MODE:\n    FINAL_RESULT_FILE = FINAL_RESULT_FILE.replace(".pkl", "_smoke.pkl")\nwith open(FINAL_RESULT_FILE, "rb") as handle:\n    _payload = pickle.load(handle)\nresults = _payload["results"]\n_params = _payload["params"]\np, L = _params["p"], _params["L"]\ndelta_list = np.asarray(_params["delta_list"])\nfixed_noise_scale = {noise_value}\nnum_trials = _params["num_trials"]\nfractional_chgpt_locations = np.asarray(_params["fractional_chgpt_locations"])\nrandom_seed = _params["random_seed"]\nmethods_to_evaluate = _params["methods_evaluated"]\n{focus_block}method_names_list = [method["name"] for method in methods_to_evaluate]\nprint(f"Loaded {{FINAL_RESULT_FILE}}")\n\n'''
    return header + source


def canonical_tighter_layout_plot_source():
    """Reuse the canonical M1 alpha=0.3 tighter-layout plotting section."""
    reference = nbformat.read(M1_COMPARISON, as_version=4)
    styles = reference.cells[30].source
    plots = reference.cells[31].source
    for competitor in ("McScan", "MOSEG", "DCDP", "DPDU"):
        styles = styles.replace(rf"\textbf{{{competitor}}}", competitor)
        plots = plots.replace(rf"\textbf{{{competitor}}}", competitor)

    original_hero = r"""    {
        'name': 'WERM-unp', 
        'label': r'\textbf{WERM}', 
        'color': 'black', 
        'marker': '*', 
        'linestyle': '-', 
        'linewidth': 2.5, 
        'zorder': 10
    },"""
    final_werm_methods = r"""    {
        'name': 'WERM-L1-jointCV',
        'label': r'\textbf{WERM+L1}',
        'color': 'black',
        'marker': 'D',
        'linestyle': '-',
        'linewidth': 2.5,
        'zorder': 10
    },"""
    if original_hero not in styles:
        raise RuntimeError("Canonical M1 WERM style block was not found")
    styles = styles.replace(original_hero, final_werm_methods, 1)
    _moseg_start = styles.index("'name': 'MOSEG'")
    _moseg_end = styles.index("    },", _moseg_start)
    _moseg_style = styles[_moseg_start:_moseg_end]
    styles = (
        styles[:_moseg_start]
        + _moseg_style.replace("'marker': '^'", "'marker': 's'")
        + styles[_moseg_end:]
    )
    for _method_name, _marker in {"DCDP": "X", "DPDU": "P"}.items():
        _style_start = styles.index(f"'name': '{_method_name}'")
        _style_end = styles.index("    },", _style_start)
        _style = styles[_style_start:_style_end]
        _style = re.sub(r"'marker': '[^']+'", f"'marker': '{_marker}'", _style, count=1)
        styles = styles[:_style_start] + _style + styles[_style_end:]
    plots = plots.replace(
        "results/M1b_alpha03_cp035065McScan_WERM-unp_DCDP_DPDU_MOSEG_"
        "results_p200_L4_delta123456789_noisecale0.316_seed42.pkl",
        "results/M1_upsample_final_alpha03_seed46_100trials.pkl",
    )
    delta_filter = r"""
    # This final presentation intentionally starts at delta=2. Keep the result
    # arrays and the changepoint-record indices aligned with the filtered grid.
    _delta_keep_indices = np.flatnonzero(np.asarray(δ_list) >= 2)
    _delta_index_map = {old_idx: new_idx
                        for new_idx, old_idx in enumerate(_delta_keep_indices)}
    for _key, _value in list(results.items()):
        if _key.startswith(('haus_', 'size_', 'runtime_')):
            results[_key] = np.asarray(_value)[_delta_keep_indices, ...]
        elif _key.startswith('chgpts_'):
            results[_key] = [
                (_delta_index_map[_old_idx], _chgpts)
                for _old_idx, _chgpts in _value
                if _old_idx in _delta_index_map
            ]
    δ_list = np.asarray(δ_list)[_delta_keep_indices]
"""
    load_anchor = "    δ_list = params['delta_list']\n"
    if load_anchor not in plots:
        raise RuntimeError("Canonical M1 delta-list load was not found")
    plots = plots.replace(load_anchor, load_anchor + delta_filter, 1)

    # Keep the current final-notebook histogram cadence (delta 2, 5, 8 after
    # filtering), while retaining the canonical reference implementation.
    plots = plots.replace(
        "selected_deltas_idx = list(range(len(δ_list)))[::4]",
        "selected_deltas_idx = list(range(len(δ_list)))[::3]",
        1,
    ).replace("selected_deltas = δ_list[::4]", "selected_deltas = δ_list[::3]", 1)

    # Preserve the reference notebook's original opaque error bars and markers.
    # Only correct its positional linestyle assignment so the declared WERM+L1
    # solid-black publication style is actually honored in all line panels.
    plots = plots.replace(
        "label=labels[i], marker=markers[i % len(markers)], ls=linestyles[i % len(linestyles)],",
        "label=labels[i], marker=methods_to_evaluate[i]['marker'], "
        "ls=methods_to_evaluate[i]['linestyle'],",
    )
    return styles + "\n\n" + plots


def dense_m3_style_plot_source(
    artifact=ALPHA10_TARGET,
    section_label="Dense M1b (alpha=1.0): M3-final-style panels",
):
    """Five dense-M1 panels using the audited M3-final presentation grammar."""
    dense_distribution = distribution_block().replace(
        "x_y_label_fontsize = 18", "x_y_label_fontsize = 22"
    ).replace(
        "x_label=r'$\\delta$'", "x_label=r'$\\delta = n/p$'"
    ).replace(
        '        show_legend=False,\n        tick_label_format=".0f",',
        '        show_legend=False,\n        tick_label_format=".0f",\n        tick_stride=2,',
        1,
    )
    return rf'''import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend("module://matplotlib_inline.backend_inline")
plt.rcParams.update({{
    'pgf.texsystem': 'pdflatex',
    'pgf.preamble': r'\usepackage{{amsmath}}\usepackage{{amsfonts}}\usepackage{{amssymb}}\usepackage{{siunitx}}',
    'font.size': 14,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif'],
    'text.usetex': True,
    'axes.linewidth': 1.5,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'legend.frameon': True,
    'legend.fancybox': True,
    'legend.edgecolor': 'black',
    'legend.framealpha': 0.95,
    'lines.linewidth': 4.0,
    'lines.markersize': 8,
    'figure.dpi': 72,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
}})

FINAL_RESULT_FILE = "results/{artifact.name}"
if os.environ.get("WERM_SMOKE") == "1":
    FINAL_RESULT_FILE = FINAL_RESULT_FILE.replace(".pkl", "_smoke.pkl")
with open(FINAL_RESULT_FILE, "rb") as handle:
    _payload = pickle.load(handle)
results = _payload["results"]
_params = _payload["params"]
p, L = _params["p"], _params["L"]
delta_list = np.asarray(_params["delta_list"], dtype=float)
fractional_chgpt_locations = np.asarray(_params["fractional_chgpt_locations"])

methods_to_evaluate = [
    dict(name='McScan', label=r'McScan', color='#AA3377', marker='o', linestyle='--'),
    dict(name='WERM-unp-10fold', label=r'\textbf{{WERM}}', color='black', marker='D', linestyle='-'),
]
method_names_list = [method['name'] for method in methods_to_evaluate]
for method_name in method_names_list:
    _size = np.asarray(results[f'size_{{method_name}}'], dtype=float)
    _haus = np.asarray(results[f'haus_{{method_name}}'], dtype=float)
    results[f'haus_{{method_name}}'] = np.where(_size > 0, _haus, np.nan)

def get_asymmetric_errors(data):
    data = np.asarray(data, dtype=float)
    means = np.nanmean(data, axis=1)
    p25 = np.nanpercentile(data, 25, axis=1)
    p75 = np.nanpercentile(data, 75, axis=1)
    return [np.abs(means - p25), np.abs(p75 - means)]

def _plot_asymmetric_metric(metric, ylabel, reference_y=None, y_limits=None):
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for method in methods_to_evaluate:
        values = np.asarray(results[f'{{metric}}_{{method["name"]}}'], dtype=float)
        means = np.nanmean(values, axis=1)
        ax.errorbar(
            delta_list, means, yerr=get_asymmetric_errors(values),
            label=method['label'], marker=method['marker'], ls=method['linestyle'],
            color=method['color'], capsize=4, capthick=2, markersize=8,
            linewidth=3.5, markeredgewidth=2, markeredgecolor='white',
        )
    if reference_y is not None:
        ax.axhline(reference_y, color='gray', linestyle='-', linewidth=2.5,
                   alpha=0.8, label=r'True no. change points')
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.set_xlabel(r'$\delta = n/p$', fontsize=22)
    ax.set_ylabel(ylabel, fontsize=22)
    ax.set_xticks(delta_list[::2])
    ax.set_xticklabels([f'{{value:.0f}}' for value in delta_list[::2]], fontsize=22)
    ax.tick_params(axis='both', which='major', width=1.5, length=6, labelsize=22)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.legend(loc='best', prop={{'size': 17}}, fancybox=True, framealpha=0.4)
    fig.tight_layout()
    plt.show()
    return fig, ax

def plot_size_asymmetric_errors():
    return _plot_asymmetric_metric(
        'size', r'No. change points',
        reference_y=len(fractional_chgpt_locations), y_limits=(0, L),
    )

def plot_location_asymmetric_errors():
    return _plot_asymmetric_metric('haus', r'Hausdorff dist./$n$')

def plot_runtime_asymmetric_errors():
    return _plot_asymmetric_metric('runtime', r'Runtime (seconds)', y_limits=(0, None))

{dense_distribution}

print({section_label!r})
print("1. Location error: asymmetric-error line plot")
plot_location_asymmetric_errors()
print("2. Location error: boxplot + deterministic raw-point jitter")
plot_location_boxplots(
    delta_list, results, methods_to_evaluate, "M1_dense_final",
    plot_style="box_jitter", collapsed_style="line_and_marker",
)
print("3. Number of change points: asymmetric-error line plot")
plot_size_asymmetric_errors()
print("4. Number of change points: boxplot + deterministic raw-point jitter")
plot_size_boxplots(
    delta_list, results, methods_to_evaluate, L, fractional_chgpt_locations,
    "M1_dense_final", plot_style="box_jitter", collapsed_style="line_and_marker",
)
print("5. Runtime: asymmetric-error line plot")
plot_runtime_asymmetric_errors()
print("6. Runtime: boxplot + deterministic raw-point jitter")
plot_runtime_boxplots(
    delta_list, results, methods_to_evaluate, "M1_dense_final",
    plot_style="box_jitter", collapsed_style="line_and_marker",
)
'''


def code_cell(source, metadata):
    cell = nbformat.v4.new_code_cell(source=normalize_presentation(source), metadata=metadata)
    cell["execution_count"] = None
    cell["outputs"] = []
    return cell


def execute_plot(cell):
    mini = nbformat.v4.new_notebook(cells=[deepcopy(cell)])
    mini.metadata.kernelspec = {
        "display_name": "weighted-erm", "language": "python", "name": "python3"
    }
    kernel_manager = KernelManager(kernel_name="python3")
    kernel_manager.kernel_spec.argv[0] = sys.executable
    client = NotebookClient(
        mini, timeout=600, km=kernel_manager,
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
    return mini.cells[0]


def main():
    source_nb = nbformat.read(SOURCE, as_version=4)
    subset_payload(ALPHA03_SOURCE, ALPHA03_TARGET, SPARSE_METHODS, SPARSE_CATALOG, (10,))
    subset_payload(ALPHA10_SOURCE, ALPHA10_TARGET, DENSE_METHODS, DENSE_CATALOG, (10,))

    sparse_exp = code_cell(experiment_source(source_nb.cells[2], SPARSE_METHODS, SPARSE_CATALOG, ALPHA03_TARGET), {"m1_final": True, "alpha": 0.3, "kind": "experiment"})
    sparse_plot = code_cell(plot_source(source_nb.cells[4], ALPHA03_TARGET, SPARSE_CATALOG), {"m1_final": True, "alpha": 0.3, "kind": "plot"})
    dense_exp = code_cell(experiment_source(source_nb.cells[6], DENSE_METHODS, DENSE_CATALOG, ALPHA10_TARGET), {"m1_final": True, "alpha": 1.0, "kind": "experiment"})
    dense_plot = code_cell(
        dense_m3_style_plot_source(),
        {"m1_final": True, "alpha": 1.0, "kind": "dense_m3_style_plot"},
    )
    m1a_exp = code_cell(
        m1a_experiment_source(source_nb.cells[2]),
        {"m1_final": True, "experiment": "M1a", "kind": "experiment"},
    )
    m1a_plot = code_cell(
        m1a_original_plot_source(),
        {"m1_final": True, "experiment": "M1a", "kind": "m1a_plot"},
    )
    dense_gaussian_exp = code_cell(
        dense_gaussian_experiment_source(source_nb.cells[6]),
        {
            "m1_final": True, "alpha": 1.0,
            "experiment": "M1b-dense-Gaussian", "kind": "dense_gaussian_experiment",
        },
    )
    dense_gaussian_plot = code_cell(
        dense_m3_style_plot_source(
            ALPHA10_GAUSSIAN_TARGET,
            "Dense M1b (alpha=1.0, Gaussian noise): M3-final-style panels",
        ),
        {
            "m1_final": True, "alpha": 1.0,
            "experiment": "M1b-dense-Gaussian", "kind": "dense_gaussian_plot",
        },
    )

    focus_plot = code_cell(
        plot_source(
            source_nb.cells[4], ALPHA03_TARGET,
            {name: SPARSE_CATALOG[name] for name in ["WERM-L1-jointCV", "DCDP", "DPDU"]},
            selected_names=["WERM-L1-jointCV", "DCDP", "DPDU"],
        ),
        {"m1_final": True, "alpha": 0.3, "kind": "focused_plot"},
    )
    sparse_plot = execute_plot(sparse_plot)
    focus_plot = execute_plot(focus_plot)
    dense_plot = execute_plot(dense_plot)
    canonical_plot = code_cell(
        canonical_tighter_layout_plot_source(),
        {"m1_final": True, "alpha": 0.3, "kind": "canonical_tighter_layout_plot"},
    )
    canonical_plot = execute_plot(canonical_plot)
    if M1A_TARGET.exists():
        m1a_plot = execute_plot(m1a_plot)
    if ALPHA10_GAUSSIAN_TARGET.exists():
        dense_gaussian_plot = execute_plot(dense_gaussian_plot)
    nb = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell("# M1 upsample: final comparisons\n\nSelf-contained 100-trial experiments using the fast WERM solvers. Run each experiment cell, then its following plot cell."),
            nbformat.v4.new_markdown_cell("## M1b: $\\alpha=0.3$\n\nMcScan, WERM+L1 (2-fold full joint CV), WERM unpenalized (10-fold CV), DCDP, DPDU, and MOSEG."),
            sparse_exp, sparse_plot, focus_plot,
            nbformat.v4.new_markdown_cell("## M1b dense: $\\alpha=1.0$\n\nMcScan versus WERM unpenalized with 10-fold CV."),
            dense_exp, dense_plot,
            nbformat.v4.new_markdown_cell(
                "## Canonical M1 tighter-layout plots ($\\alpha=0.3$)\n\n"
                "The exact plotting structure beneath `Load and plot with tighter layout` "
                "in `M1 Comparison Experiments.ipynb`, adapted to the final method keys."
            ),
            canonical_plot,
            nbformat.v4.new_markdown_cell(
                "## M1a: corrected comparison ($\\alpha=0.5$)\n\n"
                "The original M1a design with 100 paired trials. McScan and native "
                "single-scale MOSEG both receive the known count, L-1=2; WERM uses "
                "the original direct unpenalized fixed-$L$ call (no CV) and the fast "
                "exact least-squares solver. Native MOSEG uses n.cps=2 and refined.cps; "
                "it can return fewer than two estimates if too few candidate peaks "
                "are available. These native outputs are recorded without padding."
            ),
            m1a_exp,
            m1a_plot,
            nbformat.v4.new_markdown_cell(
                "## M1b dense: $\\alpha=1.0$ with Gaussian noise\n\n"
                "A separate 100-trial paired rerun of the dense experiment, changing "
                "the noise draw from Student-$t_4$ to Gaussian with the same scale, "
                "and using unpenalized least-squares WERM (exact coefficient solver). "
                "Ten-fold count selection retains absolute-error CV scoring."
            ),
            dense_gaussian_exp,
            dense_gaussian_plot,
        ],
        metadata=deepcopy(source_nb.metadata),
    )
    nb.metadata["m1_final"] = True
    if TARGET.exists():
        previous = nbformat.read(TARGET, as_version=4)
        for new_cell, old_cell in zip(nb.cells, previous.cells):
            if (new_cell.cell_type == old_cell.cell_type == "code"
                    and new_cell.source == old_cell.source
                    and not new_cell.outputs):
                new_cell.outputs = old_cell.outputs
                new_cell.execution_count = old_cell.execution_count
    nbformat.write(nb, TARGET)
    print(TARGET)


if __name__ == "__main__":
    main()
