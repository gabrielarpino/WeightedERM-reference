# Build the self-contained publication-facing M2 final notebook.

from copy import deepcopy
import pickle
from pathlib import Path
import re

import nbformat

from tools.build_m1_upsample_final import (
    code_cell,
    distribution_block,
    distribution_calls,
    execute_plot,
    replace_method_block,
    style_block,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "M2_upsample_redo2.ipynb"
TARGET = ROOT / "M2_upsample_final.ipynb"
M2_COMPARISON = ROOT / "M2 Comparison Experiments.ipynb"
SOURCE_RESULT = ROOT / "results" / (
    "M2_upscale_c_extended_varydelta_alpha03_cp0205_McScan_charcoal_"
    "WERM-unp-2fold_WERM-unp-5fold_WERM-unp-10fold_charcoalmax9_"
    "results_p200_L4_delta0123456789101112131415_noisecale0.316_seed42.pkl"
)
TARGET_RESULT = ROOT / "results" / "M2_upsample_final_alpha03_seed42_100trials.pkl"
CHARCOAL_REFRESH_RESULT = ROOT / "results" / "M2_upsample_final_alpha03_seed42_100trials_charcoal_refresh.pkl"
METHOD_NAMES = ["McScan", "charcoal", "WERM-unp-10fold"]
METHOD_CATALOG = {
    "McScan": dict(name="McScan", label=r"\textbf{McScan}", color="#AA3377", marker="o", linestyle="--"),
    "charcoal": dict(name="charcoal", label=r"\textbf{charcoal}", color="#336677", marker="v", linestyle="-."),
    "WERM-unp-10fold": dict(name="WERM-unp-10fold", label=r"\textbf{WERM}", color="black", marker="D", linestyle="-"),
}


def _keep_record(record):
    shared = {"data_hash", "n", "true_chgpts", "fractional_true_chgpts", "methods_status", "diagnostic_charcoal"}
    filtered = {key: value for key, value in record.items() if key in shared or any(key.endswith("_" + name) for name in METHOD_NAMES)}
    if "methods_status" in filtered:
        filtered["methods_status"] = {name: filtered["methods_status"].get(name, False) for name in METHOD_NAMES}
    return filtered


def create_display_payload():
    with SOURCE_RESULT.open("rb") as handle:
        payload = pickle.load(handle)
    old = payload["results"]
    keep = {"true_chgpts", "n_values", "delta_values"}
    for name in METHOD_NAMES:
        keep.update({f"haus_{name}", f"size_{name}", f"runtime_{name}", f"chgpts_{name}"})
    results = {key: value for key, value in old.items() if key in keep}
    results["successful_trials"] = {name: old["successful_trials"][name] for name in METHOD_NAMES}
    if "trial_records" in old:
        results["trial_records"] = [(i, j, _keep_record(record)) for i, j, record in old["trial_records"]]
    if "summaries" in old:
        results["summaries"] = [summary for summary in old["summaries"] if summary.get("method") in METHOD_NAMES]
    if CHARCOAL_REFRESH_RESULT.exists():
        with CHARCOAL_REFRESH_RESULT.open("rb") as handle:
            refreshed = pickle.load(handle)["results"]
        for metric in ("haus", "size", "runtime", "chgpts"):
            results[f"{metric}_charcoal"] = refreshed[f"{metric}_charcoal"]
        results["successful_trials"]["charcoal"] = refreshed["successful_trials"]["charcoal"]
        if "trial_records" in refreshed:
            results["trial_records"] = [
                (i, j, _keep_record(record)) for i, j, record in refreshed["trial_records"]
            ]
        if "summaries" in refreshed:
            results["summaries"] = [
                summary for summary in results.get("summaries", [])
                if summary.get("method") != "charcoal"
            ] + [
                summary for summary in refreshed["summaries"]
                if summary.get("method") == "charcoal"
            ]
    params = dict(payload["params"])
    params["methods_evaluated"] = [deepcopy(METHOD_CATALOG[name]) for name in METHOD_NAMES]
    params["successful_trials_counts"] = results["successful_trials"]
    params["num_folds_unpenalized"] = (10,)
    params["source_result_file"] = str(SOURCE_RESULT.relative_to(ROOT))
    params["charcoal_refresh_file"] = (
        str(CHARCOAL_REFRESH_RESULT.relative_to(ROOT))
        if CHARCOAL_REFRESH_RESULT.exists() else None
    )
    with TARGET_RESULT.open("wb") as handle:
        pickle.dump({"results": results, "params": params}, handle)


def experiment_source(cell):
    source = replace_method_block("".join(cell["source"]), METHOD_NAMES, METHOD_CATALOG)
    source = source.replace(
        "delta_list = np.array([10.5])  # Exercise the charcoal-gated path.",
        "delta_list = np.array([1.5, 10.5])  # Exercise active and gated charcoal paths.",
    )
    method_line = "methods_to_evaluate = [method_catalog[name].copy() for name in FINAL_METHOD_NAMES]\n"
    method_selector = method_line + """requested_method_names = [name.strip() for name in os.environ.get('WERM_M2_METHODS', '').split(',') if name.strip()]\nif requested_method_names:\n    unknown_method_names = sorted(set(requested_method_names) - set(FINAL_METHOD_NAMES))\n    if unknown_method_names:\n        raise ValueError(f'Unknown WERM_M2_METHODS: {unknown_method_names}')\n    methods_to_evaluate = [method_catalog[name].copy() for name in requested_method_names]\n"""
    source = source.replace(method_line, method_selector, 1)
    source = source.replace(", charcoal_thresholds, checkpoint_dir=None", ", checkpoint_dir=None")
    calibration_injection = """            # Calibrate once per (n, p), outside the timed charcoal call.\n            if method_name == 'charcoal' and delta <= CHARCOAL_MAX_DELTA and n > p:\n                key = (n, p)\n                WERM.comparison._CHARCOAL_THRESH_CACHE[key] = charcoal_thresholds[key]\n\n"""
    source = source.replace(calibration_injection, "")
    calibration_start = source.index("# Prewarm each applicable calibration exactly once")
    calibration_end = source.index("checkpoint_signature =", calibration_start)
    source = source[:calibration_start] + source[calibration_end:]
    source = source.replace(",\n    'charcoal_thresholds': tuple((int(key[0]), int(key[1]), float(charcoal_thresholds[key])) for key in sorted(charcoal_thresholds))", "")
    source = source.replace(", charcoal_thresholds=charcoal_thresholds, checkpoint_dir=", ", checkpoint_dir=")
    source = source.replace("'schema_version': 5", "'schema_version': 7")
    source = source.replace("'experiment': 'M2c-alpha-0.3-unpenalized-fold-comparison'", "'experiment': 'M2-upsample-final-alpha03-10fold-charcoal-fixed10'")
    source = source.replace("M2_upsample_alpha03_unpenalized_folds_{checkpoint_mode}", "M2_upsample_final_alpha03_10fold_{checkpoint_mode}")
    source += r'''

# Isolated final artifact; rerunning this cell updates exactly what the plot cell reads.
FINAL_RESULT_FILE = "results/M2_upsample_final_alpha03_seed42_100trials.pkl"
if requested_method_names:
    _method_tag = "_".join(requested_method_names)
    FINAL_RESULT_FILE = FINAL_RESULT_FILE.replace(".pkl", f"_{_method_tag}_refresh.pkl")
if SMOKE_MODE:
    FINAL_RESULT_FILE = FINAL_RESULT_FILE.replace(".pkl", "_smoke.pkl")
with open(FINAL_RESULT_FILE, "wb") as handle:
    pickle.dump({"results": results, "params": {
        "p": p, "L": L, "delta_list": delta_list,
        "fixed_noise_scale": fixed_noise_scale, "num_trials": num_trials,
        "fractional_chgpt_locations": fractional_chgpt_locations,
        "n_jobs": n_jobs, "successful_trials_counts": results["successful_trials"],
        "methods_evaluated": methods_to_evaluate,
        "charcoal_threshold": WERM.comparison.CHARCOAL_THRESHOLD,
        "charcoal_seed": WERM.comparison.CHARCOAL_SEED,
        "charcoal_max_delta": CHARCOAL_MAX_DELTA,
        "num_folds_unpenalized": (10,), "random_seed": random_seed,
    }}, handle)
print(f"Saved final results to {FINAL_RESULT_FILE}")
'''
    return source


def plot_source(cell):
    source = "".join(cell["source"])
    source = re.sub(r"_FINAL_M1B_STYLES = \{.*?\n\}", lambda _match: style_block(METHOD_CATALOG), source, count=1, flags=re.DOTALL)
    box_start = source.index("# --- Option 3: Publication-Ready Box Plots ---")
    box_end = source.index("# --- Publication-Ready Changepoint Histograms ---", box_start)
    source = source[:box_start] + distribution_block() + "\n" + source[box_end:]
    source = source.replace('print("3. Box plots (publication style)...")\nplot_with_boxplots_publication()', distribution_calls().replace("M1_upsample_final", "M2_upsample_final"))
    save_marker = "# Save results to a file"
    if save_marker in source:
        source = source[:source.index(save_marker)] + "print('Plots complete; experiment artifact was not modified.')\n"
    header = r'''import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

SMOKE_MODE = os.environ.get("WERM_SMOKE") == "1"
FINAL_RESULT_FILE = "results/M2_upsample_final_alpha03_seed42_100trials.pkl"
if SMOKE_MODE:
    FINAL_RESULT_FILE = FINAL_RESULT_FILE.replace(".pkl", "_smoke.pkl")
with open(FINAL_RESULT_FILE, "rb") as handle:
    _payload = pickle.load(handle)
results = _payload["results"]
_params = _payload["params"]
p, L = _params["p"], _params["L"]
delta_list = np.asarray(_params["delta_list"])
fixed_noise_scale = _params["fixed_noise_scale"]
num_trials = _params["num_trials"]
fractional_chgpt_locations = np.asarray(_params["fractional_chgpt_locations"])
random_seed = _params["random_seed"]
CHARCOAL_MAX_DELTA = _params.get("charcoal_max_delta", 9.0)
methods_to_evaluate = _params["methods_evaluated"]
method_names_list = [method["name"] for method in methods_to_evaluate]
print(f"Loaded {FINAL_RESULT_FILE}")

'''
    return header + source


def canonical_comparison_plot_source():
    """Reuse the reference notebook's final plotting setup verbatim where possible."""
    reference = nbformat.read(M2_COMPARISON, as_version=4)
    styles = reference.cells[29].source
    plots = reference.cells[30].source

    # The final experiment retains the reference methods except that its WERM
    # checkpoint key explicitly records the selected 10-fold CV procedure.
    styles = styles.replace("'name': 'WERM-unp'", "'name': 'WERM-unp-10fold'")
    _werm_style_anchor = "'name': 'WERM-unp-10fold'"
    _werm_style_start = styles.index(_werm_style_anchor)
    _werm_style_end = styles.index("    },", _werm_style_start)
    _werm_style = styles[_werm_style_start:_werm_style_end]
    styles = (
        styles[:_werm_style_start]
        + _werm_style.replace("'marker': '*'", "'marker': 'D'")
        + styles[_werm_style_end:]
    )
    plots = plots.replace(
        "results/M2c_extended_varydelta_alpha03_cp0205_McScan_charcoal_"
        "WERM-unp_results_p200_L4_delta0123456789101112131415_"
        "noisecale0.316_seed42.pkl",
        "results/M2_upsample_final_alpha03_seed42_100trials.pkl",
    )
    plots = plots.replace(
        "# selected_deltas = δ_list[::4]\n"
        "    selected_deltas = np.array([2.5, 5.5, 8.5])\n"
        "    # selected_deltas_idx = list(range(len(δ_list)))[::4]\n"
        "    selected_deltas_idx = list(np.searchsorted(δ_list, selected_deltas))",
        "selected_deltas_idx = list(range(len(δ_list)))[4:9:2]\n"
        "    selected_deltas = δ_list[4:9:2]",
        1,
    )
    plots = plots.replace(
        "label=labels[i], marker=markers[i % len(markers)], ls=linestyles[i % len(linestyles)],",
        "label=labels[i], marker=methods_to_evaluate[i]['marker'], "
        "ls=methods_to_evaluate[i]['linestyle'],",
    )
    return styles + "\n\n" + plots


def main():
    source_nb = nbformat.read(SOURCE, as_version=4)
    create_display_payload()
    experiment = code_cell(experiment_source(source_nb.cells[2]), {"m2_final": True, "alpha": 0.3, "kind": "experiment"})
    plot = code_cell(plot_source(source_nb.cells[4]), {"m2_final": True, "alpha": 0.3, "kind": "plot"})
    plot = execute_plot(plot)
    canonical_plot = code_cell(
        canonical_comparison_plot_source(),
        {"m2_final": True, "alpha": 0.3, "kind": "canonical_comparison_plot"},
    )
    canonical_plot = execute_plot(canonical_plot)
    nb = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell("# M2 upsample: final comparison\n\nSelf-contained 100-trial M2c experiment using exact least-squares WERM."),
            nbformat.v4.new_markdown_cell("## Experiment\n\nMcScan, charcoal (documented fixed threshold 10; gated through $\\delta \\leq 9$), and unpenalized WERM with 10-fold CV."),
            experiment,
            nbformat.v4.new_markdown_cell("## Plots\n\nComplete publication plot suite, including M3-style boxplots with deterministic raw-point jitter."),
            plot,
            nbformat.v4.new_markdown_cell(
                "## Canonical M2 Comparison Experiment plots\n\n"
                "The exact final plotting format from `M2 Comparison Experiments.ipynb`, "
                "including its delta subsampling and histogram setup."
            ),
            canonical_plot,
        ],
        metadata=deepcopy(source_nb.metadata),
    )
    nb.metadata["m2_final"] = True
    nbformat.write(nb, TARGET)
    print(TARGET)


if __name__ == "__main__":
    main()
