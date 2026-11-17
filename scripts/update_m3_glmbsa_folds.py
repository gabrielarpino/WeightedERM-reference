"""Add 2-, 5-, and 10-fold GLMBSA CV to every expanded M3d experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat


OLD_ARTIFACT_TOKEN = "glmbsa2fold_eleven_methods"
NEW_ARTIFACT_TOKEN = "glmbsa2-5-10_thirteen_methods"


def _replace_once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"expected one anchor, found {count}: {old[:100]!r}")
    return source.replace(old, new, 1)


def _old_experiment(source: str) -> str:
    if "signal_alpha = 0.3" in source:
        return "M3d-redo-alpha03-eleven-method-logistic-fold-comparison-glmbsa2fold"
    if "signal_alpha = 0.1" in source:
        return "M3d-redo-alpha01-eleven-method-logistic-fold-comparison-glmbsa2fold"
    return "M3d-redo-eleven-method-logistic-fold-comparison-glmbsa2fold"


def _prior_dir_expression(source: str) -> str:
    if "signal_alpha =" in source:
        return (
            "f'M3_upsample_d_redo_full_cv_delta{fixed_delta:g}_alpha{signal_alpha:g}_"
            "{checkpoint_mode}_seed{random_seed}_p{p}_L{L}_schema11',"
        )
    return (
        "f'M3_upsample_d_redo_full_cv_delta{fixed_delta:g}_"
        "{checkpoint_mode}_seed{random_seed}_p{p}_L{L}_schema11',"
    )


def update_experiment(source: str) -> str:
    # kmax is a pre-specified method budget. It must never be inferred from the
    # number of simulated ground-truth changepoints.
    source = source.replace(
        "kmax=len(fractional_chgpt_locations)", "kmax=glmbsa_kmax"
    )
    source = source.replace(
        "glmbsa_kmax = 2  # pre-specified; independent of ground truth",
        "glmbsa_kmax = L - 2  # original M3d protocol; independent of ground truth",
    )
    if "glmbsa_kmax = L - 2" not in source:
        source = source.replace(
            "glmbsa_cv_folds = ",
            "glmbsa_kmax = L - 2  # original M3d protocol; independent of ground truth\n"
            "glmbsa_cv_folds = ",
            1,
        )
    if "'schema_version': 12" in source:
        return source

    old_experiment = _old_experiment(source)
    source = _replace_once(
        source,
        "methods_to_evaluate = [\n"
        "    {'name': 'GLMBSA-CV', 'label': '\\\\textbf{GLMBSA-CV (2-fold)}', 'color': '#E69F00', 'marker': 'p', 'linestyle': ':'},",
        "glmbsa_cv_folds = {'GLMBSA-CV': 2, 'GLMBSA-CV-5fold': 5, 'GLMBSA-CV-10fold': 10}\n"
        "methods_to_evaluate = [\n"
        "    {'name': 'GLMBSA-CV', 'label': '\\\\textbf{GLMBSA-CV (2-fold)}', 'color': '#E69F00', 'marker': 'p', 'linestyle': ':'},\n"
        "    {'name': 'GLMBSA-CV-5fold', 'label': 'GLMBSA-CV (5-fold)', 'color': '#56B4E9', 'marker': '<', 'linestyle': ':'},\n"
        "    {'name': 'GLMBSA-CV-10fold', 'label': 'GLMBSA-CV (10-fold)', 'color': '#009E73', 'marker': '>', 'linestyle': ':'},",
    )

    branch_start = source.index("elif method_name == 'GLMBSA-CV':")
    branch_end = source.index(
        "elif method_name == 'GLMBSA-fixed0.18-2fold':", branch_start
    )
    replacement = """elif method_name in glmbsa_cv_folds:
                delta_fraction = 1/10
                selected_c1, cp, glmbsa_cv_scores = _glm_bsa3_cv(
                    X_scaled, y, kmax=glmbsa_kmax,
                    delta=delta_fraction, c1_candidates=glmbsa_c1_candidates,
                    num_folds=glmbsa_cv_folds[method_name],
                )
                trial_results[f'selected_c1_{method_name}'] = float(selected_c1)
                trial_results[f'cv_scores_{method_name}'] = np.asarray(glmbsa_cv_scores).copy()
                print(
                    f'{method_name} selected c1={selected_c1:g}; cp={cp}; '
                    f'folds={glmbsa_cv_folds[method_name]}'
                )

            """
    source = source[:branch_start] + replacement + source[branch_end:]

    source = _replace_once(source, "'schema_version': 11,", "'schema_version': 12,")
    source = _replace_once(
        source,
        f"'experiment': '{old_experiment}',",
        f"'experiment': '{old_experiment.replace('eleven-method', 'thirteen-method').replace('-glmbsa2fold', '-glmbsa2-5-10')}',",
    )
    source = _replace_once(
        source, "    'glmbsa_cv_folds': 2,", "    'glmbsa_cv_folds': (2, 5, 10),"
    )

    block_start = source.index("rerun_methods =")
    block_end = source.index("completed_results = {}", block_start)
    reuse_block = f"""rerun_methods = ('GLMBSA-CV-5fold', 'GLMBSA-CV-10fold')

def _without_method_results(trial_result, method_names):
    cleaned = dict(trial_result)
    for key in tuple(cleaned):
        if any(key.endswith(f'_{{method_name}}') for method_name in method_names):
            cleaned.pop(key, None)
    cleaned['methods_status'] = dict(cleaned.get('methods_status', {{}}))
    for method_name in method_names:
        cleaned['methods_status'].pop(method_name, None)
    cleaned.pop('reused_methods', None)
    return cleaned

prior_checkpoint_signature = dict(checkpoint_signature)
prior_checkpoint_signature['schema_version'] = 11
prior_checkpoint_signature['experiment'] = '{old_experiment}'
prior_checkpoint_signature['methods'] = tuple(
    name for name in method_names_list if name not in rerun_methods
)
prior_checkpoint_signature['glmbsa_cv_folds'] = 2
prior_checkpoint_dir = os.path.join(
    'results', 'checkpoints',
    {_prior_dir_expression(source)}
)
legacy_results = {{}}
if os.path.isdir(prior_checkpoint_dir):
    for setting_idx, _, trial_idx in tasks:
        prior_checkpoint = _load_trial_checkpoint(
            prior_checkpoint_dir, prior_checkpoint_signature, setting_idx, trial_idx
        )
        if prior_checkpoint is not None:
            legacy_results[(int(setting_idx), int(trial_idx))] = _without_method_results(
                prior_checkpoint[2], rerun_methods
            )
print(
    f'Validated {{len(legacy_results)}}/{{len(tasks)}} schema-11 checkpoints; '
    'reusing completed methods and running GLMBSA 5- and 10-fold CV'
)

"""
    source = source[:block_start] + reuse_block + source[block_end:]
    source = source.replace(OLD_ARTIFACT_TOKEN, NEW_ARTIFACT_TOKEN)
    source = source.replace("eleven-method comparison", "thirteen-method comparison")
    return source


def update_plot(source: str) -> str:
    source = source.replace(OLD_ARTIFACT_TOKEN, NEW_ARTIFACT_TOKEN)
    if "'name': 'GLMBSA-CV-5fold'" not in source:
        source = _replace_once(
            source,
            "    {'name': 'GLMBSA-CV', 'label': '\\\\textbf{GLMBSA-CV (2-fold)}', 'color': '#E69F00', 'marker': 'p', 'linestyle': ':'},",
            "    {'name': 'GLMBSA-CV', 'label': '\\\\textbf{GLMBSA-CV (2-fold)}', 'color': '#E69F00', 'marker': 'p', 'linestyle': ':'},\n"
            "    {'name': 'GLMBSA-CV-5fold', 'label': 'GLMBSA-CV (5-fold)', 'color': '#56B4E9', 'marker': '<', 'linestyle': ':'},\n"
            "    {'name': 'GLMBSA-CV-10fold', 'label': 'GLMBSA-CV (10-fold)', 'color': '#009E73', 'marker': '>', 'linestyle': ':'},",
        )
    return source


def update_notebook(notebook):
    for cell in notebook.cells:
        if cell.metadata.get("m3_two_method_replot"):
            continue
        if cell.cell_type != "code":
            continue
        if "def run_single_trial(" in cell.source and "GLMBSA-CV" in cell.source:
            updated = update_experiment(cell.source)
            if updated != cell.source:
                cell.source = updated
                cell.outputs = []
                cell.execution_count = None
        elif "filename_to_load = 'results/M3d_redo" in cell.source:
            updated = update_plot(cell.source)
            if updated != cell.source:
                cell.source = updated
                cell.outputs = []
                cell.execution_count = None
    return notebook


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebooks", nargs="+")
    args = parser.parse_args()
    for name in args.notebooks:
        path = Path(name)
        notebook = nbformat.read(path, as_version=4)
        nbformat.write(update_notebook(notebook), path)


if __name__ == "__main__":
    main()
