"""Switch every expanded M3d GLMBSA-CV comparison to two folds."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat


def _replace_once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"expected one anchor, found {count}: {old[:100]!r}")
    return source.replace(old, new, 1)


def _prior_config(source: str):
    if "signal_alpha = 0.3" in source:
        return 10, "M3d-redo-alpha03-eleven-method-logistic-fold-comparison", True
    if "signal_alpha = 0.1" in source:
        return 10, "M3d-redo-alpha01-eleven-method-logistic-fold-comparison", True
    return 9, "M3d-redo-eleven-method-logistic-fold-comparison", False


def update_experiment(source: str) -> str:
    prior_schema, prior_experiment, has_alpha = _prior_config(source)
    branch_start = source.index("elif method_name == 'GLMBSA-CV':")
    branch_end = source.index(
        "elif method_name == 'GLMBSA-fixed0.18-2fold':", branch_start
    )
    branch = source[branch_start:branch_end]
    branch = _replace_once(branch, "num_folds=5,", "num_folds=2,")
    source = source[:branch_start] + branch + source[branch_end:]
    source = source.replace("GLMBSA-CV}", "GLMBSA-CV (2-fold)}")

    source = _replace_once(
        source,
        f"'schema_version': {prior_schema},",
        "'schema_version': 11,",
    )
    source = _replace_once(
        source,
        f"'experiment': '{prior_experiment}',",
        f"'experiment': '{prior_experiment}-glmbsa2fold',",
    )
    source = _replace_once(
        source,
        "    'glmbsa_c1_candidates': tuple(float(value) for value in glmbsa_c1_candidates),",
        "    'glmbsa_c1_candidates': tuple(float(value) for value in glmbsa_c1_candidates),\n"
        "    'glmbsa_cv_folds': 2,",
    )

    block_start = source.index("legacy_method_names =")
    block_end = source.index("completed_results = {}", block_start)
    if has_alpha:
        prior_dir = (
            "f'M3_upsample_d_redo_full_cv_delta{fixed_delta:g}_alpha{signal_alpha:g}_"
            f"{{checkpoint_mode}}_seed{{random_seed}}_p{{p}}_L{{L}}_schema{prior_schema}',"
        )
    else:
        prior_dir = (
            "f'M3_upsample_d_redo_full_cv_delta{fixed_delta:g}_"
            f"{{checkpoint_mode}}_seed{{random_seed}}_p{{p}}_L{{L}}_schema{prior_schema}',"
        )
    reuse_block = f"""rerun_methods = ('GLMBSA-CV',)

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
prior_checkpoint_signature.pop('glmbsa_cv_folds', None)
prior_checkpoint_signature['schema_version'] = {prior_schema}
prior_checkpoint_signature['experiment'] = '{prior_experiment}'
prior_checkpoint_dir = os.path.join(
    'results', 'checkpoints',
    {prior_dir}
)
legacy_results = {{}}
if not SMOKE_MODE and os.path.isdir(prior_checkpoint_dir):
    for setting_idx, _, trial_idx in tasks:
        prior_checkpoint = _load_trial_checkpoint(
            prior_checkpoint_dir, prior_checkpoint_signature, setting_idx, trial_idx
        )
        if prior_checkpoint is not None:
            legacy_results[(int(setting_idx), int(trial_idx))] = _without_method_results(
                prior_checkpoint[2], rerun_methods
            )
print(
    f'Validated {{len(legacy_results)}}/{{len(tasks)}} prior checkpoints; '
    'reusing every method except GLMBSA-CV'
)

"""
    source = source[:block_start] + reuse_block + source[block_end:]

    source = source.replace(
        "M3d_redo_delta3_eleven_methods_folds2-5-10_seed42",
        "M3d_redo_delta3_glmbsa2fold_eleven_methods_folds2-5-10_seed42",
    )
    source = source.replace(
        "M3d_redo_delta3_alpha03_eleven_methods_folds2-5-10_seed42",
        "M3d_redo_delta3_alpha03_glmbsa2fold_eleven_methods_folds2-5-10_seed42",
    )
    source = source.replace(
        "M3d_redo_delta3_alpha01_eleven_methods_folds2-5-10_seed42",
        "M3d_redo_delta3_alpha01_glmbsa2fold_eleven_methods_folds2-5-10_seed42",
    )
    source = source.replace(
        "M3d_redo_delta6_eleven_methods_folds2-5-10_seed42",
        "M3d_redo_delta6_glmbsa2fold_eleven_methods_folds2-5-10_seed42",
    )
    return source


def update_plot(source: str) -> str:
    source = source.replace("GLMBSA-CV}", "GLMBSA-CV (2-fold)}")
    source = source.replace(
        "M3d_redo_delta3_eleven_methods_folds2-5-10_seed42",
        "M3d_redo_delta3_glmbsa2fold_eleven_methods_folds2-5-10_seed42",
    )
    source = source.replace(
        "M3d_redo_delta3_alpha03_eleven_methods_folds2-5-10_seed42",
        "M3d_redo_delta3_alpha03_glmbsa2fold_eleven_methods_folds2-5-10_seed42",
    )
    source = source.replace(
        "M3d_redo_delta3_alpha01_eleven_methods_folds2-5-10_seed42",
        "M3d_redo_delta3_alpha01_glmbsa2fold_eleven_methods_folds2-5-10_seed42",
    )
    source = source.replace(
        "M3d_redo_delta6_eleven_methods_folds2-5-10_seed42",
        "M3d_redo_delta6_glmbsa2fold_eleven_methods_folds2-5-10_seed42",
    )
    return source


def update_notebook(notebook):
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        if (
            cell.metadata.get("m3_alpha_experiment") is not None
            or "'experiment': 'M3d-redo-eleven-method-logistic-fold-comparison'" in cell.source
        ):
            cell.source = update_experiment(cell.source)
            cell.outputs = []
            cell.execution_count = None
        elif (
            cell.metadata.get("m3_alpha_plot") is not None
            or "filename_to_load = 'results/M3d_redo_delta" in cell.source
        ):
            cell.source = update_plot(cell.source)
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
