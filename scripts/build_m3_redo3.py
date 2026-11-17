"""Build the self-contained four-experiment M3 redo3 notebook."""

from __future__ import annotations

import copy
import re
from pathlib import Path

import nbformat


SOURCE_DELTA3 = Path("M3_upsample_redo2.ipynb")
SOURCE_DELTA6 = Path("M3_upsample_redo_delta6.ipynb")
OUTPUT = Path("M3_upsample_redo3.ipynb")

OLD_ARTIFACT_TOKEN = "glmbsa2-5-10_thirteen_methods_folds2-5-10_seed42"
NEW_ARTIFACT_TOKEN = (
    "glmbsa2-5-10_thirteen_methods_folds2-5-10_"
    "widegrid_c0018-018-18_C003-03-3_seed42"
)

RERUN_METHODS = (
    "GLMBSA-CV",
    "GLMBSA-CV-5fold",
    "GLMBSA-CV-10fold",
    "WERM-L1-fullCV",
    "WERM-L1-fullCV-2fold",
    "WERM-L1-fullCV-10fold",
)


def _transform_experiment(source: str) -> str:
    match = re.search(r"'experiment': '([^']+)'", source)
    if match is None:
        raise RuntimeError("experiment signature anchor is missing")
    old_experiment = match.group(1)

    source = source.replace(
        "glmbsa_c1_candidates = (0.09, 0.18, 0.36)",
        "glmbsa_c1_candidates = (0.018, 0.18, 1.8)",
    )
    source = source.replace(
        "joint_l1_constants = (0.15, 0.3, 0.6)",
        "joint_l1_constants = (0.03, 0.3, 3.0)",
    )
    source = source.replace("'schema_version': 12,", "'schema_version': 13,", 1)
    source = source.replace(
        f"'experiment': '{old_experiment}',",
        f"'experiment': '{old_experiment}-widegrid-c0018-018-18-C003-03-3',",
        1,
    )
    source = source.replace(
        "rerun_methods = ('GLMBSA-CV-5fold', 'GLMBSA-CV-10fold')",
        f"rerun_methods = {RERUN_METHODS!r}",
        1,
    )

    reuse_start = source.index("prior_checkpoint_signature = dict(checkpoint_signature)")
    reuse_end = source.index("legacy_results = {}", reuse_start)
    reuse_setup = f"""prior_checkpoint_signature = dict(checkpoint_signature)
prior_checkpoint_signature['schema_version'] = 12
prior_checkpoint_signature['experiment'] = '{old_experiment}'
prior_checkpoint_signature['joint_l1_constants'] = (0.15, 0.3, 0.6)
prior_checkpoint_signature['glmbsa_c1_candidates'] = (0.09, 0.18, 0.36)
prior_checkpoint_dir = checkpoint_dir.rsplit('_schema', 1)[0] + '_schema12'
"""
    source = source[:reuse_start] + reuse_setup + source[reuse_end:]
    source = source.replace(
        "schema-11 checkpoints; '",
        "schema-12 checkpoints; '",
        1,
    ).replace(
        "reusing completed methods and running GLMBSA 5- and 10-fold CV",
        "reusing unchanged methods and rerunning all grid-dependent GLMBSA/WERM CV methods",
        1,
    )
    source = source.replace(OLD_ARTIFACT_TOKEN, NEW_ARTIFACT_TOKEN)

    if "signal_alpha = 0.3" in source:
        scenario = "delta=3, alpha=0.3"
    elif "signal_alpha = 0.1" in source:
        scenario = "delta=3, alpha=0.1"
    elif "fixed_delta = 6.0" in source:
        scenario = "delta=6, alpha=0.5"
    else:
        scenario = "delta=3, alpha=0.5"
    return f"# redo3 scenario: {scenario}\n" + source


def _transform_cell(cell):
    result = copy.deepcopy(cell)
    if result.cell_type == "code":
        if "def run_single_trial(" in result.source:
            result.source = _transform_experiment(result.source)
        else:
            result.source = result.source.replace(
                OLD_ARTIFACT_TOKEN, NEW_ARTIFACT_TOKEN
            )
        result.outputs = []
        result.execution_count = None
    return result


def build_notebook():
    delta3 = nbformat.read(SOURCE_DELTA3, as_version=4)
    delta6 = nbformat.read(SOURCE_DELTA6, as_version=4)

    notebook = copy.deepcopy(delta3)
    notebook.cells = [_transform_cell(cell) for cell in delta3.cells]
    notebook.cells[0].source = (
        "# M3d redo3: wider GLMBSA and WERM tuning grids\n\n"
        "Four self-contained logistic experiments run serially: delta=3 with "
        "alpha=0.5, 0.3, and 0.1, followed by delta=6 with alpha=0.5. "
        "Estimator inputs are truth-free; ground truth is used only for simulation "
        "and post-fit evaluation."
    )

    delta6_heading = copy.deepcopy(delta6.cells[0])
    delta6_heading.source = "## M3d redo3 at $\\delta=6$, $\\alpha=0.5$"
    notebook.cells.append(delta6_heading)
    notebook.cells.extend(_transform_cell(cell) for cell in delta6.cells[1:])
    for index, cell in enumerate(notebook.cells):
        cell.id = f"m3redo3-{index:02d}"
    nbformat.validate(notebook)
    return notebook


def main() -> None:
    nbformat.write(build_notebook(), OUTPUT)


if __name__ == "__main__":
    main()
