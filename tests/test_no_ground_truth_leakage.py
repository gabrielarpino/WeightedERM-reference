import ast
import inspect

import nbformat

import WERM.comparison as comparison
from scripts.update_m3_glmbsa_folds import update_notebook


REDO_NOTEBOOKS = (
    "M1_upsample_redo.ipynb",
    "M1_upsample_redo2.ipynb",
    "M2_upsample_redo.ipynb",
    "M2_upsample_redo2.ipynb",
    "M3_upsample_redo.ipynb",
    "M3_upsample_redo2.ipynb",
    "M3_upsample_redo3.ipynb",
    "M3_upsample_redo_delta6.ipynb",
)

ESTIMATOR_CALLEES = {
    "WERM.comparison.DCDP",
    "WERM.comparison.DPDU",
    "WERM.comparison.MOSEG",
    "WERM.comparison.glm_bsa3",
    "WERM.comparison.run_charcoal",
    "WERM.comparison.run_mcscan",
    "WERM.estimators.chgpt_k_fold_cv",
    "WERM.estimators.weighted_erm",
    "WERM.estimators.weighted_erm_num_chgptCV",
    "_glm_bsa3_cv",
    "_weighted_erm_joint_cv",
}

FORBIDDEN_ORACLE_NAMES = {
    "fractional_chgpt_locations",
    "true_chgpts",
    "true_chgpt_locations_fn",
    "fixed_noise_scale",
    "noise_std",
    "signal_alpha",
    "alpha",
    "sigma",
    "kappa",
    "α",
    "σ",
    "σ_w",
    "σ_l_arr",
    "κ",
    "B_tilde",
    "beta_true",
}


def _code_source(path):
    notebook = nbformat.read(path, as_version=4)
    if path in ("M3_upsample_redo2.ipynb", "M3_upsample_redo_delta6.ipynb"):
        notebook = update_notebook(notebook)
    return "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )


def _callee_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _callee_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def test_estimator_calls_do_not_receive_ground_truth_or_its_count():
    forbidden = (
        "kmax=len(fractional_chgpt_locations)",
        "num_chgpts=len(fractional_chgpt_locations)",
        "run_charcoal(np.array(X_scaled), np.array(y), fixed_noise_scale, true_chgpt_locations_fn(n))",
    )
    for path in REDO_NOTEBOOKS:
        source = _code_source(path)
        for expression in forbidden:
            assert expression not in source, f"ground-truth leakage in {path}: {expression}"


def test_charcoal_public_api_cannot_accept_ground_truth():
    assert tuple(inspect.signature(comparison.run_charcoal).parameters) == (
        "X_unit",
        "Y_unit",
    )


def test_estimator_calls_do_not_receive_dgp_oracle_parameters():
    for path in REDO_NOTEBOOKS:
        notebook = nbformat.read(path, as_version=4)
        if path in ("M3_upsample_redo2.ipynb", "M3_upsample_redo_delta6.ipynb"):
            notebook = update_notebook(notebook)
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            tree = ast.parse(cell.source)
            for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
                callee = _callee_name(call.func)
                if callee not in ESTIMATOR_CALLEES:
                    continue
                argument_nodes = [*call.args, *(kw.value for kw in call.keywords)]
                names = {
                    node.id
                    for argument in argument_nodes
                    for node in ast.walk(argument)
                    if isinstance(node, ast.Name)
                }
                leaked = names & FORBIDDEN_ORACLE_NAMES
                assert not leaked, f"oracle parameter(s) {sorted(leaked)} passed to {callee} in {path}"


def test_m2_truth_free_moseg_cannot_reuse_truth_informed_checkpoints():
    source = _code_source("M2_upsample_redo2.ipynb")
    assert "'schema_version': 5" in source
    assert "'moseg_num_chgpts': None" in source
    assert "legacy_checkpoint_signature" not in source
