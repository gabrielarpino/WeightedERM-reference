import ast
import json
import os
import pickle

from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "M3_upsample_redo.ipynb"


def _source():
    notebook = json.loads(NOTEBOOK.read_text())
    return "".join(notebook["cells"][1]["source"])


def _checkpoint_helpers():
    tree = ast.parse(_source())
    wanted = {
        "_trial_checkpoint_path",
        "_write_trial_checkpoint",
        "_load_trial_checkpoint",
    }
    definitions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert {node.name for node in definitions} == wanted
    namespace = {"os": os, "pickle": pickle}
    exec(
        compile(ast.Module(body=definitions, type_ignores=[]), str(NOTEBOOK), "exec"),
        namespace,
    )
    return namespace


def test_m3_trial_checkpoints_are_atomic_and_signature_safe(tmp_path):
    helpers = _checkpoint_helpers()
    signature = {"schema_version": 2, "experiment": "synthetic"}
    result = (2, 7, {"data_hash": "a" * 32, "value": [1, 2, 3]})

    assert helpers["_load_trial_checkpoint"](tmp_path, signature, 2, 7) is None
    helpers["_write_trial_checkpoint"](tmp_path, signature, result)
    assert helpers["_load_trial_checkpoint"](tmp_path, signature, 2, 7) == result
    assert (
        helpers["_load_trial_checkpoint"](
            tmp_path, {"schema_version": 3}, 2, 7
        )
        is None
    )
    assert not list(tmp_path.glob("*.tmp-*"))


def test_m3_recovery_loop_preserves_semantics_and_bounds_native_state():
    source = _source()
    assert "num_folds=5, n_jobs=1" in source
    assert "except TerminatedWorkerError:" in source
    assert "one task per fresh worker generation" in source
    assert "checkpoint_signature=checkpoint_signature" in source
    assert "pending = [task for task in tasks" in source
    assert "assert set(completed_results) == expected_keys" in source
    assert (
        "assert len(parallel_results) == num_trials * len(kappa_list)"
        in source
    )
