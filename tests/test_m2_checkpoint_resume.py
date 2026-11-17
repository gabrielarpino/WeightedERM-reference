import ast
import json
import os
import pickle
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "M2_upsample_redo.ipynb"
REDO_NOTEBOOK = Path(__file__).resolve().parents[1] / "M2_upsample_redo2.ipynb"


def _source():
    notebook = json.loads(NOTEBOOK.read_text())
    return "".join(notebook["cells"][1]["source"])


def _redo_source():
    notebook = json.loads(REDO_NOTEBOOK.read_text())
    return "".join(
        line
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        for line in cell["source"]
    )


def _helpers():
    tree = ast.parse(_source())
    wanted = {"_trial_checkpoint_path", "_write_trial_checkpoint", "_load_trial_checkpoint"}
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    assert {node.name for node in definitions} == wanted
    namespace = {"os": os, "pickle": pickle}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace


def test_m2_trial_checkpoints_are_atomic_and_signature_safe(tmp_path):
    helpers = _helpers()
    signature = {"schema_version": 2, "experiment": "M2c-alpha-0.3"}
    result = (2, 7, {"data_hash": "a" * 32, "value": [1, 2, 3]})
    assert helpers["_load_trial_checkpoint"](tmp_path, signature, 2, 7) is None
    helpers["_write_trial_checkpoint"](tmp_path, signature, result)
    assert helpers["_load_trial_checkpoint"](tmp_path, signature, 2, 7) == result
    assert helpers["_load_trial_checkpoint"](tmp_path, {"schema_version": 3}, 2, 7) is None
    assert not list(tmp_path.glob("*.tmp-*"))


def test_m2_corrupt_checkpoint_is_rejected(tmp_path):
    helpers = _helpers()
    signature = {"schema_version": 2, "experiment": "M2c-alpha-0.3"}
    path = tmp_path / "setting_00_trial_000.pkl"
    path.write_bytes(b"not a pickle")
    assert helpers["_load_trial_checkpoint"](tmp_path, signature, 0, 0) is None


def test_m2_missing_only_resume_keeps_completed_order_after_worker_failure(tmp_path):
    helpers = _helpers()
    signature = {"schema_version": 2, "experiment": "M2c-alpha-0.3"}
    tasks = [(0, 0.5, 0), (0, 0.5, 1), (1, 1.5, 0)]
    result = lambda task: (task[0], task[2], {"data_hash": "b" * 32, "value": task})
    helpers["_write_trial_checkpoint"](tmp_path, signature, result(tasks[0]))
    completed = {}
    for task in tasks:
        loaded = helpers["_load_trial_checkpoint"](tmp_path, signature, task[0], task[2])
        if loaded is not None:
            completed[(task[0], task[2])] = loaded
    pending = [task for task in tasks if (task[0], task[2]) not in completed]
    assert pending == [tasks[1], tasks[2]]
    try:
        raise RuntimeError("simulated worker failure")
    except RuntimeError:
        pass
    helpers["_write_trial_checkpoint"](tmp_path, signature, result(tasks[1]))
    helpers["_write_trial_checkpoint"](tmp_path, signature, result(tasks[2]))
    ordered = [helpers["_load_trial_checkpoint"](tmp_path, signature, task[0], task[2]) for task in tasks]
    assert [item[:2] for item in ordered] == [(0, 0), (0, 1), (1, 0)]


def test_m2_resume_loop_preserves_order_and_bounds_native_state():
    source = _source()
    assert "from joblib.externals.loky.process_executor import TerminatedWorkerError" in source
    assert "one task per fresh worker generation" not in source or "batch = pending[:batch_n_jobs]" in source
    assert "checkpoint_signature=checkpoint_signature" in source
    assert "'charcoal_thresholds': tuple((int(key[0]), int(key[1]), float(charcoal_thresholds[key]))" in source
    assert "pending = [task for task in tasks" in source
    assert "assert set(completed_results) == expected_keys" in source
    assert "assert len(parallel_results) == num_trials * len(delta_list)" in source
    ast.parse(source)


def test_m2_redo_catches_the_installed_rpy2_runtime_error():
    source = _redo_source()
    assert "rpy2.rinterface_lib.embedded.RRuntimeError" in source
    assert "rpy2.rinterface.RRuntimeError" not in source


def test_m2_redo_gates_charcoal_at_the_original_delta_cutoff():
    source = _redo_source()
    assert "CHARCOAL_MAX_DELTA = 9.0" in source
    assert "if delta <= CHARCOAL_MAX_DELTA:" in source
    assert "cp = None  # Method intentionally not run above the paper cutoff." in source
    assert "delta <= CHARCOAL_MAX_DELTA" in source.split("applicable_charcoal_keys", 1)[1]
    assert "'charcoal_max_delta': float(CHARCOAL_MAX_DELTA)" in source
