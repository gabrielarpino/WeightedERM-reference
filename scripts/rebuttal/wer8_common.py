#!/usr/bin/env python3
"""Resumable M2/M3 experiment support for WER-8."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

# Set inner numerical-library thread limits before importing NumPy/JAX.
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[_name] = "1"
os.environ.setdefault(
    "XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
from joblib import Parallel, delayed

from WERM.config import η_to_ψ
from WERM.model import q_linear_model, q_logistic_model
from WERM.performance_measures import hausdorff
from WERM.signal_priors import SparseDiffSignal, SparseGaussianSignal
import WERM.estimators


BASE_SEED = 42
REPETITIONS = 100
M2_DELTAS = np.arange(0.5, 16.0, 1.0)
M3_KAPPAS = np.sqrt(np.array([1.0, 10.0, 25.0, 50.0, 100.0, 500.0, 1000.0, 2000.0]))

CONFIG = {
    "m2": {
        "parameters": M2_DELTAS,
        "parameter_name": "delta",
        "methods": ("McScan", "charcoal", "WERM-unp"),
    },
    "m3": {
        "parameters": M3_KAPPAS,
        "parameter_name": "kappa",
        "methods": ("GLMBSA", "WERM-CV"),
    },
}


def squared_loss(u, v):
    return jnp.power(u - v, 2)


def absolute_loss(u, v):
    return jnp.abs(u - v)


def logistic_loss(u, v):
    # Preserve the final M3 notebook's loss expression exactly.
    return jnp.log(1 + jnp.exp(u)) - jnp.multiply(v, u)


def sample_corr(p: int, df: int = 10, seed: int | None = None) -> np.ndarray:
    """Final M3 notebook's Haar-based correlation-matrix generator."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((p, p)))
    eigenvalues = rng.chisquare(df, size=p)
    matrix = q.T @ np.diag(eigenvalues) @ q
    diagonal = np.sqrt(np.diag(matrix))
    return matrix / np.outer(diagonal, diagonal)


def repetition_seed(workflow: str, parameter_index: int, repetition: int) -> int:
    """Stable independent seed for a resumable notebook task."""
    workflow_code = {"m2": 2, "m3": 3}[workflow]
    sequence = np.random.SeedSequence(
        [BASE_SEED, workflow_code, int(parameter_index), int(repetition)]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def effective_workers(requested: int) -> tuple[int, int]:
    if requested < 1:
        raise ValueError("workers must be positive")
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:
        available = os.cpu_count() or 1
    return min(requested, 32, available), available


def competitor_dependency_status() -> tuple[bool, str]:
    check = subprocess.run(
        [sys.executable, "-c", "import WERM.comparison"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if check.returncode == 0:
        return True, "WERM.comparison and its R dependencies imported"
    detail = (check.stderr or check.stdout).strip().splitlines()
    return False, detail[-1] if detail else "WERM.comparison import failed"


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n"
    )
    os.replace(temporary, path)


def _record_path(artifact_root: Path, workflow: str, parameter_index: int, repetition: int) -> Path:
    return (
        artifact_root
        / "raw"
        / workflow
        / f"parameter_{parameter_index:02d}"
        / f"repetition_{repetition:03d}.json"
    )


def _method_result(method: str, status: str, **kwargs) -> dict:
    result = {
        "method": method,
        "status": status,
        "changepoints": None,
        "selected_changepoint_count": None,
        "selected_K": None,
        "empty_estimate": None,
        "hausdorff_over_n": None,
        "metric_note": None,
        "runtime_seconds": None,
        "failure": None,
    }
    result.update(kwargs)
    return result


def _score_estimate(method: str, cp, true_cp, n: int, runtime: float, selected_k=None) -> dict:
    cp = np.asarray(cp, dtype=int).reshape(-1)
    cp = cp[cp != -1]
    empty = len(cp) == 0
    return _method_result(
        method,
        "SUCCESS",
        changepoints=cp.tolist(),
        selected_changepoint_count=int(len(cp)),
        selected_K=int(selected_k) if selected_k is not None else int(len(cp) + 1),
        empty_estimate=empty,
        hausdorff_over_n=None if empty else float(hausdorff(cp, true_cp) / n),
        metric_note=(
            "undefined because the estimated changepoint set is empty"
            if empty
            else "normalized Hausdorff distance"
        ),
        runtime_seconds=float(runtime),
    )


def _failed_method(method: str, started: float, error: BaseException) -> dict:
    return _method_result(
        method,
        "FAILED",
        runtime_seconds=float(time.perf_counter() - started),
        failure=f"{type(error).__name__}: {error}",
    )


def _m2_trial(parameter_index: int, delta: float, dependency_ok: bool, dependency_note: str) -> tuple[dict, list[dict]]:
    p, upper_segments = 200, 4
    n = int(delta * p)
    noise_scale = float(np.sqrt(0.1))
    true_cp = (n * np.array([0.2, 0.5])).astype(int)

    prior = SparseDiffSignal(8, np.sqrt(400 * delta), 0.3, upper_segments)
    beta = prior.sample(p)
    x = np.random.normal(0, np.sqrt(1 / n), (n, p))
    epsilon = np.random.normal(0, 1, size=n) * noise_scale
    x_scaled = x * np.sqrt(n)
    beta_scaled = beta / np.sqrt(n)
    theta = x_scaled @ beta_scaled
    labels = η_to_ψ(true_cp, n)
    y = np.asarray(jax.vmap(q_linear_model, in_axes=(0, 0, 0))(theta, labels, epsilon))

    methods: list[dict] = []
    if dependency_ok:
        import WERM.comparison as comparison

        started = time.perf_counter()
        try:
            cp = comparison.run_mcscan(x_scaled, y.flatten(), num_chgpts=None, method="auto")
            methods.append(_score_estimate("McScan", cp, true_cp, n, time.perf_counter() - started))
        except Exception as error:
            methods.append(_failed_method("McScan", started, error))

        if delta <= 9.0:
            started = time.perf_counter()
            try:
                cp = comparison.run_charcoal(x_scaled, y, noise_scale, true_cp)
                methods.append(_score_estimate("charcoal", cp, true_cp, n, time.perf_counter() - started))
            except Exception as error:
                methods.append(_failed_method("charcoal", started, error))
        else:
            methods.append(
                _method_result(
                    "charcoal",
                    "NOT_RUN_HISTORICAL_RULE",
                    failure="final notebook skips charcoal when delta > 9",
                )
            )
    else:
        methods.extend(
            _method_result(method, "BLOCKED_DEPENDENCY", failure=dependency_note)
            for method in ("McScan", "charcoal")
        )

    started = time.perf_counter()
    try:
        selected_k, cp = WERM.estimators.weighted_erm_num_chgptCV(
            X=x,
            y=y,
            M=squared_loss,
            Delta_n=int(n / 40),
            Lmax=upper_segments,
            brute_force=False,
            return_gof=False,
            M_scorer=absolute_loss,
            num_folds=5,
            n_jobs=1,
        )
        methods.append(
            _score_estimate(
                "WERM-unp", cp, true_cp, n, time.perf_counter() - started, selected_k
            )
        )
    except Exception as error:
        traceback.print_exc()
        methods.append(_failed_method("WERM-unp", started, error))

    metadata = {
        "n": n,
        "p": p,
        "Lmax": upper_segments,
        "delta": float(delta),
        "noise_scale": noise_scale,
        "alpha": 0.3,
        "var_beta_1": 8,
        "sigma_w": float(np.sqrt(400 * delta)),
        "fractional_changepoints": [0.2, 0.5],
        "true_changepoints": true_cp.tolist(),
        "Delta_n": int(n / 40),
        "folds": 5,
        "estimator_path": "weighted_erm_num_chgptCV -> weighted_erm -> chgpt_gof_dp",
    }
    return metadata, methods


_M3_SIGMA = None


def _m3_trial(parameter_index: int, kappa: float, dependency_ok: bool, dependency_note: str) -> tuple[dict, list[dict]]:
    global _M3_SIGMA
    p, upper_segments, delta = 200, 4, 3.0
    n = int(delta * p)
    true_cp = (n * np.array([1 / 3, 8 / 15])).astype(int)
    if _M3_SIGMA is None:
        _M3_SIGMA = sample_corr(p, df=10, seed=BASE_SEED)

    prior = SparseGaussianSignal(0.5, delta, kappa * np.sqrt(delta) * np.ones(upper_segments))
    beta = prior.sample(p)
    x = np.random.normal(0, np.sqrt(1 / n), (n, p)) @ np.linalg.cholesky(_M3_SIGMA).T
    epsilon = np.random.uniform(0, 1, size=n)
    x_scaled = x * np.sqrt(n)
    beta_scaled = beta / np.sqrt(n)
    theta = x_scaled @ beta_scaled
    labels = η_to_ψ(true_cp, n)
    y = np.asarray(jax.vmap(q_logistic_model, in_axes=(0, 0, 0))(theta, labels, epsilon))

    methods: list[dict] = []
    if dependency_ok:
        import WERM.comparison as comparison

        started = time.perf_counter()
        try:
            cp = np.asarray(
                comparison.glm_bsa2(
                    x, y.reshape((n, 1)), kmax=upper_segments - 2,
                    c1=0.18, m1=None, delta=0.1, exact=False,
                )
            ) - 1
            cp = cp[(cp > 0) & (cp < n - 1)]
            methods.append(_score_estimate("GLMBSA", cp, true_cp, n, time.perf_counter() - started))
        except Exception as error:
            methods.append(_failed_method("GLMBSA", started, error))
    else:
        methods.append(_method_result("GLMBSA", "BLOCKED_DEPENDENCY", failure=dependency_note))

    lambda_hat = np.sqrt(2 * np.log(p) / n)
    l1_weight = 10 * n * lambda_hat

    def coefficient_penalty(beta_value):
        return l1_weight * jnp.sum(jnp.abs(beta_value))

    started = time.perf_counter()
    try:
        selected_k, cp = WERM.estimators.weighted_erm_num_chgptCV(
            X=x,
            y=y,
            M=logistic_loss,
            Delta_n=int(n / 10),
            Lmax=upper_segments,
            brute_force=False,
            return_gof=False,
            P_func=coefficient_penalty,
            M_scorer=logistic_loss,
            num_folds=5,
            # The notebook used 3 here while also running 36 outer jobs.  One
            # avoids nested parallelism without changing folds or tuning.
            n_jobs=1,
        )
        methods.append(
            _score_estimate(
                "WERM-CV", cp, true_cp, n, time.perf_counter() - started, selected_k
            )
        )
    except Exception as error:
        traceback.print_exc()
        methods.append(_failed_method("WERM-CV", started, error))

    metadata = {
        "n": n,
        "p": p,
        "Lmax": upper_segments,
        "delta": delta,
        "kappa": float(kappa),
        "noise_std_not_used": float(np.sqrt(0.6)),
        "alpha": 0.5,
        "sigma_l": float(kappa * np.sqrt(delta)),
        "fractional_changepoints": [1 / 3, 8 / 15],
        "true_changepoints": true_cp.tolist(),
        "Delta_n": int(n / 10),
        "folds": 5,
        "coefficient_l1_weight": float(l1_weight),
        "estimator_path": "weighted_erm_num_chgptCV -> weighted_erm -> chgpt_gof_dp",
        "inner_cv_workers": 1,
    }
    return metadata, methods


def _run_task(task: dict, artifact_root: str, commit: str, dependency_ok: bool, dependency_note: str) -> dict:
    artifact_path = Path(artifact_root)
    workflow = task["workflow"]
    parameter_index = task["parameter_index"]
    repetition = task["repetition"]
    seed = repetition_seed(workflow, parameter_index, repetition)
    record_path = _record_path(artifact_path, workflow, parameter_index, repetition)
    log_path = (
        artifact_path / "task_logs" / workflow / f"parameter_{parameter_index:02d}"
        / f"repetition_{repetition:03d}.txt"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)

    started_wall = dt.datetime.now(dt.timezone.utc)
    started = time.perf_counter()
    np.random.seed(seed)
    record = {
        "schema_version": 1,
        "accounted": True,
        "workflow": workflow,
        "parameter_index": parameter_index,
        "parameter_name": CONFIG[workflow]["parameter_name"],
        "parameter_value": task["parameter_value"],
        "repetition": repetition,
        "seed": seed,
        "base_seed": BASE_SEED,
        "commit": commit,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "started_utc": started_wall.isoformat(),
        "competitor_dependency": {"available": dependency_ok, "note": dependency_note},
    }

    with log_path.open("w") as log_file, contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
        print(json.dumps({key: record[key] for key in ("workflow", "parameter_index", "repetition", "seed", "commit")}, sort_keys=True))
        try:
            if workflow == "m2":
                metadata, methods = _m2_trial(parameter_index, task["parameter_value"], dependency_ok, dependency_note)
            else:
                metadata, methods = _m3_trial(parameter_index, task["parameter_value"], dependency_ok, dependency_note)
            record["configuration"] = metadata
            record["methods"] = methods
            record["task_status"] = "ACCOUNTED"
        except Exception as error:
            traceback.print_exc()
            record["configuration"] = None
            record["methods"] = [
                _method_result(method, "NOT_RUN_DATA_FAILURE", failure=f"{type(error).__name__}: {error}")
                for method in CONFIG[workflow]["methods"]
            ]
            record["task_status"] = "ACCOUNTED_DATA_FAILURE"

    ended_wall = dt.datetime.now(dt.timezone.utc)
    record["ended_utc"] = ended_wall.isoformat()
    record["runtime_seconds"] = float(time.perf_counter() - started)
    record["task_log"] = str(log_path.resolve())
    _atomic_json(record_path, record)
    return {
        "record": str(record_path),
        "workflow": workflow,
        "parameter_index": parameter_index,
        "repetition": repetition,
        "statuses": {item["method"]: item["status"] for item in record["methods"]},
        "runtime_seconds": record["runtime_seconds"],
    }


def _is_done(path: Path, retry_failed: bool) -> bool:
    if not path.exists():
        return False
    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if not record.get("accounted"):
        return False
    if retry_failed:
        statuses = {method.get("status") for method in record.get("methods", [])}
        return "FAILED" not in statuses and "NOT_RUN_DATA_FAILURE" not in statuses
    return True


def run_experiment(workflow: str, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"Run resumable WER-8 {workflow.upper()} repetitions")
    parser.add_argument("--repetitions", type=int, default=REPETITIONS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=REPO_ROOT / "rebuttal" / "WER-8-m2-m3-100" / "artifacts",
    )
    parser.add_argument(
        "--parameter-indices",
        help="comma-separated zero-based indices; default is the full historical grid",
    )
    parser.add_argument("--max-tasks", type=int, help="cheap validation only; do not use for the final run")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.repetitions <= REPETITIONS:
        parser.error(f"repetitions must be in 1,...,{REPETITIONS}")

    workers, available = effective_workers(args.workers)
    print(f"EFFECTIVE WORKERS: {workers} (requested={args.workers}, affinity={available}, cap=32)", flush=True)
    dependency_ok, dependency_note = competitor_dependency_status()
    print(f"COMPETITOR DEPENDENCIES: {'AVAILABLE' if dependency_ok else 'BLOCKED'} - {dependency_note}", flush=True)

    parameters = CONFIG[workflow]["parameters"]
    if args.parameter_indices:
        indices = [int(value) for value in args.parameter_indices.split(",")]
        if any(index < 0 or index >= len(parameters) for index in indices):
            parser.error("parameter index outside the historical grid")
    else:
        indices = list(range(len(parameters)))

    artifact_root = args.artifact_root.resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    lock_path = artifact_root / ".heavy_experiment.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(f"another heavy experiment may be active; inspect {lock_path}")

    commit = _git_commit()
    command = " ".join([sys.executable, *sys.argv])
    started = dt.datetime.now(dt.timezone.utc)
    tasks = [
        {
            "workflow": workflow,
            "parameter_index": index,
            "parameter_value": float(parameters[index]),
            "repetition": repetition,
        }
        for index in indices
        for repetition in range(args.repetitions)
    ]
    tasks = [
        task for task in tasks
        if not _is_done(
            _record_path(artifact_root, workflow, task["parameter_index"], task["repetition"]),
            args.retry_failed,
        )
    ]
    if args.max_tasks is not None:
        tasks = tasks[: args.max_tasks]

    os.write(
        descriptor,
        json.dumps({"pid": os.getpid(), "hostname": socket.gethostname(), "workflow": workflow, "started_utc": started.isoformat()}).encode(),
    )
    os.close(descriptor)
    print(f"PENDING TASKS: {len(tasks)}", flush=True)

    run_result = {
        "workflow": workflow,
        "commit": commit,
        "command": command,
        "settings": {
            "requested_repetitions": args.repetitions,
            "parameter_indices": indices,
            "requested_workers": args.workers,
            "effective_workers": workers,
            "available_cpu_affinity": available,
            "inner_threads": 1,
            "base_seed": BASE_SEED,
            "seed_policy": "SeedSequence([42, workflow code, parameter index, repetition])",
            "artifact_root": str(artifact_root),
        },
        "competitor_dependency": {"available": dependency_ok, "note": dependency_note},
        "started_utc": started.isoformat(),
        "task_count_started": len(tasks),
    }
    exit_code = 0
    try:
        summaries = Parallel(n_jobs=workers, backend="loky", verbose=5)(
            delayed(_run_task)(task, str(artifact_root), commit, dependency_ok, dependency_note)
            for task in tasks
        )
        run_result["task_summaries"] = summaries
    except BaseException as error:
        exit_code = 1
        run_result["driver_failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        run_result["ended_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        run_result["exit_code"] = exit_code
        runs_dir = artifact_root / "runs"
        run_name = f"{started.strftime('%Y%m%dT%H%M%SZ')}_{workflow}_{os.getpid()}.json"
        _atomic_json(runs_dir / run_name, run_result)
        lock_path.unlink(missing_ok=True)
    print(f"RUN RECORD: {(runs_dir / run_name).resolve()}", flush=True)
    return exit_code
