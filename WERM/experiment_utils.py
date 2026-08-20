import hashlib

import numpy as np

from WERM.performance_measures import hausdorff


def trial_rng(experiment, setting_idx, trial_idx):
    """Return a stable NumPy generator keyed by experiment task identity."""
    key = f"{experiment}|{int(setting_idx)}|{int(trial_idx)}".encode()
    seed = int.from_bytes(
        hashlib.blake2b(key, digest_size=8).digest(), "big"
    ) % (2**32)
    return np.random.default_rng(seed)


def data_hash(X, y):
    """Hash the exact contiguous design/response bytes for one trial."""
    X_bytes = np.ascontiguousarray(X).tobytes()
    y_bytes = np.ascontiguousarray(y).tobytes()
    return hashlib.blake2b(X_bytes + y_bytes, digest_size=16).hexdigest()


def evaluate_changepoints(cp, true_chgpts, n):
    """Apply the common comparison metric and three allowed result statuses."""
    if cp is None:
        return {
            "raw_cp": None,
            "status": "not_applicable",
            "haus": np.nan,
            "size": np.nan,
        }

    raw_cp = np.asarray(cp, dtype=int).reshape(-1)
    if raw_cp.size == 0:
        return {
            "raw_cp": raw_cp,
            "status": "empty_estimate",
            "haus": 1.0,
            "size": 0,
        }

    return {
        "raw_cp": raw_cp,
        "status": "ok",
        "haus": float(hausdorff(raw_cp, true_chgpts)) / int(n),
        "size": int(raw_cp.size),
    }


def summarize_setting(haus, sizes, statuses):
    """Summarize one method/setting with empty estimates penalized in haus."""
    haus = np.asarray(haus, dtype=float)
    sizes = np.asarray(sizes, dtype=float)
    statuses = np.asarray(statuses, dtype=object)
    if not set(statuses.tolist()) <= {"ok", "empty_estimate", "not_applicable"}:
        raise ValueError("unexpected experiment status")
    return {
        "mean_haus": float(np.nanmean(haus)),
        "median_haus": float(np.nanmedian(haus)),
        "empty_estimate_rate": float(np.mean(statuses == "empty_estimate")),
        "not_applicable_rate": float(np.mean(statuses == "not_applicable")),
        "mean_size": float(np.nanmean(sizes)),
    }
