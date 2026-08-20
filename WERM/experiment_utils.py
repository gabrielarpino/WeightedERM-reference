import hashlib

import numpy as np


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
