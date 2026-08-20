import numpy as np
from joblib import Parallel, delayed

from WERM.experiment_utils import data_hash, trial_rng
from WERM.signal_priors import SparseDiffSignal, SparseGaussianSignal


def _manifest_hash(task):
    experiment, setting_idx, trial_idx = task
    rng = trial_rng(experiment, setting_idx, trial_idx)
    X = rng.normal(size=(24, 7))
    y = rng.standard_t(4, size=24)
    return data_hash(X, y)


def test_serial_and_parallel_manifests_have_identical_hashes():
    manifest = [
        ("M1b-alpha-0.3", setting_idx, trial_idx)
        for setting_idx in range(3)
        for trial_idx in range(5)
    ]
    serial = [_manifest_hash(task) for task in manifest]
    parallel = Parallel(n_jobs=2)(
        delayed(_manifest_hash)(task) for task in manifest
    )
    assert parallel == serial
    assert [_manifest_hash(task) for task in manifest] == serial


def test_sparse_gaussian_prior_accepts_a_trial_generator():
    prior = SparseGaussianSignal(
        α=0.3, δ=2.0, σ_l_arr=np.array([1.0, 2.0, 3.0])
    )
    left = prior.sample(20, rng=trial_rng("prior", 1, 2))
    right = prior.sample(20, rng=trial_rng("prior", 1, 2))
    np.testing.assert_array_equal(left, right)


def test_sparse_difference_prior_accepts_a_trial_generator():
    prior = SparseDiffSignal(var_1=2.0, σ_w=0.5, α=0.3, L=3)
    left = prior.sample(20, rng=trial_rng("prior", 2, 3))
    right = prior.sample(20, rng=trial_rng("prior", 2, 3))
    np.testing.assert_array_equal(left, right)
