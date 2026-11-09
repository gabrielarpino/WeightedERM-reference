"""Numerical checks of AMP integration and posterior conventions."""
import numpy as np
from scipy.stats import multivariate_normal

from tools.werm_amp_adapter import (
    _normalized, fit_amp, gaussian_row_loglik, load_amp_source, output_state_mc,
)
from tools.werm_amp_partition import prior_marginals


def test_initial_output_state_matches_analytic_gaussian_expectation():
    source = load_amp_source()
    n, L = 80, 3
    phi = prior_marginals(n, L, 4)
    rho = np.eye(L)
    actual = output_state_mc(source, phi, rho, np.sqrt(.1), np.zeros((L, L)),
                             rho, np.random.default_rng(33), mc_samples=32768)
    expected = (phi @ phi.T) / (n * 1.1)
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < .04


def test_row_likelihood_has_identical_map_scores_to_source_joint_gaussian():
    source = load_amp_source()
    rng = np.random.default_rng(45)
    L = 4
    rho = np.eye(L)
    nu = .4 * np.eye(L)
    kappa = .24 * np.eye(L)
    theta, y = rng.normal(size=(15, L)), rng.normal(size=15)
    actual = gaussian_row_loglik(theta, y, rho, np.sqrt(.1), nu, kappa)
    joint = []
    for label in range(L):
        cov = np.asarray(getattr(source, _normalized("form_Σ"))(
            label, np.sqrt(.1), rho, nu, kappa))
        joint.append(multivariate_normal.logpdf(np.c_[theta, y],
                     cov=cov + 1e-10 * np.eye(L + 1)))
    joint = np.stack(joint, axis=1)
    np.testing.assert_allclose(actual - actual[:, [0]], joint - joint[:, [0]], atol=1e-11)


def test_final_pseudodata_state_alignment_and_unmodified_design():
    # At t=0 theta=X B0 and its state is exactly nu=0,kappa=rho. Thus
    # posterior label likelihoods are equal for iid prior variances. Using
    # the *updated* state would instead introduce spurious label evidence.
    rng = np.random.default_rng(2)
    n, p, L, gap, seed = 36, 12, 3, 3, 9
    delta = n / p
    X = rng.normal(size=(n, p)) / np.sqrt(n)
    X[:, 1] += .7 * X[:, 0]  # deliberately correlated; no whitening allowed
    original = X.copy()
    y = rng.normal(size=n)
    phi = prior_marginals(n, L, gap)
    result = fit_amp(X, y, delta, np.sqrt(.1), delta * np.eye(L), phi,
                     gap, seed, iterations=1, mc_samples=1024, return_details=True)
    assert result["status"] == "ok"
    init_seed = np.random.SeedSequence(seed).spawn(2)[0]
    B0 = np.random.default_rng(init_seed).standard_normal((p, L)) * np.sqrt(delta)
    np.testing.assert_array_equal(X, original)
    np.testing.assert_allclose(result["theta"], original @ B0)
    np.testing.assert_array_equal(result["posterior_nu"], np.zeros((L, L)))
    np.testing.assert_array_equal(result["posterior_kappa"], np.eye(L))
    np.testing.assert_allclose(result["row_loglik"],
                               np.repeat(result["row_loglik"][:, [0]], L, axis=1))
    assert result["count"] == 0  # no evidence, uniform count/prior MAP prefers K=1


def test_known_count_and_trial_reproducibility():
    rng = np.random.default_rng(12)
    n, p, L, gap = 50, 20, 3, 4
    delta = n / p
    X = rng.normal(size=(n, p)) / np.sqrt(n)
    y = rng.normal(size=n)
    phi = prior_marginals(n, L, gap, fixed_count=True)
    args = (X, y, delta, np.sqrt(.1), delta * np.eye(L), phi, gap, 188)
    first = fit_amp(*args, iterations=2, mc_samples=1024, fixed_count=True)
    second = fit_amp(*args, iterations=2, mc_samples=1024, fixed_count=True)
    assert first["status"] == second["status"] == "ok"
    assert first["count"] == second["count"] == L - 1
    np.testing.assert_array_equal(first["cp"], second["cp"])
    np.testing.assert_allclose(first["candidate_objectives"], second["candidate_objectives"])
