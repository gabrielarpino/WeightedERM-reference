"""Independent checks of the first-order WERM-CV construction.

These validate its Gaussian calculations and index/normalization conventions;
they do not constitute a proof for adaptive segment refits or CV argmins.
"""
import itertools

import numpy as np
import pytest

from tools.el_state_evolution import squared_se
from tools.werm_cv_theory import expected_partition, uniform_refit_risk, predict_cv


def test_expected_loss_profile_matches_gaussian_surrogate_monte_carlo():
    rng = np.random.default_rng(808)
    n, signals, samples = 60, 3, 8000
    labels = np.repeat(np.arange(3), [12, 18, 30])
    gamma = .6 ** np.abs(np.arange(3)[:, None] - np.arange(3)[None, :])
    phi = rng.uniform(.1, 1., (signals, n))
    phi /= phi.sum(axis=0)
    result = expected_partition(phi, gamma, labels, 4., .25, 3)
    state = result['state']
    Z = rng.normal(size=(samples, n, 3)) @ np.linalg.cholesky(gamma).T
    G = rng.normal(size=(samples, n, signals)) @ state['covariance_root'].T
    y = Z[:, np.arange(n), labels] + rng.normal(0, .5, (samples, n))
    theta = ((1-state['omega']) * (Z @ state['lam'] + G)
             + state['omega'] * y[:, :, None])
    residuals = (theta-y[:, :, None])**2
    # Aggregate independently within each true regime, not across jumps.
    for regime in range(3):
        rows = labels == regime
        np.testing.assert_allclose(
            result['profile'][rows].mean(axis=0),
            residuals[:, rows].mean(axis=(0, 1)), rtol=.025, atol=.002)

    # An independent exhaustive search checks the profile-to-partition step.
    profile = result['profile']
    candidates = []
    for a, b in itertools.combinations(range(1, n), 2):
        if b-a >= 3:
            loss = profile[:a, 0].sum() + profile[a:b, 1].sum() + profile[b:, 2].sum()
            candidates.append((loss, a, b))
    _, a, b = min(candidates)
    np.testing.assert_array_equal(result['cp'], [a, b])


def test_uniform_refits_match_uniform_se_and_absolute_gaussian_risk():
    n, p = 240, 30
    labels = np.repeat(np.arange(3), 80)
    train = np.flatnonzero(np.arange(n) % 4 != 0)
    test = np.flatnonzero(np.arange(n) % 4 == 0)
    cp = np.array([90])  # Original location 121: first training row after split.
    gamma = np.array([[1., .3, .1], [.3, 1.5, .4], [.1, .4, 2.]])
    noise_variance = .25
    result = uniform_refit_risk(gamma, labels, train, test, cp, p, noise_variance)
    np.testing.assert_array_equal(result['segment_sizes'], [90, 90])
    np.testing.assert_allclose(result['segment_ratios'], [3., 3.])
    boundaries = np.r_[0, train[cp], n]
    expected_score = 0.
    rng = np.random.default_rng(77)
    for s, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        segment_train = train[(train >= left) & (train < right)]
        segment_test = test[(test >= left) & (test < right)]
        state = squared_se(np.ones((1, len(segment_train))), gamma,
                           labels[segment_train], len(segment_train)/p, noise_variance)
        q = np.bincount(labels[segment_train], minlength=3)/len(segment_train)
        fit = result['segment_risks'][s]
        np.testing.assert_allclose(fit['q'], q)
        np.testing.assert_allclose(state['lam'][:, 0], q)
        np.testing.assert_allclose(fit['kappa'], state['kappa'][0, 0])
        # Generate independent fresh predictions directly from Z and the
        # coefficient-noise limit; measure absolute loss, without using the
        # closed-form absolute-risk expression being tested.
        Z = rng.normal(size=(150000, 3)) @ np.linalg.cholesky(gamma).T
        predicted = Z @ q + rng.normal(0, np.sqrt(state['kappa'][0, 0]), len(Z))
        response = Z + rng.normal(0, np.sqrt(noise_variance), (len(Z), 1))
        observed_risk = np.abs(response-predicted[:, None]).mean(axis=0)
        np.testing.assert_allclose(fit['risk_by_regime'], observed_risk, rtol=.015)
        expected_score += np.asarray(fit['risk_by_regime'])[labels[segment_test]].sum()
    assert result['score'] == pytest.approx(expected_score)
    assert result['mean_score'] == pytest.approx(expected_score/len(test))


def test_uniform_homogeneous_closed_form_uses_segment_sampling_ratio():
    # At local delta=3, homogeneous OLS has predictive variance sigma²*3/2.
    # This catches accidentally using full-data delta or renormalizing Gamma.
    train = np.arange(90)
    test = np.arange(90, 120)
    result = uniform_refit_risk(np.array([[7.]]), np.zeros(120, int),
                               train, test, np.array([], int), 30, .4)
    expected = np.sqrt(2/np.pi)*np.sqrt(.4*3/2)
    assert result['mean_score'] == pytest.approx(expected)
    assert result['score'] == pytest.approx(30*expected)


@pytest.mark.parametrize('training_size', [29, 30])
def test_uniform_refit_rejects_unsupported_local_ratio(training_size):
    train = np.arange(training_size)
    test = np.arange(training_size, 60)
    with pytest.raises(ValueError):
        uniform_refit_risk(np.eye(1), np.zeros(60, int), train, test,
                           np.array([], int), 30, .25)


def test_cv_unequal_fold_normalization_and_homogeneous_count_selection():
    # n is not divisible by ten: fold-specific ratios and test sizes matter.
    n, p, sigma2 = 207, 10, .4
    weights = {}
    for m in (186, 187):
        t = np.linspace(.01, .99, m)
        weights[m, 1] = np.ones((1, m))
        weights[m, 2] = np.vstack((1-t, t))
    result = predict_cv(np.array([[5.]]), np.zeros(n, int), p, sigma2,
                        2, weights, .1, folds=10)
    expected = []
    for v in range(10):
        test_n = len(np.arange(v, n, 10))
        delta = (n-test_n)/p
        expected.append(test_n*np.sqrt(2/np.pi)*np.sqrt(sigma2*delta/(delta-1)))
    np.testing.assert_allclose(result['fold_scores'][0], expected)
    assert result['normalized_cv_scores'][0] == pytest.approx(sum(expected)/n)
    # Additional segments only inflate refit variance when there is no change.
    assert result['normalized_cv_scores'][1] > result['normalized_cv_scores'][0]
    assert result['selected_by_bound'][1]['selected_segments'] == 1
    assert result['selected_by_bound'][2]['selected_segments'] == 1


@pytest.mark.parametrize('n,count,gap', [(11, 3, 2), (10, 2, 3), (12, 4, 2), (8, 4, 2), (7, 1, 2)])
def test_uniform_minimum_length_marginals_match_exhaustive_configurations(n, count, gap):
    from tools.werm_cv_theory_experiment import uniform_marginals
    configurations = []
    for cp in itertools.combinations(range(1, n), count-1):
        if np.min(np.diff(np.r_[0, cp, n])) >= gap:
            configurations.append(np.searchsorted(cp, np.arange(n), side='right'))
    assert configurations
    oracle = np.zeros((count, n))
    for labels in configurations:
        oracle += np.eye(count)[labels].T/len(configurations)
    actual = uniform_marginals(n, count, gap)
    np.testing.assert_allclose(actual, oracle, atol=1e-13, rtol=1e-12)
    np.testing.assert_allclose(actual.sum(axis=0), 1.)
    np.testing.assert_allclose(actual.sum(axis=1), n/count)
    assert np.min(actual) >= -1e-13


def test_uniform_marginals_reject_infeasible_minimum_lengths():
    from tools.werm_cv_theory_experiment import uniform_marginals
    with pytest.raises(ValueError):
        uniform_marginals(5, 3, 2)
