"""Enumeration checks for the shared WERM/AMP integer-grid prior and MAP."""

from itertools import combinations
from math import log

import numpy as np
import pytest

from tools.werm_amp_partition import (
    amp_map_partition, amp_prior_marginals, amp_signal_label_subsets,
    log_partition_counts, map_partition, partition_minimize, prior_marginals,
)


def configurations(n, K, gap):
    return [cp for cp in combinations(range(1, n), K - 1)
            if np.all(np.diff(cp) >= gap)]


@pytest.mark.parametrize('n,L,gap', [(9, 4, 1), (12, 4, 3), (8, 3, 5), (1, 1, 2)])
def test_counts_and_marginals_match_exhaustive_prior(n, L, gap):
    counts = []
    mixed = np.zeros((L, n))
    for K in range(1, L + 1):
        configs = configurations(n, K, gap)
        counts.append(len(configs))
        conditional = np.zeros((K, n))
        for cp in configs:
            labels = np.searchsorted(cp, np.arange(n), side='right')
            conditional[labels, np.arange(n)] += 1. / len(configs)
        np.testing.assert_allclose(prior_marginals(n, K, gap, True), conditional,
                                   atol=1e-14)
        mixed[:K] += conditional / L
    np.testing.assert_allclose(log_partition_counts(n, L, gap), np.log(counts))
    np.testing.assert_allclose(prior_marginals(n, L, gap), mixed, atol=1e-14)
    np.testing.assert_allclose(mixed.sum(axis=0), 1.)


@pytest.mark.parametrize('fixed', [False, True])
@pytest.mark.parametrize('n,L,gap', [(13, 4, 2), (8, 3, 4), (9, 1, 3)])
def test_map_matches_exhaustive_joint_posterior(fixed, n, L, gap):
    rng = np.random.default_rng(918 + n + L)
    log_likelihood = rng.normal(size=(n, L))
    actual = map_partition(log_likelihood, gap, fixed)
    candidates = []
    for K in ((L,) if fixed else range(1, L + 1)):
        configs = configurations(n, K, gap)
        for cp in configs:
            labels = np.searchsorted(cp, np.arange(n), side='right')
            objective = -log_likelihood[np.arange(n), labels].sum() + log(len(configs))
            objective += 0. if fixed else log(L)
            candidates.append((float(objective), cp, K))
    expected = min(candidates, key=lambda x: x[0])
    np.testing.assert_allclose(actual['objective'], expected[0])
    np.testing.assert_array_equal(actual['cp'], expected[1])
    assert actual['segments'] == expected[2]
    assert actual['count'] == len(actual['cp'])


def test_shared_support_keeps_short_endpoints_and_exact_gap():
    # The source AMP enumerator excludes these endpoint configurations;
    # this comparison deliberately uses the canonical WERM search support.
    n, K, gap = 8, 4, 3
    cp = partition_minimize(np.zeros((n, K)), gap)
    np.testing.assert_array_equal(cp, [1, 4, 7])
    assert np.exp(log_partition_counts(n, K, gap)[-1]) == 1.
    expected = np.eye(K)[np.searchsorted(cp, np.arange(n), side='right')].T
    np.testing.assert_array_equal(prior_marginals(n, K, gap, True), expected)


def test_large_problem_is_normalized_without_enumeration():
    phi = prior_marginals(2300, 4, 58)
    assert phi.shape == (4, 2300)
    assert np.isfinite(phi).all() and np.min(phi) >= 0.
    np.testing.assert_allclose(phi.sum(axis=0), 1., atol=2e-14)
    np.testing.assert_allclose(phi[:, 0], [1., 0., 0., 0.])
    np.testing.assert_allclose(phi[:, -1], np.ones(4) / 4.)


def test_infeasible_count_and_invalid_inputs_are_rejected():
    assert np.isneginf(log_partition_counts(7, 4, 3)[-1])
    with pytest.raises(ValueError, match='Infeasible'):
        prior_marginals(7, 4, 3)
    with pytest.raises(ValueError, match='Infeasible'):
        map_partition(np.zeros((7, 4)), 3)
    with pytest.raises(ValueError):
        partition_minimize(np.zeros((12, 4)), 1.5)
    with pytest.raises(ValueError):
        map_partition(np.array([[np.nan]]), 1)


@pytest.mark.parametrize('n,L,gap', [(9, 4, 1), (12, 4, 3), (8, 3, 5), (1, 1, 2)])
def test_amp_marginals_and_map_match_one_joint_enumerated_prior(n, L, gap):
    rng = np.random.default_rng(451 + n)
    ll = rng.normal(size=(n, L))
    phi = np.zeros((L, n))
    count_probabilities = np.zeros(L)
    candidates = []
    for K in range(1, L + 1):
        configs = configurations(n, K, gap)
        # Independently spell out the source MAP support, not the production
        # helper, so this test catches missing constant or skipped labels.
        label_sets = ([(j,) for j in range(L)] if K == 1 else
                      [(0, *tail) for tail in combinations(range(1, L), K - 1)])
        assert amp_signal_label_subsets(L, K) == label_sets
        probability = 1 / (L * len(configs) * len(label_sets))
        for cp in configs:
            for labels in label_sets:
                assigned = np.asarray(labels)[np.searchsorted(cp, np.arange(n), side='right')]
                phi[assigned, np.arange(n)] += probability
                count_probabilities[K - 1] += probability
                objective = -float(ll[np.arange(n), assigned].sum()) - log(probability)
                candidates.append((objective, cp, labels, K))
    np.testing.assert_allclose(count_probabilities, np.ones(L) / L, atol=1e-14)
    np.testing.assert_allclose(amp_prior_marginals(n, L, gap), phi, atol=1e-14)
    np.testing.assert_allclose(phi.sum(axis=0), 1., atol=1e-14)
    best = min(candidates, key=lambda c: c[0])
    actual = amp_map_partition(ll, gap)
    np.testing.assert_allclose(actual['objective'], best[0], atol=1e-12)
    np.testing.assert_array_equal(actual['cp'], best[1])
    assert actual['selected_signal_labels'] == best[2]
    assert actual['segments'] == best[3]


def test_amp_no_change_prior_mass_is_balanced_and_search_can_use_last_signal():
    L, n, gap = 4, 30, 3
    phi = amp_prior_marginals(n, L, gap)
    # At observation 0 all multi-segment configurations use signal 0;
    # no-change probability 1/L is shared equally among all L signals.
    np.testing.assert_allclose(phi[:, 0], [13/16, 1/16, 1/16, 1/16])
    assert np.all(phi >= 1 / L**2 - 1e-14)
    ll = np.full((n, L), -100.)
    ll[:, 3] = 0.
    selected = amp_map_partition(ll, gap)
    assert selected['count'] == 0
    assert selected['selected_signal_labels'] == (3,)
    np.testing.assert_allclose(selected['log_prior'], -2 * log(L))


def test_amp_can_skip_an_unused_signal_without_adding_a_false_change():
    n, L, gap = 30, 4, 3
    truth, labels = [8, 19], (0, 1, 3)
    assigned = np.asarray(labels)[np.searchsorted(truth, np.arange(n), side='right')]
    ll = np.full((n, L), -20.)
    ll[np.arange(n), assigned] = 0.
    selected = amp_map_partition(ll, gap)
    assert selected['count'] == 2
    assert selected['selected_signal_labels'] == labels
    np.testing.assert_array_equal(selected['cp'], truth)


def test_known_count_amp_weights_and_search_are_exactly_unchanged():
    rng = np.random.default_rng(891)
    n, L, gap = 71, 4, 7
    np.testing.assert_array_equal(amp_prior_marginals(n,L,gap,True),
                                   prior_marginals(n,L,gap,True))
    ll = rng.normal(size=(n,L))
    before, after = map_partition(ll,gap,True), amp_map_partition(ll,gap,True)
    np.testing.assert_array_equal(before['cp'], after['cp'])
    np.testing.assert_array_equal(before['candidate_objectives'], after['candidate_objectives'])
    assert before['objective'] == after['objective']
    assert after['selected_signal_labels'] == (0,1,2,3)
