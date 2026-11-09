"""Enumeration checks for the shared WERM/AMP integer-grid prior and MAP."""

from itertools import combinations
from math import log

import numpy as np
import pytest

from tools.werm_amp_partition import (
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
