"""Focused correctness checks for the Phase 1 exact-DP candidate."""

import contextlib
import io
import itertools
import math

import jax.numpy as jnp
import numpy as np
import pytest

from WERM.gof import (
    chgpt_binary_segmentation,
    chgpt_gof_bf_mem_eff,
    chgpt_gof_dp,
    legacy_gamma_k_penalty,
    manuscript_log_binomial_k_penalty,
    table_k_penalty,
    zero_k_penalty,
)


TOLERANCE = 1e-9
SQUARED_LOSS = lambda prediction, outcome: jnp.square(prediction - outcome)


def raw_objective(Z, y, changepoints):
    boundaries = [0, *np.asarray(changepoints, dtype=int).tolist(), len(y)]
    return float(
        sum(
            np.square(Z[boundaries[column] : boundaries[column + 1], column]
                      - y[boundaries[column] : boundaries[column + 1]]).sum()
            for column in range(len(boundaries) - 1)
        )
    )


def feasible_partitions(n, K, Delta):
    if K == 1:
        yield np.array([], dtype=int)
        return
    for candidate in itertools.combinations(range(1, n), K - 1):
        if all(right - left >= Delta for left, right in zip(candidate, candidate[1:])):
            yield np.asarray(candidate, dtype=int)


def exhaustive_fixed_K(Z, y, K, Delta):
    scored = [(raw_objective(Z, y, cp), cp) for cp in feasible_partitions(len(y), K, Delta)]
    if not scored:
        raise ValueError("infeasible exhaustive case")
    minimum = min(score for score, _ in scored)
    optima = [cp for score, cp in scored if np.isclose(score, minimum, atol=1e-12, rtol=0)]
    return minimum, optima


def exhaustive_upper_bound(Z, y, L, Delta, penalty):
    candidates = []
    for K in range(1, L + 1):
        try:
            raw, optima = exhaustive_fixed_K(Z, y, K, Delta)
        except ValueError:
            continue
        candidates.append((raw + penalty(K), K, raw, optima))
    best_total = min(value[0] for value in candidates)
    best = [value for value in candidates if np.isclose(value[0], best_total, atol=1e-12, rtol=0)]
    return best_total, best


def deterministic_case(boundaries, n):
    K = len(boundaries) + 1
    Z = np.full((n, K), 3.0)
    y = np.zeros(n)
    endpoints = [0, *boundaries, n]
    for column in range(K):
        Z[endpoints[column] : endpoints[column + 1], column] = 0.0
    return Z, y


@pytest.mark.parametrize(
    "n,K,Delta,boundaries",
    [
        (7, 1, 1, ()),
        (9, 2, 1, (4,)),
        (12, 4, 3, (2, 6, 10)),
    ],
)
def test_three_hand_constructed_examples(n, K, Delta, boundaries):
    Z, y = deterministic_case(boundaries, n)
    result = chgpt_gof_dp(SQUARED_LOSS, Z, y, n, K, Delta)
    brute_raw, optima = exhaustive_fixed_K(Z, y, K, Delta)

    assert result["K"] == K
    assert np.isclose(result["raw_objective"], brute_raw, atol=TOLERANCE, rtol=0)
    assert np.isclose(raw_objective(Z, y, result["changepoints"]), brute_raw,
                      atol=TOLERANCE, rtol=0)
    assert any(np.array_equal(result["changepoints"], cp) for cp in optima)
    if boundaries:
        assert np.array_equal(result["changepoints"], np.asarray(boundaries))


@pytest.mark.parametrize("seed", range(20))
def test_seeded_random_examples_match_exhaustive_brute_force(seed):
    rng = np.random.default_rng(1000 + seed)
    K = 1 + seed % 4
    Delta = (1, 2, 3)[seed % 3]
    n = 8 + seed % 4
    Z = rng.normal(size=(n, K))
    y = rng.normal(size=n)

    result = chgpt_gof_dp(SQUARED_LOSS, Z, y, n, K, Delta)
    brute_raw, optima = exhaustive_fixed_K(Z, y, K, Delta)
    recomputed = raw_objective(Z, y, result["changepoints"])

    # A DP value on either side of brute force is an objective/feasible-set failure.
    assert np.isclose(result["raw_objective"], brute_raw, atol=TOLERANCE, rtol=0)
    assert np.isclose(recomputed, brute_raw, atol=TOLERANCE, rtol=0)
    if len(optima) == 1:
        assert np.array_equal(result["changepoints"], optima[0])


def test_manuscript_penalty_uses_K_minus_one_changepoints():
    n = 13
    weight = 0.7
    penalty = manuscript_log_binomial_k_penalty(n, weight)
    assert penalty(1) == 0.0
    assert np.isclose(penalty(2), weight * np.log(n - 1), atol=1e-14, rtol=0)


def test_legacy_penalty_matches_live_formula():
    n, Lmax, Delta, gamma = 17, 5, 3, 0.4
    penalty = legacy_gamma_k_penalty(n, Lmax, Delta, gamma)
    assert penalty(1) == 0.0
    for K in range(2, Lmax + 1):
        effective = (n - 1) - (K - 2) * (Delta - 1)
        expected = gamma * np.log(Lmax * math.comb(effective, K - 1))
        assert np.isclose(penalty(K), expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("penalty_id", ["zero", "manuscript", "legacy", "table"])
def test_upper_bound_selection_with_all_K_only_penalties(penalty_id):
    rng = np.random.default_rng(44)
    n, L, Delta = 10, 4, 2
    Z = rng.normal(size=(n, L))
    y = rng.normal(size=n)
    penalties = {
        "zero": zero_k_penalty,
        "manuscript": manuscript_log_binomial_k_penalty(n, 0.25),
        "legacy": legacy_gamma_k_penalty(n, L, Delta, 0.2),
        "table": table_k_penalty({1: 0.0, 2: 0.3, 3: 0.8, 4: 1.5}),
    }
    penalty = penalties[penalty_id]

    result = chgpt_gof_dp(
        SQUARED_LOSS, Z, y, n, L, Delta,
        L_is_upper_bound=True, penalty=penalty,
    )
    brute_total, best = exhaustive_upper_bound(Z, y, L, Delta, penalty)
    brute_K = {candidate[1] for candidate in best}

    assert np.isclose(result["total_objective"], brute_total, atol=TOLERANCE, rtol=0)
    assert np.isclose(
        raw_objective(Z, y, result["changepoints"]) + penalty(result["K"]),
        brute_total, atol=TOLERANCE, rtol=0,
    )
    assert result["K"] in brute_K
    if len(best) == 1 and len(best[0][3]) == 1:
        assert result["K"] == best[0][1]
        assert np.array_equal(result["changepoints"], best[0][3][0])


def test_fixed_K_penalty_cannot_change_raw_optimum():
    rng = np.random.default_rng(91)
    n, K, Delta = 11, 3, 2
    Z = rng.normal(size=(n, K))
    y = rng.normal(size=n)
    unpenalized = chgpt_gof_dp(SQUARED_LOSS, Z, y, n, K, Delta)
    penalized = chgpt_gof_dp(
        SQUARED_LOSS, Z, y, n, K, Delta,
        penalty=lambda selected_K: 17.0 + selected_K,
    )
    assert np.array_equal(unpenalized["changepoints"], penalized["changepoints"])
    assert unpenalized["raw_objective"] == penalized["raw_objective"]


@pytest.mark.parametrize("K,Delta", [(2, 1), (3, 1), (4, 3)])
def test_matches_live_brute_force_oracle(K, Delta):
    rng = np.random.default_rng(700 + K + Delta)
    n = 13
    Z = rng.normal(size=(n, K))
    y = rng.normal(size=n)
    result = chgpt_gof_dp(SQUARED_LOSS, Z, y, n, K, Delta)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        brute_cp, brute_total = chgpt_gof_bf_mem_eff(
            SQUARED_LOSS, Z, y, n, K, Delta,
            γ=0.0, gof_opt_output=True,
        )
    brute_total = float(brute_total)
    assert np.isclose(result["total_objective"], brute_total, atol=TOLERANCE, rtol=0)
    assert np.isclose(raw_objective(Z, y, brute_cp), brute_total,
                      atol=TOLERANCE, rtol=0)
    _, optima = exhaustive_fixed_K(Z, y, K, Delta)
    if len(optima) == 1:
        assert np.array_equal(result["changepoints"], brute_cp)


def test_legacy_upper_bound_selection_matches_live_brute_force():
    rng = np.random.default_rng(777)
    n, L, Delta, gamma = 10, 4, 2, 0.15
    Z = rng.normal(size=(n, L))
    y = rng.normal(size=n)
    penalty = legacy_gamma_k_penalty(n, L, Delta, gamma)
    result = chgpt_gof_dp(
        SQUARED_LOSS, Z, y, n, L, Delta,
        L_is_upper_bound=True, penalty=penalty,
    )
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        brute_cp, brute_total = chgpt_gof_bf_mem_eff(
            SQUARED_LOSS, Z, y, n, L, Delta, γ=gamma,
            L_is_upper_bound=True, gof_opt_output=True,
        )
    brute_total = float(brute_total)
    expected_total, expected_optima = exhaustive_upper_bound(Z, y, L, Delta, penalty)
    assert len(expected_optima) == 1
    selected_K = expected_optima[0][1]
    expected_cp = expected_optima[0][3]
    assert len(expected_cp) == 1
    assert np.isclose(result["total_objective"], brute_total, atol=TOLERANCE, rtol=0)
    assert np.isclose(result["total_objective"], expected_total, atol=TOLERANCE, rtol=0)
    assert result["K"] == len(brute_cp) + 1 == selected_K
    assert np.array_equal(result["changepoints"], expected_cp[0])
    assert np.array_equal(result["changepoints"], brute_cp)


@pytest.mark.parametrize("seed", range(6))
def test_dp_is_not_worse_than_independently_rescored_greedy(seed):
    rng = np.random.default_rng(800 + seed)
    n, K, Delta = 12, 2 + seed % 3, 1 + seed % 2
    Z = rng.normal(size=(n, K))
    y = rng.normal(size=n)
    result = chgpt_gof_dp(SQUARED_LOSS, Z, y, n, K, Delta)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        greedy_cp = chgpt_binary_segmentation(
            SQUARED_LOSS, Z, y, n, K, Delta, γ=0.0,
        )
    assert len(greedy_cp) == K - 1
    greedy_raw = raw_objective(Z, y, greedy_cp)
    assert result["raw_objective"] <= greedy_raw + TOLERANCE


@pytest.mark.parametrize(
    "Z,y,n,L,Delta",
    [
        (np.zeros((5, 2)), np.zeros(4), 5, 2, 1),
        (np.zeros((5, 1)), np.zeros(5), 5, 2, 1),
        (np.zeros((5, 3)), np.zeros(5), 5, 3, 5),
    ],
)
def test_invalid_dimensions_and_infeasible_partitions(Z, y, n, L, Delta):
    with pytest.raises(ValueError):
        chgpt_gof_dp(SQUARED_LOSS, Z, y, n, L, Delta)


def test_returned_changepoints_are_ordered_and_in_range():
    rng = np.random.default_rng(123)
    n, L, Delta = 15, 4, 3
    Z = rng.normal(size=(n, L))
    y = rng.normal(size=n)
    result = chgpt_gof_dp(SQUARED_LOSS, Z, y, n, L, Delta)
    changepoints = result["changepoints"]
    assert np.all((changepoints >= 1) & (changepoints <= n - 1))
    assert np.all(np.diff(changepoints) >= Delta)


def test_upper_bound_skips_infeasible_K_and_accepts_direct_table():
    rng = np.random.default_rng(321)
    n, L, Delta = 5, 4, 3
    Z = rng.normal(size=(n, L))
    y = rng.normal(size=n)
    result = chgpt_gof_dp(
        SQUARED_LOSS, Z, y, n, L, Delta,
        L_is_upper_bound=True, penalty={1: 0.0, 2: 0.1, 3: 0.2},
    )
    assert result["K"] <= 3
    assert np.isinf(result["raw_by_K"][4])
    assert np.isnan(result["penalty_by_K"][4])
    assert np.isinf(result["total_by_K"][4])


def test_partition_penalty_must_be_finite_and_K_only():
    Z, y = deterministic_case((3,), 7)
    calls = []

    def penalty(K):
        calls.append(K)
        return 0.0

    chgpt_gof_dp(SQUARED_LOSS, Z, y, 7, 2, 1,
                 L_is_upper_bound=True, penalty=penalty)
    assert calls == [1, 2]
    with pytest.raises(ValueError):
        chgpt_gof_dp(SQUARED_LOSS, Z, y, 7, 2, 1,
                     penalty=lambda K: np.inf)
