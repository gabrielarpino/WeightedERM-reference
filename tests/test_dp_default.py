"""Narrow regression checks for the accepted exact-DP production route."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

import WERM.estimators as estimators
import WERM.gof as gof


SQUARED_LOSS = lambda prediction, outcome: jnp.square(prediction - outcome)


def test_production_adapter_matches_exact_dp_with_legacy_penalty():
    rng = np.random.default_rng(71)
    n, L, Delta, gamma = 11, 4, 2, 0.2
    predictions = rng.normal(size=(n, L))
    outcome = rng.normal(size=n)
    penalty = gof.legacy_gamma_k_penalty(n, L, Delta, gamma)
    expected = gof.chgpt_gof_dp(
        SQUARED_LOSS,
        predictions,
        outcome,
        n,
        L,
        Delta,
        L_is_upper_bound=True,
        penalty=penalty,
    )

    changepoints, objective = estimators._exact_partition_search(
        SQUARED_LOSS,
        predictions,
        outcome,
        n,
        L,
        Delta,
        gamma,
        True,
        False,
        None,
    )

    assert np.array_equal(changepoints, expected["changepoints"])
    assert objective == expected["total_objective"]


def test_legacy_brute_force_flag_keeps_exact_dp_route(monkeypatch):
    predictions = np.column_stack(
        [np.r_[np.zeros(4), np.ones(4)], np.r_[np.ones(4), np.zeros(4)]]
    )
    outcome = np.zeros(8)

    def forbidden(*args, **kwargs):
        raise AssertionError("production routing called an audit oracle")

    monkeypatch.setattr(gof, "chgpt_gof_bf_mem_eff", forbidden)
    monkeypatch.setattr(gof, "chgpt_binary_segmentation", forbidden)
    with pytest.warns(FutureWarning, match="now use exact DP"):
        changepoints, objective = estimators._exact_partition_search(
            SQUARED_LOSS,
            predictions,
            outcome,
            8,
            2,
            1,
            0.0,
            False,
            True,
            None,
        )

    assert changepoints.shape == (1,)
    assert np.isfinite(objective)


def test_custom_penalty_is_k_only_and_unambiguous():
    rng = np.random.default_rng(72)
    n, L = 9, 3
    predictions = rng.normal(size=(n, L))
    outcome = rng.normal(size=n)
    penalty = {1: 10.0, 2: 2.0, 3: 0.0}

    changepoints, objective = estimators._exact_partition_search(
        SQUARED_LOSS,
        predictions,
        outcome,
        n,
        L,
        1,
        0.0,
        True,
        False,
        penalty,
    )
    expected = gof.chgpt_gof_dp(
        SQUARED_LOSS,
        predictions,
        outcome,
        n,
        L,
        1,
        L_is_upper_bound=True,
        penalty=penalty,
    )
    assert np.array_equal(changepoints, expected["changepoints"])
    assert objective == expected["total_objective"]

    with pytest.raises(ValueError, match="alternative K-only penalties"):
        estimators._exact_partition_search(
            SQUARED_LOSS,
            predictions,
            outcome,
            n,
            L,
            1,
            0.1,
            True,
            False,
            penalty,
        )
    with pytest.raises(TypeError, match=r"penalty\(K\)"):
        gof.chgpt_gof_dp(
            SQUARED_LOSS,
            predictions,
            outcome,
            n,
            L,
            1,
            penalty=lambda K, changepoints: 0.0,
        )


def test_one_segment_production_objective_is_finite():
    predictions = np.ones((6, 1))
    outcome = np.zeros(6)
    changepoints, objective = estimators._exact_partition_search(
        SQUARED_LOSS,
        predictions,
        outcome,
        6,
        1,
        1,
        0.0,
        False,
        False,
        None,
    )
    assert changepoints.size == 0
    assert objective == 6.0


def test_weighted_erm_forwards_to_exact_adapter(monkeypatch):
    captured = {}

    def fake_minimize(*args, **kwargs):
        return SimpleNamespace(x=np.array([0.0, 0.0]))

    def fake_search(*args):
        captured["args"] = args
        return np.array([3]), 12.5

    monkeypatch.setattr(estimators.scipy.optimize, "minimize", fake_minimize)
    monkeypatch.setattr(estimators, "_exact_partition_search", fake_search)
    custom_penalty = {1: 0.0, 2: 1.0}
    estimate, objective = estimators.weighted_erm(
        X=np.ones((6, 2)),
        y=np.zeros(6),
        M=SQUARED_LOSS,
        L=2,
        Delta_n=1,
        gamma=0.0,
        L_is_upper_bound=True,
        brute_force=True,
        return_gof=True,
        partition_penalty=custom_penalty,
    )

    assert np.array_equal(estimate, np.array([3]))
    assert objective == 12.5
    assert captured["args"][1].shape == (6, 2)
    assert captured["args"][7] is True
    assert captured["args"][8] is True
    assert captured["args"][9] is custom_penalty


def test_hvi_forwards_to_exact_adapter(monkeypatch):
    captured = {}

    def fake_search(*args):
        captured["args"] = args
        return np.array([2]), 4.0

    monkeypatch.setattr(estimators, "_exact_partition_search", fake_search)
    custom_penalty = {1: 0.0, 2: 0.5}
    estimate, objective = estimators.HVI(
        X=np.ones((5, 2)),
        y=np.zeros(5),
        M=SQUARED_LOSS,
        L=2,
        pi=np.full((2, 5), 0.5),
        Delta_n=1,
        gamma=0.0,
        L_is_upper_bound=False,
        brute_force=False,
        return_gof=True,
        num_EM_iters=0,
        partition_penalty=custom_penalty,
    )

    assert np.array_equal(estimate, np.array([2]))
    assert objective == 4.0
    assert captured["args"][1].shape == (5, 2)
    assert captured["args"][9] is custom_penalty
