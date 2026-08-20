from types import SimpleNamespace

import numpy as np
import pytest
import scipy.optimize

import WERM.estimators
from WERM.experiment_utils import weighted_erm_joint_cv


def test_joint_cv_uses_fold_local_penalty_for_detection_and_segment_refits(
    monkeypatch,
):
    werm_calls = []
    segment_penalties = []

    def fake_weighted_erm(**kwargs):
        p = kwargs["X"].shape[1]
        werm_calls.append(
            {
                "n": len(kwargs["y"]),
                "L": kwargs["L"],
                "Delta_n": kwargs["Delta_n"],
                "gamma": kwargs["gamma"],
                "brute_force": kwargs["brute_force"],
                "penalty": float(kwargs["P_func"](np.ones(p))),
            }
        )
        return np.array([], dtype=int)

    def fake_minimize(fun, x0, **kwargs):
        segment_penalties.append(float(fun(np.ones_like(x0))))
        return SimpleNamespace(x=np.zeros_like(x0))

    monkeypatch.setattr(WERM.estimators, "weighted_erm", fake_weighted_erm)
    monkeypatch.setattr(scipy.optimize, "minimize", fake_minimize)

    X = np.ones((10, 4))
    y = np.arange(10.0)
    selected, cp = weighted_erm_joint_cv(
        X,
        y,
        M=object(),
        M_scorer=lambda prediction, truth: np.zeros_like(truth),
        Lmax=1,
        delta_fraction=0.25,
        l1_constants=(0.1, 1.0),
    )

    assert selected == (1, 0.1)
    assert cp.size == 0
    assert len(werm_calls) == 11
    assert all(call["gamma"] == 0.0 for call in werm_calls)
    assert all(call["brute_force"] is False for call in werm_calls)
    assert all(call["Delta_n"] == 2 for call in werm_calls)

    expected_fold_penalties = [
        4 * C * np.sqrt(2 * np.log(4) / 8)
        for C in (0.1, 1.0)
        for _ in range(5)
    ]
    assert [call["penalty"] for call in werm_calls[:-1]] == pytest.approx(
        expected_fold_penalties
    )
    assert segment_penalties == pytest.approx(expected_fold_penalties)
    expected_full_penalty = 4 * 0.1 * np.sqrt(2 * np.log(4) / 10)
    assert werm_calls[-1]["penalty"] == pytest.approx(expected_full_penalty)


@pytest.mark.parametrize(
    "constants", [(), (0.0,), (-1.0, 1.0), (np.nan,), (np.inf,)]
)
def test_joint_cv_rejects_invalid_l1_grid(constants):
    with pytest.raises(ValueError, match="l1_constants"):
        weighted_erm_joint_cv(
            np.ones((10, 2)), np.ones(10), object(), object(), 2, 0.1,
            l1_constants=constants,
        )


def test_joint_cv_preserves_interlaced_fold_and_changepoint_mapping(monkeypatch):
    segment_training_indices = []
    werm_call = 0

    def fake_weighted_erm(**kwargs):
        nonlocal werm_call
        werm_call += 1
        return np.array([1] if werm_call <= 2 else [3])

    def fake_minimize(fun, x0, **kwargs):
        # The first design column stores the original observation index.
        closure = {
            name: cell.cell_contents
            for name, cell in zip(fun.__code__.co_freevars, fun.__closure__)
        }
        objective_X = closure["X_segment"]
        segment_training_indices.append(np.asarray(objective_X)[:, 0].astype(int))
        return SimpleNamespace(x=np.zeros_like(x0))

    monkeypatch.setattr(WERM.estimators, "weighted_erm", fake_weighted_erm)
    monkeypatch.setattr(scipy.optimize, "minimize", fake_minimize)

    X = np.c_[np.arange(6), np.ones(6)]
    _, cp = weighted_erm_joint_cv(
        X,
        np.arange(6.0),
        M=object(),
        M_scorer=lambda prediction, truth: np.zeros_like(truth),
        Lmax=1,
        delta_fraction=0.2,
        l1_constants=(1.0,),
        num_folds=2,
    )

    assert [indices.tolist() for indices in segment_training_indices] == [
        [1], [3, 5], [0], [2, 4]
    ]
    np.testing.assert_array_equal(cp, [3])


def test_joint_cv_scales_train_test_and_full_responses_by_training_std(monkeypatch):
    werm_responses = []
    validation_truths = []

    def fake_weighted_erm(**kwargs):
        werm_responses.append(np.asarray(kwargs["y"]))
        return np.array([], dtype=int)

    def fake_minimize(fun, x0, **kwargs):
        return SimpleNamespace(x=np.zeros_like(x0))

    def scorer(prediction, truth):
        validation_truths.extend(np.asarray(truth).reshape(-1))
        return np.zeros_like(np.asarray(truth).reshape(-1))

    monkeypatch.setattr(WERM.estimators, "weighted_erm", fake_weighted_erm)
    monkeypatch.setattr(scipy.optimize, "minimize", fake_minimize)

    X = np.ones((6, 2))
    y = np.arange(6.0)
    weighted_erm_joint_cv(
        X, y, object(), scorer, 1, 0.2,
        l1_constants=(1.0,), num_folds=2, standardize_y=True,
    )

    odd, even = np.array([1, 3, 5]), np.array([0, 2, 4])
    np.testing.assert_allclose(werm_responses[0], y[odd] / np.std(y[odd]))
    np.testing.assert_allclose(werm_responses[1], y[even] / np.std(y[even]))
    np.testing.assert_allclose(werm_responses[2], y / np.std(y))
    np.testing.assert_allclose(
        validation_truths,
        np.r_[y[even] / np.std(y[odd]), y[odd] / np.std(y[even])],
    )


@pytest.mark.parametrize("guarded_std", [0.0, np.nan])
def test_joint_cv_response_scale_guard_uses_one(monkeypatch, guarded_std):
    responses = []
    monkeypatch.setattr(np, "std", lambda values: guarded_std)
    monkeypatch.setattr(
        WERM.estimators,
        "weighted_erm",
        lambda **kwargs: responses.append(np.asarray(kwargs["y"])) or np.array([]),
    )
    monkeypatch.setattr(
        scipy.optimize,
        "minimize",
        lambda fun, x0, **kwargs: SimpleNamespace(x=np.zeros_like(x0)),
    )
    y = np.arange(6.0)
    weighted_erm_joint_cv(
        np.ones((6, 2)), y, object(), lambda pred, truth: np.zeros(1),
        1, 0.2, l1_constants=(1.0,), num_folds=2, standardize_y=True,
    )
    np.testing.assert_array_equal(responses[-1], y)


def test_joint_cv_high_snr_recovers_exact_change():
    import jax.numpy as jnp

    rng = np.random.default_rng(7)
    n, p = 40, 5
    X = rng.normal(size=(n, p))
    beta = np.array([2.0, -1.0, 0.0, 0.0, 0.0])
    y = np.r_[X[:20] @ beta, X[20:] @ -beta]
    y += 0.05 * rng.normal(size=n)
    squared_loss = lambda pred, truth: jnp.square(pred - truth)

    selected, cp = weighted_erm_joint_cv(
        X,
        y,
        M=squared_loss,
        M_scorer=squared_loss,
        Lmax=2,
        delta_fraction=0.1,
        l1_constants=(0.1, 1.0),
        num_folds=2,
        n_jobs=1,
        standardize_y=True,
    )

    assert selected == (2, 1.0)
    np.testing.assert_array_equal(cp, [20])
