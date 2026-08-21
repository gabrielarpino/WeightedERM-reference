import numpy as np
import pytest

import WERM.comparison as comparison
from WERM.experiment_utils import glm_bsa3_cv


def test_glm_bsa3_segment_beta_uses_exact_package_lambda(monkeypatch):
    seen = {}

    def fake_coef(X, y, lam):
        seen["shape"] = np.asarray(X).shape
        seen["lambda"] = float(lam)
        return np.arange(np.asarray(X).shape[1], dtype=float)

    monkeypatch.setattr(comparison, "_glm_bsa3_coef_r", fake_coef)
    X = np.ones((20, 4))
    beta = comparison.glm_bsa3_segment_beta(X, np.tile([0, 1], 10), 0.18)

    expected = 0.18 * (
        np.sqrt(2 * np.log(8) / 20) + np.log(8) / 20
    )
    assert seen["shape"] == (20, 4)
    assert np.isclose(seen["lambda"], expected)
    assert np.array_equal(beta, np.arange(4, dtype=float))


def test_glm_bsa3_cv_is_fold_local_and_maps_compressed_boundaries(monkeypatch):
    bsa_calls = []
    beta_rows = []

    def fake_bsa(X, y, **kwargs):
        rows = np.asarray(X)[:, 0].astype(int)
        bsa_calls.append((rows.copy(), kwargs.copy()))
        if len(rows) == 12:
            return np.array([1, 6, 12])
        return np.array([1, 2, len(rows)])

    def fake_beta(X, y, c1):
        beta_rows.append(np.asarray(X)[:, 0].astype(int).copy())
        return np.zeros(1)

    monkeypatch.setattr(comparison, "glm_bsa3", fake_bsa)
    monkeypatch.setattr(comparison, "glm_bsa3_segment_beta", fake_beta)

    X = np.arange(12, dtype=float).reshape(-1, 1)
    y = np.tile([0.0, 1.0], 6)
    best, cp, scores = glm_bsa3_cv(
        X, y, c1_candidates=(0.18,), num_folds=3
    )

    assert best == 0.18
    assert np.array_equal(cp, np.array([6]))
    assert scores.shape == (1,) and np.isfinite(scores[0])
    assert len(bsa_calls) == 4
    for fold_number, (rows, kwargs) in enumerate(bsa_calls[:3]):
        assert not np.intersect1d(rows, np.arange(fold_number, 12, 3)).size
        assert kwargs["m1"] is None
        assert kwargs["c1"] == 0.18
        assert kwargs["quiet"] is True
    # Fold 0 trains on [1,2,4,5,7,8,10,11]; compressed cp=2 maps to 4.
    assert np.array_equal(beta_rows[0], np.array([1, 2]))
    assert np.array_equal(beta_rows[1], np.array([4, 5, 7, 8, 10, 11]))


def test_glm_bsa3_cv_selects_by_prediction_loss_without_penalty(monkeypatch):
    def fake_bsa(X, y, **kwargs):
        return np.array([1, len(y)])

    def fake_beta(X, y, c1):
        return np.array([c1])

    monkeypatch.setattr(comparison, "glm_bsa3", fake_bsa)
    monkeypatch.setattr(comparison, "glm_bsa3_segment_beta", fake_beta)

    X = np.ones((10, 1))
    y = np.zeros(10)
    candidates = (0.1, 0.2, 0.4)
    best, cp, scores = glm_bsa3_cv(
        X, y, c1_candidates=candidates, num_folds=5
    )

    expected = 10 * np.logaddexp(0.0, np.asarray(candidates))
    assert np.allclose(scores, expected)
    assert best == 0.1
    assert cp.size == 0


def test_glm_bsa3_cv_skips_failed_candidate_and_validates_grid(monkeypatch):
    monkeypatch.setattr(
        comparison,
        "glm_bsa3",
        lambda X, y, **kwargs: np.array([1, len(y)]),
    )

    def fake_beta(X, y, c1):
        if c1 == 0.1:
            raise RuntimeError("candidate failure")
        return np.zeros(X.shape[1])

    monkeypatch.setattr(comparison, "glm_bsa3_segment_beta", fake_beta)
    best, cp, scores = glm_bsa3_cv(
        np.ones((10, 2)),
        np.tile([0.0, 1.0], 5),
        c1_candidates=(0.1, 0.2),
        num_folds=2,
    )
    assert best == 0.2
    assert np.isinf(scores[0]) and np.isfinite(scores[1])
    assert cp.size == 0

    for bad in ((), (0.0,), (np.nan,), (np.inf,)):
        with pytest.raises(ValueError):
            glm_bsa3_cv(
                np.ones((10, 2)),
                np.tile([0.0, 1.0], 5),
                c1_candidates=bad,
                num_folds=2,
            )
