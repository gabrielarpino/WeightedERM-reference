import json
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.optimize

import WERM.estimators as estimators
import WERM.experiment_utils as experiment_utils


def _notebook_code(path):
    notebook = json.loads(Path(path).read_text())
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def test_weighted_erm_structured_l1_path_uses_fast_solver_and_warm_starts(
    monkeypatch,
):
    calls = []

    def fake_fast_solver(X, y, sample_weight, **kwargs):
        calls.append((np.asarray(sample_weight), kwargs))
        coefficient = np.full(X.shape[1], len(calls), dtype=float)
        return SimpleNamespace(coef=coefficient, converged=True)

    monkeypatch.setattr(estimators, "fit_weighted", fake_fast_solver)
    monkeypatch.setattr(
        estimators.scipy.optimize,
        "minimize",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("generic nonsmooth optimizer was called")
        ),
    )
    monkeypatch.setattr(
        estimators,
        "_exact_partition_search",
        lambda *args: (np.array([3]), 1.0),
    )

    changepoints, coefficients = estimators.weighted_erm(
        X=np.arange(18.0).reshape(6, 3),
        y=np.array([0, 1, 0, 1, 0, 1]),
        M=lambda prediction, truth: jnp.logaddexp(0.0, prediction)
        - truth * prediction,
        L=2,
        Delta_n=1,
        gamma=0.0,
        return_B_hat=True,
        coefficient_loss="logistic",
        coefficient_lambda=0.25,
        unpenalized_feature_indices=(0,),
        coefficient_solver_options={"tol": 1e-9},
    )

    np.testing.assert_array_equal(changepoints, [3])
    np.testing.assert_array_equal(coefficients[:, 0], np.ones(3))
    np.testing.assert_array_equal(coefficients[:, 1], np.full(3, 2.0))
    assert len(calls) == 2
    assert all(weight.sum() == pytest.approx(1.0) for weight, _ in calls)
    assert calls[0][1] == {
        "loss": "logistic",
        "loss_scale": 1.0,
        "penalty": "l1",
        "penalty_strength": 0.25,
        "huber_delta": 1.35,
        "unpenalized_indices": (0,),
        "initial": None,
        "tol": 1e-9,
    }
    np.testing.assert_array_equal(calls[1][1]["initial"], np.ones(3))


def test_weighted_erm_explicit_marginals_are_used_and_validated(monkeypatch):
    calls = []

    def fake_fast_solver(X, y, sample_weight, **kwargs):
        calls.append(np.asarray(sample_weight))
        return SimpleNamespace(coef=np.zeros(X.shape[1]), converged=True)

    monkeypatch.setattr(estimators, "fit_weighted", fake_fast_solver)
    monkeypatch.setattr(
        estimators, "_exact_partition_search",
        lambda *args: (np.array([3]), 1.0),
    )
    X = np.arange(12.0).reshape(6, 2)
    y = np.arange(6.0)
    M = lambda prediction, truth: 0.5 * (prediction - truth) ** 2
    phi = np.array([[1, 1, .8, .2, 0, 0], [0, 0, .2, .8, 1, 1]])
    estimators.weighted_erm(
        X, y, M, L=2, Delta_n=1, gamma=0,
        marginal_weights=phi, coefficient_loss="least_squares",
        coefficient_penalty="none",
    )
    np.testing.assert_allclose(calls[0], phi[0] / phi[0].sum())
    np.testing.assert_allclose(calls[1], phi[1] / phi[1].sum())
    for invalid in (phi[:, :-1], -phi, phi * 2, np.zeros_like(phi)):
        with pytest.raises(ValueError, match="marginal_weights"):
            estimators.weighted_erm(
                X, y, M, L=2, Delta_n=1, gamma=0,
                marginal_weights=invalid,
            )


def test_joint_cv_structured_huber_path_uses_fast_solver_for_segment_refits(
    monkeypatch,
):
    werm_calls = []
    segment_calls = []

    def fake_weighted_erm(**kwargs):
        werm_calls.append(kwargs)
        return np.array([], dtype=int)

    def fake_fast_solver(X, y, sample_weight, **kwargs):
        segment_calls.append((np.asarray(X), np.asarray(sample_weight), kwargs))
        return SimpleNamespace(coef=np.zeros(X.shape[1]), converged=True)

    monkeypatch.setattr(estimators, "weighted_erm", fake_weighted_erm)
    monkeypatch.setattr(experiment_utils, "fit_weighted", fake_fast_solver)
    monkeypatch.setattr(
        scipy.optimize,
        "minimize",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("generic nonsmooth segment optimizer was called")
        ),
    )

    experiment_utils.weighted_erm_joint_cv(
        X=np.c_[np.ones(8), np.arange(8.0)],
        y=np.arange(8.0),
        M=lambda prediction, truth: jnp.where(
            jnp.abs(prediction - truth) <= 0.9,
            0.5 * (prediction - truth) ** 2,
            0.9 * (jnp.abs(prediction - truth) - 0.45),
        ),
        M_scorer=lambda prediction, truth: np.zeros_like(truth),
        Lmax=1,
        delta_fraction=0.2,
        l1_constants=(0.5,),
        num_folds=2,
        coefficient_loss="huber",
        huber_delta=0.9,
        unpenalized_feature_indices=(0,),
        coefficient_solver_options={"tol": 1e-8},
    )

    assert len(werm_calls) == 3
    assert len(segment_calls) == 2
    for call in werm_calls:
        assert call["coefficient_loss"] == "huber"
        assert call["huber_delta"] == 0.9
        assert call["unpenalized_feature_indices"] == (0,)
        assert call["coefficient_solver_options"] == {"tol": 1e-8}
        assert call["coefficient_lambda"] > 0.0
        assert call["coefficient_penalty"] == "l1"
    for X_segment, weights, kwargs in segment_calls:
        np.testing.assert_array_equal(weights, np.ones(X_segment.shape[0]))
        assert kwargs["loss"] == "huber"
        assert kwargs["penalty"] == "l1"
        assert kwargs["penalty_strength"] > 0.0
        assert kwargs["huber_delta"] == pytest.approx(0.9)
        assert kwargs["unpenalized_indices"] == (0,)
        assert kwargs["tol"] == 1e-8


def test_redo_notebooks_select_the_structured_fast_l1_solver():
    m1 = _notebook_code("M1_upsample_redo.ipynb")
    m3 = _notebook_code("M3_upsample_redo.ipynb")
    m1_redo2 = _notebook_code("M1_upsample_redo2.ipynb")
    m2_redo2 = _notebook_code("M2_upsample_redo2.ipynb")
    m3_redo2 = _notebook_code("M3_upsample_redo2.ipynb")

    assert 'coefficient_loss="huber"' in m1
    assert 'coefficient_loss="logistic"' in m3
    for folds in (2, 5, 10):
        assert f"WERM-unp-{folds}fold" in m1_redo2
        assert f"WERM-unp-{folds}fold" in m2_redo2
    assert "WERM-L1-jointCV" in m1_redo2
    assert "[0.03, 0.3, 3.0]" in m1_redo2
    assert "selection_strategy='exhaustive'" in m1_redo2
    assert "charcoal" in m2_redo2
    assert "GLMBSA-CV" in m3_redo2
    glmbsa_cv_branch = m3_redo2.split("elif method_name in glmbsa_cv_folds:", 1)[1].split(
        "elif method_name == 'GLMBSA-fixed0.18-2fold':", 1
    )[0]
    assert "num_folds=glmbsa_cv_folds[method_name]" in glmbsa_cv_branch
    assert "'GLMBSA-CV': 2" in m3_redo2
    assert "'GLMBSA-CV-5fold': 5" in m3_redo2
    assert "'GLMBSA-CV-10fold': 10" in m3_redo2
    assert "GLMBSA-CV (2-fold)" in m3_redo2
    assert "GLMBSA-fixed0.18-2fold" in m3_redo2
    assert "WERM-L1-fullCV" in m3_redo2
    assert "WERM-L1-C0.3-2fold" in m3_redo2
    for method_name in (
        "WERM-unp-2fold",
        "WERM-unp-10fold",
        "WERM-L1-C0.18-2fold",
        "WERM-L1-C0.18-10fold",
        "WERM-L1-C0.3-10fold",
        "WERM-L1-fullCV-2fold",
        "WERM-L1-fullCV-10fold",
    ):
        assert method_name in m3_redo2
    assert "c1_candidates=(0.18,)" in m3_redo2
    assert "l1_constants=(0.3,)" in m3_redo2
    assert "l1_constants=(0.18,)" in m3_redo2
    assert m3_redo2.count("num_folds=2") >= 2
    assert "num_folds=cv_folds" in m3_redo2
    assert "penalty_norm='none'" in m3_redo2
    assert "coefficient_solver_options={'tol': 1e-4, 'max_iter': 20000}" in m3_redo2
    assert "existing_trial_result" in m3_redo2
    assert "reused_methods" in m3_redo2
    assert "batch_n_jobs = max(1, active_n_jobs)" in m3_redo2
    assert "batch_n_jobs = min(max(1, active_n_jobs), len(pending))" not in m3_redo2
    assert "prior_checkpoint_signature" in m3_redo2
    assert "rerun_methods = ('GLMBSA-CV-5fold', 'GLMBSA-CV-10fold')" in m3_redo2
    assert "'glmbsa_cv_folds': (2, 5, 10)" in m3_redo2
    assert "if os.path.isdir(prior_checkpoint_dir):" in m3_redo2
    assert 'coefficient_loss="logistic"' in m3_redo2
    assert "jnp.logaddexp(0.0, u) - jnp.multiply(v, u)" in m3_redo2
    assert "selection_strategy='exhaustive'" in m3_redo2
    for plot_call in (
        "plot_location_asymmetric_errors(",
        "plot_size_asymmetric_errors(",
        "plot_runtime_asymmetric_errors(",
        "plot_location_boxplots(",
        "plot_size_boxplots(",
        "plot_runtime_boxplots(",
        "plot_changepoint_histograms_publication(",
    ):
        assert plot_call in m3_redo2
    assert "from scripts.run_m1_upsample_redo2 import" not in m1_redo2
    assert "subprocess" not in m1_redo2 + m2_redo2 + m3_redo2


def test_m3_delta6_notebook_mirrors_the_extended_delta3_comparison():
    delta6 = _notebook_code("M3_upsample_redo_delta6.ipynb")
    assert "fixed_delta = 6.0" in delta6
    assert "num_trials = 100" in delta6
    assert "WERM_N_JOBS', '36'" in delta6
    assert "coefficient_loss='logistic'" in delta6
    for method_name in (
        "GLMBSA-CV",
        "GLMBSA-fixed0.18-2fold",
        "WERM-L1-fullCV",
        "WERM-L1-C0.3-2fold",
        "WERM-unp-2fold",
        "WERM-unp-10fold",
        "WERM-L1-C0.18-2fold",
        "WERM-L1-C0.18-10fold",
        "WERM-L1-C0.3-10fold",
        "WERM-L1-fullCV-2fold",
        "WERM-L1-fullCV-10fold",
    ):
        assert method_name in delta6
    for plot_call in (
        "plot_location_asymmetric_errors(",
        "plot_size_asymmetric_errors(",
        "plot_runtime_asymmetric_errors(",
        "plot_location_boxplots(",
        "plot_size_boxplots(",
        "plot_runtime_boxplots(",
        "plot_changepoint_histograms_publication(",
    ):
        assert plot_call in delta6


def test_m3_expanded_notebooks_use_short_unique_result_artifacts():
    """Keep result/figure path components below common 255-byte filesystem limits."""
    expected = {
        "M3_upsample_redo2.ipynb": "M3d_redo_delta3_glmbsa2-5-10_thirteen_methods_folds2-5-10_seed42",
        "M3_upsample_redo_delta6.ipynb": "M3d_redo_delta6_glmbsa2-5-10_thirteen_methods_folds2-5-10_seed42",
    }
    for notebook_name, stem in expected.items():
        source = _notebook_code(notebook_name)
        assert stem in source
        assert "pickle.dump(payload" in source
        assert "'_smoke' if SMOKE_MODE else '_100trials'" in source
        assert ".replace('_100trials.pkl', '_smoke.pkl')" in source
        assert len(stem + "_100trials.pkl") < 255


def test_num_chgpt_cv_keeps_fit_loss_separate_from_absolute_scorer(monkeypatch):
    fit_loss = lambda prediction, truth: jnp.where(
        jnp.abs(prediction - truth) <= 1.35,
        0.5 * (prediction - truth) ** 2,
        1.35 * (jnp.abs(prediction - truth) - 0.675),
    )
    score_loss = lambda prediction, truth: jnp.abs(prediction - truth)
    captured = {}

    def fake_cv(**kwargs):
        captured.update(kwargs)
        return 1, np.array([], dtype=int)

    monkeypatch.setattr(estimators, "chgpt_k_fold_cv", fake_cv)
    estimators.weighted_erm_num_chgptCV(
        np.ones((8, 2)), np.arange(8.0), fit_loss, Delta_n=1,
        Lmax=1, M_scorer=score_loss, num_folds=2, n_jobs=1,
    )

    assert captured["M_scorer"] is score_loss
    assert captured["M_fitter"] is fit_loss
