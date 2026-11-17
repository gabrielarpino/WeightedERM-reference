"""Actual-design smoke checks against native CV and penalized WERM.

Run once before the matched 100-trial study. These checks use its frozen
weights and n=1000, p=100 data, without rerunning the full experiment.
"""
import numpy as np
import pytest


@pytest.fixture(scope="module")
def matched_design():
    from tools import el_matched_selection as study

    _, specification, weights = study.prepare(trials=100, theory_draws=20)
    X, y, beta, truth, labels = study.generate_data(0)
    assert X.shape == (1000, 100)
    np.testing.assert_array_equal(truth, [250, 500, 750])
    np.testing.assert_array_equal(labels, np.repeat(np.arange(4), 250))
    return X, y, weights


def half_squared(prediction, response):
    return .5 * (prediction - response) ** 2


def test_matched_cv_matches_native_at_actual_design(monkeypatch, matched_design):
    import WERM.config
    from WERM.estimators import weighted_erm
    from WERM.experiment_utils import weighted_erm_joint_cv
    from tools import werm_cv_empirical as fast

    X, y, weights = matched_design

    def frozen(Lmin, Lmax, Delta_small, n_small=50, n_large=1000,
               p_l=None, verbose=False):
        assert Lmin == Lmax and p_l is None
        assert n_small == 50 and Delta_small == 5
        return weights['cv'][n_large, Lmax]

    monkeypatch.setattr(WERM.config, 'extrapolated_marginals', frozen)
    expected, expected_cp, diagnostics = weighted_erm_joint_cv(
        X, y, half_squared, lambda a, b: np.abs(a - b),
        Lmax=6, delta_fraction=.1, l1_constants=(0.,), penalty_norm='none',
        coefficient_loss='least_squares', num_folds=10, n_jobs=1,
        standardize_y=False, selection_strategy='exhaustive',
        return_diagnostics=True,
    )
    actual = fast.run_cv(X, y, 6, weights['cv'], .1, folds=10,
                         minimum_segment_fraction=None)
    np.testing.assert_allclose(actual['cv_scores'], diagnostics['scores'],
                               rtol=1e-10, atol=1e-8)
    assert actual['selected_by_bound'][6]['selected_segments'] == expected[0]
    np.testing.assert_array_equal(actual['selected_by_bound'][6]['cp'], expected_cp)

    # Verify every possible final count, not only the winner at the largest L.
    native_full_fits = {}
    for count in range(1, 7):
        native_full_fits[count] = weighted_erm(
            X=X, y=y, M=half_squared, L=count, Delta_n=100, gamma=0.,
            L_is_upper_bound=False, coefficient_loss='least_squares',
            coefficient_penalty='none', coefficient_lambda=0.,
            marginal_weights=weights['cv'][1000, count],
        )
        np.testing.assert_array_equal(actual['full_fits'][count]['cp'],
                                      native_full_fits[count])

    for bound in range(1, 7):
        winner = int(np.argmin(np.asarray(diagnostics['scores'])[:bound])) + 1
        selected = actual['selected_by_bound'][bound]
        assert selected['selected_segments'] == winner
        np.testing.assert_array_equal(selected['cp'], native_full_fits[winner])


def test_matched_penalty_matches_native_at_every_bound(matched_design):
    from WERM.estimators import weighted_erm
    from WERM.gof import manuscript_log_binomial_k_penalty
    from tools import el_state_evolution as penalty

    X, y, weights = matched_design
    for bound in range(1, 7):
        phi = weights['penalty'][bound]
        theta = penalty.empirical_fit(X, y, phi)
        actual = penalty.search(theta, y, 100, upper_bound=True,
                                penalty_weight=6.)
        native_cp, native_objective = weighted_erm(
            X=X, y=y, M=half_squared, L=bound, Delta_n=100, gamma=0.,
            L_is_upper_bound=True, return_gof=True,
            coefficient_loss='least_squares', coefficient_penalty='none',
            coefficient_lambda=0., marginal_weights=phi,
            partition_penalty=manuscript_log_binomial_k_penalty(1000, 6.),
        )
        np.testing.assert_array_equal(actual['cp'], native_cp)
        assert actual['segments'] == len(native_cp) + 1
        assert actual['total_loss'] == pytest.approx(native_objective,
                                                    rel=1e-10, abs=1e-8)
