"""Scientific equivalence checks for the isolated adjusted-M2 experiment."""
import numpy as np
import pytest
from tools import m2_adjusted_experiment as experiment


@pytest.mark.parametrize('n,gap', [(12, 1), (23, 3), (51, 5)])
@pytest.mark.parametrize('count', [1, 2, 3, 4])
def test_fast_fixed_count_dp_matches_reference(n, gap, count):
    from WERM.gof import chgpt_gof_dp
    rng = np.random.default_rng(91 + count)
    for tied in (False, True):
        theta = np.zeros((n, count)) if tied else rng.normal(size=(n, count))
        y = np.zeros(n) if tied else rng.normal(size=n)
        expected = chgpt_gof_dp(lambda u, v: (u-v)**2, theta, y, n, count, gap)
        actual = experiment.squared_partition(theta, y, gap)
        np.testing.assert_array_equal(actual, expected['changepoints'])


def test_standard_cv_matches_original_m2_helper(monkeypatch):
    from scipy.special import comb
    import WERM.estimators
    from WERM.experiment_utils import weighted_erm_joint_cv
    rng = np.random.default_rng(12)
    X = rng.normal(size=(60, 6)) / np.sqrt(60)
    y = rng.normal(size=60)
    weights = {}
    for n in (54, 60):
        t = np.arange(1, n+1)/n
        for count in range(1, 5):
            weights[n, count] = np.array([comb(count-1, l)*t**l*(1-t)**(count-1-l)
                                         for l in range(count)])
    original = WERM.estimators.weighted_erm
    def same_weights(**kwargs):
        kwargs['marginal_weights'] = weights[len(kwargs['y']), kwargs['L']]
        return original(**kwargs)
    monkeypatch.setattr(WERM.estimators, 'weighted_erm', same_weights)
    pair, cp, diagnostics = weighted_erm_joint_cv(
        X, y, M=lambda u,v: (u-v)**2, M_scorer=lambda u,v: np.abs(u-v),
        Lmax=4, delta_fraction=1/40, num_folds=10, penalty_norm='none',
        coefficient_loss='least_squares', return_diagnostics=True)
    actual = experiment.select_count(X, y, False, weights)
    assert actual['selected_segments'] == pair[0]
    np.testing.assert_array_equal(actual['cp'], cp)
    np.testing.assert_allclose(actual['cv_scores'].sum(axis=1), diagnostics['scores'], rtol=1e-10)


def test_adjustment_is_paper_gradient_ascent_and_fails_without_root():
    from WERM.signal_strength import empirical_adjustment
    n, p = 60, 20
    rng = np.random.default_rng(4)
    theta, y = rng.normal(size=(n, 2)), rng.normal(size=n)
    phi = np.full((2, n), .5)
    adjusted, b = empirical_adjustment(theta, y, phi, n/p, loss='squared')
    # With constant weights the equation has b*w=1/[2(delta-1)].
    expected = theta + (theta-y[:, None])/(n/p-1)
    np.testing.assert_allclose(adjusted, expected, atol=1e-12)
    np.testing.assert_allclose(np.mean(1/(1+2*b[:,None]*phi),axis=1), 1-p/n)
    with pytest.raises(ValueError, match='delta > 1'):
        empirical_adjustment(theta, y, phi, 1., loss='squared')
    phi[0, :50] = 0
    with pytest.raises(ValueError, match='No finite positive'):
        empirical_adjustment(theta, y, phi, n/p, loss='squared')
