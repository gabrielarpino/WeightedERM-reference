"""Protocol checks for the paired WERM/AMP experiment."""
from itertools import combinations
from math import comb, log

import numpy as np
import pytest

from tools import werm_amp_experiment as exp
from tools.werm_amp_partition import amp_prior_marginals, prior_marginals


def exhaustive_upper_bound_reference(X, y, phi, gap, gamma):
    """Independent weighted-LS and enumeration oracle with one shared fit."""
    n = len(y)
    coefficients = np.column_stack([
        np.linalg.lstsq(np.sqrt(w[:, None])*X, np.sqrt(w)*y, rcond=None)[0]
        for w in phi
    ])
    prediction = X @ coefficients
    raw, points = [], []
    for k in range(1, len(phi)+1):
        candidates = [cp for cp in combinations(range(1, n), k-1)
                      if np.all(np.diff(cp) >= gap)]

        def loss(cp):
            labels = np.searchsorted(cp, np.arange(n), side='right')
            return .5*np.sum((prediction[np.arange(n), labels]-y)**2)

        winner = min(candidates, key=loss)
        raw.append(loss(winner))
        points.append(np.asarray(winner, dtype=int))
    penalties = np.array([gamma*log(comb(n-1, k-1)) for k in range(1, len(phi)+1)])
    raw = np.asarray(raw)
    winner = int(np.argmin(raw+penalties))
    return dict(raw=raw, penalties=penalties, total=raw+penalties,
                points=points, winner=winner, coefficients=coefficients)


def test_covariance_conventions_and_paired_latent_data():
    spec = dict(exp.DEFAULT_SPEC,p=20)
    matrices = exp.covariance_matrices(spec)
    assert np.allclose(matrices['ar1'][0,:4],[1.,.4,.16,.064])
    for S in matrices.values():
        assert np.allclose(S,S.T)
        assert np.allclose(np.diag(S),1.)
        assert np.linalg.eigvalsh(S).min()>0
    draws = {c:exp.generate_data(spec,'unknown',c,3.5,0,matrices) for c in matrices}
    X0 = draws['identity'][0]
    assert len({draw[3] for draw in draws.values()})==1
    for c,(X,y,truth,_) in draws.items():
        assert np.allclose(X,X0@np.linalg.cholesky(matrices[c]).T)
        assert np.array_equal(truth,[14,35])
    assert not np.allclose(draws['ar1'][0],X0)


def test_empty_and_failed_outputs_are_distinct():
    truth = np.array([20,50])
    empty = exp.add_metrics(dict(status='ok',cp=[]),truth,100)
    failed = exp.add_metrics(dict(status='failed',error='test'),truth,100)
    located = exp.add_metrics(dict(status='ok',cp=[25,50]),truth,100)
    assert empty['count']==0 and empty['empty'] and np.isnan(empty['haus'])
    assert np.isnan(failed['count']) and not failed['empty']
    assert located['haus']==.05


def test_fixed_werm_equals_direct_exhaustive_partition():
    rng = np.random.default_rng(172)
    X,y = rng.normal(size=(12,3)),rng.normal(size=12)
    phi = prior_marginals(12,3,2,fixed_count=True)
    B = np.column_stack([np.linalg.lstsq(np.sqrt(w[:,None])*X,
                                        np.sqrt(w)*y,rcond=None)[0] for w in phi])
    candidates = [cp for cp in combinations(range(1,12),2) if cp[1]-cp[0]>=2]
    def rss(cp):
        labels = np.searchsorted(cp,np.arange(12),side='right')
        return np.sum((np.einsum('ij,ij->i',X,B[:,labels].T)-y)**2)
    best = min(candidates,key=rss)
    fit = exp.fixed_werm(X,y,phi,2)
    assert np.array_equal(fit['cp'],best)
    assert np.isclose(fit['raw_loss'],rss(best))


def test_cv_matches_existing_audited_protocol_and_penalty_uses_upper_bound_fit():
    # Existing helper has separate production-parity tests; this protects
    # the new standalone runner's fold mapping, refits and squared-loss scale.
    run_cv = pytest.importorskip('tools.werm_cv_empirical').run_cv
    rng = np.random.default_rng(731)
    spec = dict(exp.DEFAULT_SPEC,p=8)
    n = 100
    X = rng.normal(size=(n,8))/np.sqrt(n)
    y = rng.normal(size=n)
    conditional = {(m,k):prior_marginals(m,k,exp.minimum_gap(m,spec),fixed_count=True)
                   for m in (90,100) for k in range(1,5)}
    weights = dict(conditional)
    weights[n, 'werm_upper'] = prior_marginals(n,4,exp.minimum_gap(n,spec))
    weights[n, 'mixture'] = amp_prior_marginals(n,4,exp.minimum_gap(n,spec))
    assert not np.allclose(weights[n, 'werm_upper'], weights[n, 'mixture'])
    observed = exp.fit_werm_methods(X,y,weights,spec)
    expected = run_cv(X,y,4,conditional,1/40,folds=10,loss_scale=2.,
                      return_coefficients=True)
    assert np.allclose(observed['WERM+CV']['cv_scores'],expected['normalized_cv_scores'],
                       atol=1e-10,rtol=1e-10)
    assert np.array_equal(observed['WERM+CV']['cp'],expected['selected_by_bound'][4]['cp'])
    from WERM.estimators import weighted_erm
    from WERM.gof import chgpt_gof_dp, manuscript_log_binomial_k_penalty
    loss = lambda a, b: .5*(np.asarray(a)-np.asarray(b))**2
    penalty = manuscript_log_binomial_k_penalty(n,6.)
    cp, coefficients = weighted_erm(
        X, y, loss, L=4, Delta_n=exp.minimum_gap(n,spec), gamma=0.,
        L_is_upper_bound=True, return_B_hat=True,
        partition_penalty=penalty, coefficient_loss='least_squares',
        coefficient_penalty='none', marginal_weights=weights[n, 'werm_upper'])
    native = chgpt_gof_dp(loss, X@coefficients, y, n, 4, exp.minimum_gap(n,spec),
                         L_is_upper_bound=True, penalty=penalty)
    np.testing.assert_array_equal(observed['WERM+P']['cp'], cp)
    np.testing.assert_allclose(observed['WERM+P']['candidate_raw'],
                               [native['raw_by_K'][k] for k in range(1,5)],
                               atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(observed['WERM+P']['candidate_penalties'],
                               [penalty(k) for k in range(1,5)])


@pytest.mark.parametrize('n,p,zero_response', [(103, 8, False), (100, 45, False),
                                             (103, 8, True)])
def test_cv_matches_canonical_native_procedure_with_identical_priors(
        monkeypatch, n, p, zero_response):
    """Check actual M2 CV, isolating its different prior construction.

    Cases cover unequal fold sizes, underdetermined segment refits and exact
    count ties. Only the native estimator's marginal weights are supplied;
    its fitting, partition search, refits, scoring and selection run unchanged.
    """
    import WERM.estimators
    from WERM.experiment_utils import weighted_erm_joint_cv

    rng = np.random.default_rng(1731)
    spec = dict(exp.DEFAULT_SPEC, p=p, separation_fraction=1/10)
    X = rng.normal(size=(n, p))/np.sqrt(n)
    beta = rng.normal(size=(p, 3))*np.sqrt(n/p)
    labels = np.searchsorted([round(.25*n), round(.5*n)], np.arange(n),
                             side='right')
    y = (np.zeros(n) if zero_response else
         np.sum(X*beta[:, labels].T, axis=1)+rng.normal(scale=.3, size=n))
    sizes = {n, *[n-len(range(fold, n, 10)) for fold in range(10)]}
    weights = {(m, k): prior_marginals(m, k, exp.minimum_gap(m, spec),
                                      fixed_count=True)
               for m in sizes for k in range(1, 5)}
    weights[n, 'werm_upper'] = prior_marginals(n, 4, exp.minimum_gap(n, spec))
    native_estimator = WERM.estimators.weighted_erm
    native_calls = []

    def native_with_matching_prior(*args, **kwargs):
        m, k = len(kwargs['y']), kwargs['L']
        assert kwargs['coefficient_penalty'] == 'none'
        assert kwargs['coefficient_lambda'] == 0.
        assert kwargs['gamma'] == 0.
        assert not kwargs['L_is_upper_bound']
        assert kwargs['Delta_n'] == exp.minimum_gap(m, spec)
        native_calls.append((m, k))
        return native_estimator(*args, **kwargs, marginal_weights=weights[m, k])

    monkeypatch.setattr(WERM.estimators, 'weighted_erm', native_with_matching_prior)
    pair, cp, diagnostics = weighted_erm_joint_cv(
        X, y, lambda a, b: (a-b)**2, lambda a, b: np.abs(a-b),
        Lmax=4, delta_fraction=1/10, l1_constants=(0.,), penalty_norm='none',
        coefficient_loss='least_squares', num_folds=10, n_jobs=1,
        standardize_y=False, selection_strategy='exhaustive',
        return_diagnostics=True)
    actual = exp.fit_werm_methods(X, y, weights, spec)['WERM+CV']
    np.testing.assert_allclose(actual['cv_scores'], np.asarray(diagnostics['scores'])/n,
                               rtol=1e-9, atol=1e-9)
    np.testing.assert_array_equal(actual['cp'], cp)
    assert actual['segments'] == pair[0]
    assert pair[1] == 0.
    assert len(native_calls) == 4*10+1
    assert native_calls[-1] == (n, pair[0])
    if p == 45:
        assert actual['min_refit_rows'] < p
    if zero_response:
        assert actual['segments'] == 1
        assert actual['cp'].size == 0
        np.testing.assert_array_equal(actual['cv_scores'], np.zeros(4))


@pytest.mark.parametrize('gamma', [0., 6., 1.e6])
def test_penalized_werm_matches_enumeration_with_one_shared_fit(gamma):
    rng = np.random.default_rng(20261109)
    n, p, L, gap = 12, 3, 4, 2
    X, y = rng.normal(size=(n,p)), 4*rng.normal(size=n)
    phi = prior_marginals(n,L,gap)
    expected = exhaustive_upper_bound_reference(X,y,phi,gap,gamma)
    observed = exp.penalized_werm(X,y,phi,gap,gamma)

    np.testing.assert_allclose(observed['candidate_raw'], expected['raw'],
                               atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(observed['candidate_penalties'], expected['penalties'])
    np.testing.assert_allclose(observed['candidate_total'], expected['total'],
                               atol=1e-9, rtol=1e-10)
    for actual, reference in zip(observed['candidate_cp'], expected['points']):
        np.testing.assert_array_equal(actual, reference)
    winner = expected['winner']
    assert observed['segments'] == winner+1
    assert observed['count'] == winner
    assert observed['status'] == 'ok'
    assert observed['coefficient_fit_scheme'] == 'upper_bound_half_squared_v1'
    np.testing.assert_array_equal(observed['cp'], expected['points'][winner])
    assert np.isclose(observed['raw_loss'], expected['raw'][winner])
    if gamma == 1.e6:
        assert observed['count'] == 0
        assert observed['cp'].size == 0
        assert observed['candidate_penalties'][0] == 0


def test_penalized_werm_fits_only_the_L_upper_bound_columns(monkeypatch):
    rng = np.random.default_rng(731)
    X, y = rng.normal(size=(20,4)), rng.normal(size=20)
    phi = prior_marginals(20,4,2)
    calls = []
    original = exp.least_squares

    def recording_fit(X, y, weight, *, loss_scale=2.):
        calls.append((np.asarray(weight).copy(), loss_scale))
        return original(X, y, weight, loss_scale=loss_scale)

    monkeypatch.setattr(exp, 'least_squares', recording_fit)
    exp.penalized_werm(X,y,phi,2,6.)
    assert len(calls) == 4  # Separate K-dependent fits would require 1+2+3+4.
    for (actual, scale), expected in zip(calls, phi):
        assert scale == 1.
        np.testing.assert_allclose(actual, expected/expected.sum())
