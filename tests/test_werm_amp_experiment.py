"""Protocol checks for the paired WERM/AMP experiment."""
import numpy as np

from tools import werm_amp_experiment as exp
from tools.werm_amp_partition import prior_marginals


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
    from itertools import combinations
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


def test_cv_matches_existing_audited_protocol_and_penalty_scale():
    # Existing helper has separate production-parity tests; this protects
    # the new standalone runner's fold mapping, refits and squared-loss scale.
    import pytest
    run_cv = pytest.importorskip('tools.werm_cv_empirical').run_cv
    rng = np.random.default_rng(731)
    spec = dict(exp.DEFAULT_SPEC,p=8)
    n = 100
    X = rng.normal(size=(n,8))/np.sqrt(n)
    y = rng.normal(size=n)
    weights = {(m,k):prior_marginals(m,k,exp.minimum_gap(m,spec),fixed_count=True)
               for m in (90,100) for k in range(1,5)}
    observed = exp.fit_werm_methods(X,y,weights,spec)
    expected = run_cv(X,y,4,weights,1/40,folds=10,loss_scale=2.,
                      return_coefficients=True)
    assert np.allclose(observed['WERM+CV']['cv_scores'],expected['normalized_cv_scores'],
                       atol=1e-10,rtol=1e-10)
    assert np.array_equal(observed['WERM+CV']['cp'],expected['selected_by_bound'][4]['cp'])
    from math import comb,log
    penalties = np.array([6*log(comb(n-1,k-1)) for k in range(1,5)])
    assert np.allclose(observed['WERM+P']['candidate_penalties'],penalties)
    raw = []
    for k in range(1,5):
        fit = expected['full_fits'][k]
        labels = np.searchsorted(fit['cp'],np.arange(n),side='right')
        predictions = np.einsum('ij,ij->i',X,fit['coefficients'][:,labels].T)
        raw.append(np.sum((predictions-y)**2))
    assert np.allclose(observed['WERM+P']['candidate_raw'],raw)
    winner = np.argmin(np.asarray(raw)+penalties)+1
    assert np.array_equal(observed['WERM+P']['cp'],expected['full_fits'][winner]['cp'])
