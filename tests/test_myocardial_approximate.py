import numpy as np
from WERM.signal_strength import configuration_prior, squared_logistic_state
from WERM.myocardial_approximate import approximate_rho, logistic_prox, logistic_state
from scipy.special import expit

def test_prox_satisfies_logistic_first_order_equation():
    v=np.linspace(-25,25,101)[:,None]
    a=np.array([.01,1.,20.])[None,:]
    for y in (0,1):
        t=logistic_prox(v,a,y)
        np.testing.assert_allclose(t+a*(expit(t)-y),np.broadcast_to(v,t.shape),atol=2e-12)

def test_approximate_rho_matches_feasible_moments():
    _,_,phi=configuration_prior(100,[20,40,60,80])
    psi=(np.arange(100)>=50).astype(int)
    rho=np.array([[.6,.15],[.15,1.1]])
    S=squared_logistic_state(rho,phi,psi,8)['moment']
    result=approximate_rho(S,phi,psi,8)
    assert result['optimizer_success']
    np.testing.assert_allclose(result['rho'],rho,atol=2e-4)
    assert result['residual_norm']<1e-6

def test_approximate_rho_returns_spd_and_reports_nonzero_error():
    _,_,phi=configuration_prior(100,[20,40,60,80]);psi=(np.arange(100)>=50).astype(int)
    result=approximate_rho(np.eye(2)*.001,phi,psi,8)
    assert np.linalg.eigvalsh(result['rho']).min()>0
    assert result['residual_norm']>.01
    assert not result['exact_moment_match']

def test_logistic_state_residual_and_covariance():
    _,_,phi=configuration_prior(30,[10,20]);psi=(np.arange(30)>=15).astype(int)
    result=logistic_state(np.array([[.4,.1],[.1,.7]]),phi,psi,12,sample_power=8)
    assert result['training_residual']<1e-5
    assert np.linalg.eigvalsh(result['kappa']).min()>0
    assert np.isfinite(result['lambda']).all()

def test_small_ridge_fit_matches_sklearn():
    from tools.myocardial_exploration import fit_logistic
    from sklearn.linear_model import LogisticRegression
    rng=np.random.default_rng(18)
    X=rng.normal(size=(160,5));y=rng.binomial(1,expit(X[:,0]))
    _,_,phi=configuration_prior(160,[40,80,120]);strength=1e-4
    fit=fit_logistic(X,y,phi,'l2',strength)
    for l,w in enumerate(phi):
        other=LogisticRegression(C=1/(2*strength*w.sum()),fit_intercept=False,
             solver='lbfgs',tol=1e-11,max_iter=3000).fit(X,y,sample_weight=w)
        np.testing.assert_allclose(fit['theta'][:,l],other.decision_function(X),atol=2e-6)

def test_finite_iteration_path_retains_requested_budgets():
    from tools.myocardial_exploration import rho_path, ITERATIONS
    _,_,phi=configuration_prior(60,[20,40]);psi=(np.arange(60)>=30).astype(int)
    S=squared_logistic_state(np.eye(2),phi,psi,12)['moment']
    path=rho_path(S,phi,psi,12,'squared')
    assert [r['iteration'] for r in path]==list(ITERATIONS)
    for r in path:
        assert np.linalg.eigvalsh(r['rho']).min()>0
        assert np.isfinite(r['moment_residual'])

def test_finite_state_returns_unconverged_estimate_and_common_draws():
    _,_,phi=configuration_prior(30,[10,20]);psi=(np.arange(30)>=15).astype(int)
    kwargs=dict(sample_power=6,max_nfev=1,seed=12)
    a=logistic_state(np.eye(2),phi,psi,12,**kwargs)
    b=logistic_state(np.eye(2),phi,psi,12,**kwargs)
    np.testing.assert_array_equal(a['parameters'],b['parameters'])
    assert a['training_residual']>1e-5
    assert not a['optimizer_success']
    assert np.isfinite(a['kappa']).all()
