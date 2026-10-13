import numpy as np
from scipy.special import expit
from scipy.special import roots_hermitenorm
from WERM.signal_strength import configuration_prior
from WERM.myocardial_intercept import response_offset, squared_intercept_state, fit_with_intercept, logistic_intercept_state


def test_offset_matches_observed_prevalence():
    z,w=roots_hermitenorm(160)
    for variance in (.01,1.,10.):
        offset=response_offset(variance,.23)
        np.testing.assert_allclose(w@expit(offset+np.sqrt(variance)*z)/np.sqrt(2*np.pi),.23,atol=1e-11)


def test_intercept_fits_are_affine_centering_invariant():
    rng=np.random.default_rng(93);X=rng.normal(size=(180,5));y=rng.binomial(1,expit(-1+X[:,0]))
    _,_,phi=configuration_prior(180,[50,100,130])
    a=fit_with_intercept(X,y,phi,'l2',1e-4)
    b=fit_with_intercept(X+np.arange(1,6),y,phi,'l2',1e-4)
    np.testing.assert_allclose(a['theta'],b['theta'],atol=2e-5)


def test_squared_intercept_state_against_independent_simulation():
    rho=np.array([[.6,.15],[.15,.9]]);rate=.23
    _,_,phi=configuration_prior(100,[20,40,60,80]);psi=(np.arange(100)>=50).astype(int)
    state=squared_intercept_state(rho,phi,psi,12,rate)
    rng=np.random.default_rng(274);size=250000;i=rng.integers(0,100,size)
    Z=rng.multivariate_normal(np.zeros(2),rho,size)
    G=rng.multivariate_normal(np.zeros(2),state['kappa'],size)
    Y=rng.binomial(1,expit(Z[np.arange(size),psi[i]]+state['response_offsets'][psi[i]]))
    V=Z@state['lambda']+G
    a=(2*state['b'][:,None]*phi/(1+2*state['b'][:,None]*phi)).T[i]
    W=a*(V+rate-Y[:,None])
    np.testing.assert_allclose(W.mean(axis=0),0,atol=.001)
    np.testing.assert_allclose(Z.T@W/size,0,atol=.002)
    np.testing.assert_allclose(12*W.T@W/size,state['kappa'],atol=.002)

def test_logistic_intercept_state_includes_intercept_score_equation():
    _,_,phi=configuration_prior(60,[20,40]);psi=(np.arange(60)>=30).astype(int)
    state=logistic_intercept_state(np.array([[.5,.1],[.1,.7]]),phi,psi,12,.23,sample_power=8)
    assert state['training_residual']<1e-6
    assert np.linalg.eigvalsh(state['kappa']).min()>0
    assert len(state['parameters'])==11
