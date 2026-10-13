import numpy as np
import pytest
from scipy.special import expit
from WERM.signal_strength import squared_logistic_state, estimate_signal_strength, empirical_adjustment, configuration_prior, gaussian_logistic_loglikelihood, signal_strength_certificate

def setup():
    n=180
    eta, prior, phi = configuration_prior(n, np.arange(20,170,10))
    psi=(np.arange(n)>=80).astype(int)
    return phi,psi

def test_prior_and_marginals_are_same_distribution():
    eta,prior,phi=configuration_prior(10,[2,5,8])
    np.testing.assert_allclose(prior,[.5,1/6,1/6,1/6])
    states=np.array([np.zeros(10,int) if e==-1 else (np.arange(10)>=e).astype(int) for e in eta])
    np.testing.assert_allclose(phi[1],prior @ states)
    np.testing.assert_allclose(phi.sum(axis=0),1)
    assert phi[0,0]==1 and phi[1,0]==0

def test_square_b_and_adjustment():
    phi,psi=setup(); delta=6
    theta=np.zeros((180,2));y=np.ones(180)
    adjusted,b=empirical_adjustment(theta,y,phi,delta,loss='squared')
    np.testing.assert_allclose(np.mean(1/(1+2*b[:,None]*phi),axis=1),1-1/delta,atol=1e-12)
    np.testing.assert_allclose(adjusted,-2*phi.T*b)

def test_reduced_state_satisfies_original_expectations_by_monte_carlo():
    phi,psi=setup();rho=np.array([[.7,.2],[.2,1.3]]);delta=6
    state=squared_logistic_state(rho,phi,psi,delta)
    rng=np.random.default_rng(16);m=400000
    ix=rng.integers(0,len(psi),m)
    z=rng.multivariate_normal([0,0],rho,m)
    g=rng.multivariate_normal([0,0],state['kappa'],m)
    v=z@state['lambda']+g
    yy=rng.random(m)<expit(z[np.arange(m),psi[ix]])
    a=2*state['b']*phi[:,ix].T/(1+2*state['b']*phi[:,ix].T)
    score=a*(v-yy[:,None])
    np.testing.assert_allclose(z.T@score/m,0,atol=.0015)
    np.testing.assert_allclose(delta*score.T@score/m,state['kappa'],atol=.0015)

def test_inverse_recovers_known_covariance_and_checks_identification():
    phi,psi=setup();rho=np.array([[.7,-.25],[-.25,1.3]])
    s=squared_logistic_state(rho,phi,psi,6)
    fit=estimate_signal_strength(s['moment'],phi,psi,6)
    assert fit['accepted']
    np.testing.assert_allclose(fit['rho'],rho,rtol=2e-5,atol=2e-5)
    assert fit['jacobian_rank']==3
    with pytest.raises(ValueError,match='signal'):
        estimate_signal_strength(s['moment'],phi,np.zeros_like(psi),6)

def test_joint_gaussian_likelihood_uses_lambda_not_cross_covariance():
    # Lambda=0 makes Z independent of V, hence P(Y|V)=1/2.
    from scipy.stats import multivariate_normal
    v=np.array([[.2,-.1],[.5,.3]])
    rho=np.array([[1.,.2],[.2,2.]])
    k=np.array([[.8,.1],[.1,.7]])
    got=gaussian_logistic_loglikelihood(v,np.array([0,1]),rho,np.zeros((2,2)),k,np.array([0,1]))
    expected=multivariate_normal.logpdf(v,mean=[0,0],cov=k).sum()-2*np.log(2)
    assert got==pytest.approx(expected,abs=1e-12)

def test_linear_certificate_matches_nonlinear_forward_map():
    phi,psi=setup();rho=np.array([[.7,-.25],[-.25,1.3]])
    state=squared_logistic_state(rho,phi,psi,6)
    certificate=signal_strength_certificate(state['moment'],phi,psi,6)
    C=state['h'][:,None]*rho*state['h'][None,:]
    np.testing.assert_allclose(certificate['effective_covariance'],C,atol=1e-13)
    assert certificate['feasible'] and certificate['operator_rank']==3
    assert certificate['linear_residual']<1e-13

def test_negative_effective_covariance_is_rejected_even_with_exact_linear_fit():
    phi,psi=setup()
    certificate=signal_strength_certificate(np.eye(2)*1e-5,phi,psi,6)
    assert not certificate['feasible']
    assert certificate['effective_eigenvalues'].min()<0

def test_row_scaling_does_not_change_squared_theory():
    phi,psi=setup();rho=np.array([[.7,.2],[.2,1.3]])
    a=squared_logistic_state(rho,phi,psi,6)
    b=squared_logistic_state(rho,phi*np.array([[3.],[.2]]),psi,6)
    np.testing.assert_allclose(a['moment'],b['moment'],atol=1e-12)

def test_nonzero_lambda_joint_likelihood_matches_scalar_quadrature():
    from scipy.integrate import quad
    from scipy.stats import norm,multivariate_normal
    rho=np.array([[1.,.3],[.3,.8]]);lam=np.array([[.7,.2],[-.1,.6]]);k=np.eye(2)*.2
    V=np.array([[.3,-.7]]);y=np.array([1]);psi=np.array([1])
    S=lam.T@rho@lam+k;cross=rho@lam
    mu=(cross@np.linalg.solve(S,V[0]))[1]
    var=(rho-cross@np.linalg.solve(S,cross.T))[1,1]
    prob=quad(lambda z:expit(mu+np.sqrt(var)*z)*norm.pdf(z),-10,10,epsabs=1e-12)[0]
    expected=multivariate_normal.logpdf(V[0],mean=[0,0],cov=S)+np.log(prob)
    assert gaussian_logistic_loglikelihood(V,y,rho,lam,k,psi)==pytest.approx(expected,abs=1e-10)

def test_logistic_separation_distinguishes_overlap_and_quasiseparation():
    from WERM.signal_strength import logistic_separation
    assert not logistic_separation(np.array([[1.],[1.]]),np.array([0,1]))['separated']
    assert logistic_separation(np.array([[-1.],[1.]]),np.array([0,1]))['separated']
    X=np.array([[1.,0],[-1.,0],[0,1.],[0,1.]])
    y=np.array([1,0,0,1])
    result=logistic_separation(X,y)
    assert result['separated']
    margins=(2*y-1)*(X@result['direction'])
    assert margins.min()>=-1e-10 and margins.sum()>=1-1e-8
    assert not logistic_separation(X,y,np.array([0,0,1,1]))['separated']
