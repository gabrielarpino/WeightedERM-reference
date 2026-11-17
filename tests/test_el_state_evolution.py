"""Independent checks of the SE equations, search objective and original data."""
import ast
import json
from pathlib import Path

import numpy as np
import pytest
from tools import el_state_evolution as experiment


@pytest.mark.parametrize('upper', [False, True])
@pytest.mark.parametrize('count', [1, 3, 6])
def test_search_matches_original_dp(upper, count):
    from WERM.gof import chgpt_gof_dp, manuscript_log_binomial_k_penalty
    rng = np.random.default_rng(11)
    theta, y = rng.normal(size=(42,count)), rng.normal(size=42)
    actual = experiment.search(theta,y,4,upper,1.)
    expected = chgpt_gof_dp(lambda u,v:.5*(u-v)**2,theta,y,42,count,4,
        L_is_upper_bound=upper,penalty=manuscript_log_binomial_k_penalty(42,1.) if upper else None)
    np.testing.assert_array_equal(actual['cp'],expected['changepoints'])
    assert actual['total_loss'] == pytest.approx(expected['total_objective'])


def test_exact_original_data_generator():
    notebook = json.loads(Path('E_L_upper_bound.ipynb').read_text())
    tree = ast.parse(''.join(notebook['cells'][1]['source']))
    function = next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='el_data')
    namespace = {'np':np,'EL_SEED':20261120}
    exec(compile(ast.Module(body=[function],type_ignores=[]),'<original el_data>','exec'),namespace)
    for original,actual in zip(namespace['el_data'](1000,100,3),experiment.generate_data(3)):
        np.testing.assert_array_equal(actual,original)


def test_homogeneous_closed_form_and_loss_scaling():
    # Classical isotropic least squares: lambda=1; K=sigma^2/(delta-1).
    phi=np.ones((1,60)); gamma=np.array([[2.]])
    half=experiment.squared_se(phi,gamma,np.zeros(60,int),3.,.4,1.)
    full=experiment.squared_se(phi,gamma,np.zeros(60,int),3.,.4,2.)
    np.testing.assert_allclose(half['b'],.5,atol=1e-12)
    np.testing.assert_allclose(half['lam'],1.,atol=1e-12)
    np.testing.assert_allclose(half['kappa'],.2,atol=1e-12)
    np.testing.assert_allclose(full['b']*2,half['b'],atol=1e-12)
    for key in ('omega','lam','kappa'):
        np.testing.assert_allclose(full[key],half[key],atol=1e-12)


@pytest.mark.parametrize('count',[2,4,6])
def test_rectangular_se_satisfies_general_moment_equations(count):
    # Independently sample the Gaussian surrogate and evaluate the original
    # score equations, retaining all four true regimes when count < 4.
    rng=np.random.default_rng(83)
    n=40;delta=5.;sigma=.5
    labels=np.repeat(np.arange(4),10)
    phi=rng.uniform(.1,1.,size=(count,n));phi/=phi.sum(axis=0)
    gamma=.6**np.abs(np.arange(4)[:,None]-np.arange(4)[None,:])
    se=experiment.squared_se(phi,gamma,labels,delta,sigma**2)
    assert se['lam'].shape==(4,count)
    assert se['kappa'].shape==(count,count)
    assert max(se['b_residual'],se['bias_residual'],se['covariance_residual'])<1e-11
    samples=12000
    Z=rng.normal(size=(samples,n,4))@np.linalg.cholesky(gamma).T
    G=rng.normal(size=(samples,n,count))@se['covariance_root'].T
    y=Z[:,np.arange(n),labels]+rng.normal(0,sigma,size=(samples,n))
    a=se['b']*phi.T
    V=Z@se['lam']+G
    theta=(V+a*y[:,:,None])/(1+a)
    wb=a*(theta-y[:,:,None])
    bias=np.einsum('sni,snj->ij',Z,wb)/(samples*n)
    covariance=delta*np.einsum('sni,snj->ij',wb,wb)/(samples*n)
    np.testing.assert_allclose(bias,0.,atol=.003)
    np.testing.assert_allclose(covariance,se['kappa'],rtol=.02,atol=.003)
