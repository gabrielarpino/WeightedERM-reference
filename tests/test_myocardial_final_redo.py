import numpy as np
from tools.myocardial_final_redo import perturb_rho, normalize_posterior


def test_perturbation_changes_only_eigenvalues():
    rho=np.array([[2.,.3],[.3,1.]])
    np.testing.assert_array_equal(perturb_rho(rho,1.,1.),rho)
    vals,vecs=np.linalg.eigh(rho)
    changed=perturb_rho(rho,.9,1.1)
    np.testing.assert_allclose(vecs.T@changed@vecs,np.diag(vals*[.9,1.1]),atol=1e-14)


def test_null_and_conditional_normalizations_are_distinct():
    prior=np.array([.5,.25,.25]);ll=np.log([2.,1.,3.])
    r=normalize_posterior(ll,prior)
    np.testing.assert_allclose(r['probabilities'],[.5,.125,.375])
    np.testing.assert_allclose(r['conditional'],[.25,.75])
    np.testing.assert_allclose(r['presence'],.5,atol=1e-14)
    shifted=normalize_posterior(ll+10000,prior)
    np.testing.assert_allclose(shifted['conditional'],r['conditional'])
    np.testing.assert_allclose(shifted['presence'],r['presence'])
