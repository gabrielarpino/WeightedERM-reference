import numpy as np
from WERM.myocardial_direct_moments import decode, solve_direct


def test_decode_is_positive_definite_with_signed_correlation():
    for corr in (-.9,0,.9):
        rho=decode(np.array([np.log(.2),np.log(.4),corr]))
        np.testing.assert_allclose(np.diag(rho),[.2,.4])
        assert np.linalg.eigvalsh(rho).min()>0
        np.testing.assert_allclose(rho[0,1]/np.sqrt(.2*.4),corr)


def test_exact_feasible_moment_retains_correlation():
    target=np.array([[.02,.009],[.009,.03]])
    fit=solve_direct(lambda rho:dict(moment=rho,training_residual=0.),target,
        np.array([np.log(.01),np.log(.01),0.]),max_nfev=100)
    np.testing.assert_allclose(fit['rho'],target,atol=1e-7)
    assert fit['relative_residual']<1e-5


def test_infeasible_negative_moment_is_reported_not_certified():
    fit=solve_direct(lambda rho:dict(moment=rho,training_residual=0.),-np.eye(2),
        np.array([np.log(.01),np.log(.01),0.]),max_nfev=100)
    assert fit['relative_residual']>=1
    assert np.linalg.eigvalsh(fit['rho']).min()>0
