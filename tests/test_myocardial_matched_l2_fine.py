import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from tools.myocardial_matched_l2_fine import solve_logistic_l2, penalty_grid


def test_full_solve_matches_independent_bfgs_and_is_weight_invariant():
    rng=np.random.default_rng(52);X=rng.normal(size=(80,4))
    y=rng.integers(0,2,size=80);w=rng.uniform(.2,2,size=80);lam=.02
    def fun(b):
        z=X@b
        return (w/w.sum())@(np.logaddexp(0,z)-y*z)+lam*(b@b)
    def jac(b):return X.T@((w/w.sum())*(expit(X@b)-y))+2*lam*b
    reference=minimize(fun,np.zeros(4),jac=jac,method='BFGS',options={'gtol':1e-10})
    fit=solve_logistic_l2(X,y,w,lam)
    np.testing.assert_allclose(fit['coef'],reference.x,atol=2e-7)
    assert fit['kkt']<=1e-10 and fit['gap_bound']<=1e-8
    np.testing.assert_allclose(fit['coef'],solve_logistic_l2(X,y,7*w,lam)['coef'],atol=1e-12)


def test_separable_problem_is_fully_solved_with_positive_ridge():
    X=np.array([[-2.],[-1.],[1.],[2.]])
    fit=solve_logistic_l2(X,np.array([0.,0.,1.,1.]),np.ones(4),1e-8)
    assert np.isfinite(fit['coef']).all() and fit['kkt']<=1e-10


def test_penalty_grid_is_fine_and_positive():
    grid=penalty_grid()
    assert len(grid)==35 and np.all(np.diff(grid)>0)
    assert grid[0]==1e-8 and grid[-1]==1e-3
