import numpy as np
from scipy.special import expit,logit
from WERM.myocardial_finite_intercept import finite_intercept_column


def test_first_fista_and_gd_steps_match_weighted_gradient():
    rng=np.random.default_rng(14);X=rng.normal(size=(40,3))/np.sqrt(40)
    y=rng.integers(0,2,40);w=rng.uniform(.3,2,40);wn=w/w.sum()
    D=np.column_stack([X,np.ones(40)/np.sqrt(40)])
    initial=np.r_[np.zeros(3),np.sqrt(40)*logit(wn@y)]
    L=np.linalg.eigvalsh(D.T@(wn[:,None]*D))[-1]/4
    expected=initial-D.T@(wn*(expit(D@initial)-y))/L
    for method in ('fista','gd'):
        fit=finite_intercept_column(X,y,w,method,1)
        np.testing.assert_allclose(fit['coef'],expected[:-1],atol=1e-12)
        np.testing.assert_allclose(fit['intercept'],expected[-1]/np.sqrt(40),atol=1e-12)
        assert fit['iterations']==1


def test_intercept_only_fit_and_weight_scale_invariance():
    X=np.zeros((10,2));y=np.r_[np.ones(3),np.zeros(7)];w=np.arange(1.,11.)
    for method in ('fista','gd','lbfgs'):
        fit=finite_intercept_column(X,y,w,method,5)
        np.testing.assert_allclose(expit(fit['intercept']),w@y/w.sum(),atol=1e-10)
        np.testing.assert_allclose(fit['coef'],0,atol=1e-12)
        other=finite_intercept_column(X,y,12*w,method,5)
        np.testing.assert_allclose(fit['intercept'],other['intercept'],atol=1e-10)


def test_finite_separable_fits_have_no_explicit_penalty():
    X=np.array([[-2.],[-1.],[1.],[2.]])/2;y=np.array([0.,0.,1.,1.]);w=np.ones(4)
    for method in ('fista','gd','lbfgs'):
        fit=finite_intercept_column(X,y,w,method,10)
        z=X@fit['coef']+fit['intercept']
        assert np.isfinite(z).all() and fit['iterations']<=10
        np.testing.assert_allclose(fit['objective'],np.mean(np.logaddexp(0,z)-y*z),atol=1e-12)
