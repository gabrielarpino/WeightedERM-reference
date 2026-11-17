"""Compare the accelerated CV implementation to the production helper."""
import numpy as np
import pytest
from scipy.special import comb
from tools import werm_cv_empirical as fast


@pytest.mark.parametrize('loss_scale,delta_fraction',[(1.,.1),(2.,.025)])
def test_cv_matches_production(monkeypatch,loss_scale,delta_fraction):
    from WERM.experiment_utils import weighted_erm_joint_cv
    import WERM.config
    rng=np.random.default_rng(519)
    n,p=180,5
    X=rng.normal(size=(n,p))/np.sqrt(n)
    beta=rng.normal(size=(p,3))*np.sqrt(n/p)
    labels=np.searchsorted([45,117],np.arange(n),side='right')
    y=np.sum(X*beta[:,labels].T,axis=1)+rng.normal(scale=.3,size=n)
    weights={}
    for size in (n,n*9//10):
        t=(np.arange(size)+.5)/size
        for count in range(1,5):
            weights[size,count]=np.array([comb(count-1,j)*t**j*(1-t)**(count-1-j)
                                          for j in range(count)])
    def frozen(Lmin,Lmax,Delta,n_small=50,n_large=1000,p_l=None,verbose=False):
        assert Lmin==Lmax and p_l is None
        return weights[n_large,Lmax]
    monkeypatch.setattr(WERM.config,'extrapolated_marginals',frozen)
    def loss(a,b):
        return loss_scale*.5*(a-b)**2
    expected,cp,diagnostic=weighted_erm_joint_cv(X,y,loss,lambda a,b:np.abs(a-b),
        Lmax=4,delta_fraction=delta_fraction,l1_constants=(0.,),penalty_norm='none',
        coefficient_loss='least_squares',num_folds=10,n_jobs=1,
        standardize_y=False,return_diagnostics=True)
    actual=fast.run_cv(X,y,4,weights,delta_fraction,loss_scale=loss_scale,
                       return_coefficients=True)
    np.testing.assert_allclose(actual['cv_scores'],diagnostic['scores'],rtol=1e-10,atol=1e-9)
    assert actual['selected_by_bound'][4]['selected_segments']==expected[0]
    np.testing.assert_array_equal(actual['selected_by_bound'][4]['cp'],cp)
    for bound in range(1,5):
        assert actual['selected_by_bound'][bound]['selected_segments']==np.argmin(actual['cv_scores'][:bound])+1
    assert fast.gaussian_validation_risk(actual,beta,labels,.3**2).shape==(4,)
