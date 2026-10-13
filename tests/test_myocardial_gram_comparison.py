import numpy as np
from tools.myocardial_gram_comparison import configurations,column
from tools.myocardial_gram_comparison import balanced_segment_weights
import pytest


def test_balanced_weights_equal_class_mass():
    y=np.array([0,0,0,1])
    w=balanced_segment_weights(y)
    assert w.mean()==1
    assert w[y==0].sum()==w[y==1].sum()==2
    with pytest.raises(ValueError):balanced_segment_weights(np.zeros(4))


def test_grid_includes_weak_matched_l2_and_both_losses():
    specs=configurations()
    assert len(specs)==336 and len({s['id'] for s in specs})==336
    l2=[s for s in specs if s['penalty']=='l2']
    assert {s['strength'] for s in l2}=={1e-8,1e-6,1e-5,1e-4,1e-3}
    assert {s['aux_loss'] for s in l2}=={'squared','logistic'}


def test_squared_l2_matches_stated_normalization_and_excludes_intercept():
    rng=np.random.default_rng(42);X=rng.normal(size=(30,3));y=rng.normal(size=30);w=rng.uniform(.2,1,30)
    lam=.01;spec=dict(intercept=True,penalty='l2',strength=lam,method='full',budget=0)
    got=column(X,y,w,spec,'squared');D=np.column_stack([X,np.ones(30)]);wn=w/w.sum()
    expected=np.linalg.solve(D.T@(wn[:,None]*D)+lam*np.diag([1,1,1,0]),D.T@(wn*y))
    np.testing.assert_allclose(np.r_[got['coef'],got['intercept']],expected,atol=1e-10)


def test_divisors_change_scale_not_correlation():
    B=np.array([[1.,2.],[2.,1.],[.1,.3]])
    n,p=100,len(B);Gn=B.T@B/n;Gp=B.T@B/p
    np.testing.assert_allclose(Gp,(n/p)*Gn)
    np.testing.assert_allclose(Gp[0,1]/np.sqrt(Gp[0,0]*Gp[1,1]),Gn[0,1]/np.sqrt(Gn[0,0]*Gn[1,1]))
