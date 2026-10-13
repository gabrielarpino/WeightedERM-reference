import numpy as np
from scipy.optimize import brentq
from scipy.special import expit
from WERM.myocardial_numerics import logistic_prox_fast

def test_fast_prox_matches_independent_bracketed_roots():
    rng=np.random.default_rng(823)
    v=rng.uniform(-100,100,200);a=10**rng.uniform(-6,5,200)
    for y in (0,1):
        actual=logistic_prox_fast(v,a,y)
        expected=np.array([brentq(lambda t:t+s*(expit(t)-y)-z,z-s*(1-y),z+s*y,xtol=1e-12) for z,s in zip(v,a)])
        np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=2e-10)

def test_fast_prox_matches_original_accuracy_test():
    v=np.linspace(-25,25,101)[:,None];a=np.array([.01,1.,20.])[None,:]
    for y in (0,1):
        t=logistic_prox_fast(v,a,y)
        np.testing.assert_allclose(t+a*(expit(t)-y),np.broadcast_to(v,t.shape),atol=2e-12)
