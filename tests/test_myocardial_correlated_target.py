import numpy as np
from tools.myocardial_correlated_target import blend_covariance


def test_shape_blend_has_fixed_trace_and_nonzero_correlation():
    base=.003*np.eye(2);gram=np.array([[.02,.01],[.01,.03]])
    out=blend_covariance(base,gram,.2,True)
    np.testing.assert_allclose(np.trace(out),np.trace(base))
    assert out[0,1]>0 and np.linalg.eigvalsh(out).min()>0


def test_raw_blend_is_not_renormalized_and_zero_is_baseline():
    base=.003*np.eye(2);gram=np.array([[.02,.01],[.01,.03]])
    np.testing.assert_array_equal(blend_covariance(base,gram,0,False),base)
    np.testing.assert_allclose(blend_covariance(base,gram,.5,False),.5*(base+gram))
