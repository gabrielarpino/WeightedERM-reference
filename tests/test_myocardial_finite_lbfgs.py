import numpy as np
from tools.myocardial_finite_focus import finite_lbfgs_column


def test_finite_lbfgs_is_unpenalized_and_budgeted():
    X=np.array([[-2.],[-1.],[1.],[2.]])
    y=np.array([0.,0.,1.,1.]);w=np.array([1.,2.,3.,1.])
    fit=finite_lbfgs_column(X,y,w,3)
    beta=fit.x;z=X@beta
    expected=(w/w.sum())@(np.logaddexp(0,z)-y*z)
    np.testing.assert_allclose(fit.fun,expected,atol=1e-12)
    assert 0 < fit.nit <= 3 and np.isfinite(beta).all()
    again=finite_lbfgs_column(X,y,13*w,3)
    np.testing.assert_allclose(beta,again.x,rtol=1e-10)
