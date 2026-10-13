import numpy as np
from tools.myocardial_fixedpoint_search import update_rho, correlation


def test_negative_update_is_clipped_not_mixed_with_external_matrix():
    rho=np.eye(2);S=np.eye(2)*.1
    state={'lambda':np.eye(2),'kappa':np.eye(2)}
    new,raw,clipped=update_rho(rho,S,state)
    np.testing.assert_allclose(new,.750025*np.eye(2))
    assert clipped==2 and correlation(new)==0


def test_valid_correlated_update_preserves_data_driven_off_diagonal():
    rho=np.eye(2);S=np.array([[.2,.08],[.08,.3]])
    state={'lambda':np.eye(2),'kappa':np.zeros((2,2))}
    new,raw,clipped=update_rho(rho,S,state)
    np.testing.assert_allclose(new,.75*rho+.25*S)
    assert clipped==0 and correlation(new)>0
