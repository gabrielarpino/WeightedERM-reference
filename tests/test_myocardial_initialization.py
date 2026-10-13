import numpy as np
from tools.myocardial_initialization_sweep import initialization_grid


def test_initializations_are_spd_symmetric_and_include_identity():
    grid=initialization_grid()
    assert len(grid)==9
    assert len({name for name,rho in grid})==9
    assert any(np.array_equal(rho,np.eye(2)) for name,rho in grid)
    for name,rho in grid:
        np.testing.assert_array_equal(rho,rho.T)
        assert np.linalg.eigvalsh(rho).min()>0
    assert {round(rho[0,1]/rho[0,0],6) for name,rho in grid}=={-.5,0.,.5}
