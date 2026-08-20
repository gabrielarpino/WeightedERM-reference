import numpy as np


def test_dpdu_selection_is_axis_safe():
    lams, zetas = [0.01, 0.1, 1, 2], [10.0, 15.0, 20.0]
    nz, nl = len(zetas), len(lams)
    for true_z in range(nz):
        for true_l in range(nl):
            te = np.arange(nz * nl, dtype=float).reshape(nz, nl) + 100.0
            te[true_z, true_l] = 0.0
            zi, li = np.unravel_index(int(np.argmin(te)), te.shape)
            assert (zi, li) == (true_z, true_l)
            # R column-major flat index must point at the same cell.
            assert li * nz + zi == true_l * nz + true_z
