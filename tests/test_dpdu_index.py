import numpy as np

from WERM.comparison import _r_cv_matrix


def test_r_list_matrix_conversion_preserves_r_column_major_layout():
    expected = np.arange(12, dtype=float).reshape((3, 4))
    # rpy2 converts an R list-matrix into a sequence of length-one arrays,
    # ordered down rows within each column, and drops the matrix dimensions.
    converted = [np.array([value]) for value in expected.ravel(order="F")]
    actual = _r_cv_matrix(converted, expected.shape, method_name="DPDU")
    np.testing.assert_array_equal(actual, expected)


def test_numeric_matrix_conversion_keeps_an_existing_2d_layout():
    expected = np.arange(12, dtype=float).reshape((3, 4))
    actual = _r_cv_matrix(expected, expected.shape, method_name="DPDU")
    np.testing.assert_array_equal(actual, expected)


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
