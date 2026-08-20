import pytest

from WERM.comparison import _moseg_default_Gset, _moseg_getG


@pytest.mark.parametrize(
    ("n", "expected_g", "expected_gset"),
    [
        (200, 73, [73, 97, 100]),
        (600, 101, [101, 135, 168]),
        (1200, 121, [121, 161, 202]),
        (1800, 133, [133, 177, 222]),
    ],
)
def test_moseg_package_default_bandwidths(n, expected_g, expected_gset):
    assert _moseg_getG(200, n) == expected_g
    assert _moseg_default_Gset(200, n) == expected_gset


def test_moseg_gset_is_sorted_unique_and_clipped():
    gset = _moseg_default_Gset(200, 20)
    assert gset == sorted(set(gset))
    assert all(0 < g <= 10 for g in gset)
