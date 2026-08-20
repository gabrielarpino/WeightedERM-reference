import numpy as np

import WERM.comparison as comparison


def test_dcdp_scales_selected_gamma_for_full_sample(monkeypatch):
    captured = {}

    class FakeDCDP:
        grid_n = 100

        def fit(self, train, test):
            captured["train_shape"] = train[0].shape
            return np.array([], dtype=int), (0.1, 10.0), np.array([], dtype=int)

        def dp_grid(self, data, grid, lam, gamma):
            captured["full_shape"] = data[0].shape
            captured["lam"] = lam
            captured["gamma"] = gamma
            return np.array([], dtype=int), 0.0

        def local_refine(self, data, cp):
            return []

    def factory(grid_n, lam_list, gamma_list, **kwargs):
        captured["lam_list"] = lam_list
        captured["gamma_list"] = gamma_list
        return FakeDCDP()

    monkeypatch.setattr(comparison.DCDP_utils, "dcdp_cv_grid_linear", factory)

    X = np.zeros((12, 3))
    y = np.arange(12.0)
    out = comparison.DCDP(X, y)

    assert captured["train_shape"] == (6, 1)
    assert captured["full_shape"] == (12, 1)
    assert captured["lam_list"] == [0.01, 0.1, 1, 5, 10]
    assert np.allclose(captured["gamma_list"], np.logspace(0, 3, 7))
    assert captured["lam"] == 0.1
    assert captured["gamma"] == 20.0
    assert out.shape == (0,)
    assert out.dtype == int
