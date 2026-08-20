import numpy as np

import WERM.comparison as comparison


class FakeCharcoal:
    def __init__(self, final):
        self.final = final
        self.threshold_calls = []
        self.run_calls = []

    def getTestThreshold(self, n, p, **kwargs):
        self.threshold_calls.append((n, p, kwargs))
        return [[37.0]]

    def not_cpreg(self, X, y, **kwargs):
        self.run_calls.append(kwargs)
        return [None, self.final]


def test_charcoal_uses_and_reuses_package_default_threshold(monkeypatch):
    fake = FakeCharcoal(final=[])
    monkeypatch.setattr(comparison, "charcoal", fake)
    comparison._CHARCOAL_THRESH_CACHE.clear()
    X = np.zeros((10, 3))
    y = np.zeros(10)

    first = comparison.run_charcoal(X, y, 99.0, np.array([5]))
    second = comparison.run_charcoal(X, y, 99.0, np.array([5]))

    assert first.shape == second.shape == (0,)
    assert first.dtype == second.dtype == int
    assert len(fake.threshold_calls) == 1
    _, _, kwargs = fake.threshold_calls[0]
    assert kwargs["alpha"] == 0.01 / (10 // 5)
    assert kwargs["permSize"] == 1000
    assert kwargs["sigma"] == 1.0
    assert kwargs["verbose"] is False
    assert [call["thresh"] for call in fake.run_calls] == [37.0, 37.0]
    assert all(call["verbose"] is False for call in fake.run_calls)


def test_charcoal_maps_top_level_r_null_to_empty_estimate(monkeypatch):
    fake = FakeCharcoal(final=[])
    fake.not_cpreg = lambda X, y, **kwargs: comparison.ro.NULL
    monkeypatch.setattr(comparison, "charcoal", fake)
    comparison._CHARCOAL_THRESH_CACHE.clear()

    cp = comparison.run_charcoal(
        np.zeros((10, 3)), np.zeros(10), 1.0, np.array([5])
    )

    assert cp.shape == (0,)
    assert cp.dtype == int



def test_charcoal_distinguishes_inapplicable_and_nonempty(monkeypatch):
    fake = FakeCharcoal(final=[[4], [7]])
    monkeypatch.setattr(comparison, "charcoal", fake)
    comparison._CHARCOAL_THRESH_CACHE.clear()

    assert comparison.run_charcoal(
        np.zeros((3, 3)), np.zeros(3), 1.0, np.array([1])
    ) is None
    cp = comparison.run_charcoal(
        np.zeros((10, 3)), np.zeros(10), 1.0, np.array([5])
    )
    assert np.array_equal(cp, np.array([4, 7]))
    assert cp.dtype == int
