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


def test_charcoal_uses_documented_fixed_threshold_without_oracle_sigma(monkeypatch):
    fake = FakeCharcoal(final=[])
    monkeypatch.setattr(comparison, "charcoal", fake)
    X = np.zeros((10, 3))
    y = np.zeros(10)

    first = comparison.run_charcoal(X, y)
    second = comparison.run_charcoal(X, y)

    assert first.shape == second.shape == (0,)
    assert first.dtype == second.dtype == int
    assert fake.threshold_calls == []
    assert [call["thresh"] for call in fake.run_calls] == [10.0, 10.0]
    assert all("sigma" not in call for call in fake.run_calls)
    assert [call["seed"] for call in fake.run_calls] == [42, 42]
    assert all(call["verbose"] is False for call in fake.run_calls)


def test_charcoal_maps_top_level_r_null_to_empty_estimate(monkeypatch):
    fake = FakeCharcoal(final=[])
    fake.not_cpreg = lambda X, y, **kwargs: comparison.ro.NULL
    monkeypatch.setattr(comparison, "charcoal", fake)
    comparison._CHARCOAL_THRESH_CACHE.clear()

    cp = comparison.run_charcoal(np.zeros((10, 3)), np.zeros(10))

    assert cp.shape == (0,)
    assert cp.dtype == int



def test_charcoal_distinguishes_inapplicable_and_nonempty(monkeypatch):
    fake = FakeCharcoal(final=[[4], [7]])
    monkeypatch.setattr(comparison, "charcoal", fake)
    comparison._CHARCOAL_THRESH_CACHE.clear()

    assert comparison.run_charcoal(np.zeros((3, 3)), np.zeros(3)) is None
    cp = comparison.run_charcoal(np.zeros((10, 3)), np.zeros(10))
    assert np.array_equal(cp, np.array([4, 7]))
    assert cp.dtype == int
