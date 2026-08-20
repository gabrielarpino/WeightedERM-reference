import numpy as np

from WERM.experiment_utils import evaluate_changepoints, summarize_setting
from WERM.performance_measures import closest_point_distance


def test_all_nonempty_methods_use_two_sided_hausdorff():
    result = evaluate_changepoints([10, 60, 90], [20, 80], n=100)
    assert result["status"] == "ok"
    assert result["size"] == 3
    assert np.isclose(result["haus"], 0.2)


def test_empty_estimate_gets_bounded_worst_case():
    result = evaluate_changepoints([], [20, 80], n=100)
    assert result["status"] == "empty_estimate"
    assert result["haus"] == 1.0
    assert result["size"] == 0
    assert result["raw_cp"].dtype.kind == "i"


def test_not_applicable_is_excluded_from_metric_and_count():
    result = evaluate_changepoints(None, [20, 80], n=100)
    assert result["status"] == "not_applicable"
    assert np.isnan(result["haus"])
    assert np.isnan(result["size"])
    assert result["raw_cp"] is None


def test_summary_is_unconditional_over_applicable_trials():
    summary = summarize_setting(
        haus=[0.1, 1.0, np.nan],
        sizes=[2, 0, np.nan],
        statuses=["ok", "empty_estimate", "not_applicable"],
    )
    assert np.isclose(summary["mean_haus"], 0.55)
    assert np.isclose(summary["median_haus"], 0.55)
    assert np.isclose(summary["empty_estimate_rate"], 1 / 3)
    assert np.isclose(summary["not_applicable_rate"], 1 / 3)
    assert np.isclose(summary["mean_size"], 1.0)


def test_closest_point_distance_broadcasts_unequal_lengths():
    assert closest_point_distance([5, 20, 50], [9, 41]) == 4
