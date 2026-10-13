import numpy as np
from tools.myocardial_best_ones import curve_metrics, rank_candidates


def test_reference_mass_outside_support_counts_toward_distance():
    m=curve_metrics(np.array([60.]),np.array([1.]),{40.:.5,60.:.5})
    assert m['TV']==.5 and m['above55']==1


def test_rank_has_no_probability_cutoff_and_removes_duplicate_configurations():
    rows=[dict(configuration='a',mean_TV=.1,max_TV=.12,P1_min=.2,priority=2),
          dict(configuration='a',mean_TV=.1,max_TV=.12,P1_min=.2,priority=1),
          dict(configuration='b',mean_TV=.3,max_TV=.32,P1_min=.9,priority=2)]
    ranked=rank_candidates(rows)
    assert list(ranked.configuration)==['a','b']
    assert list(ranked['rank'])==[1,2]
