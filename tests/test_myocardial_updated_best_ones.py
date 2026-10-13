import numpy as np
from tools.myocardial_updated_best_ones import paired_candidate, fitting_description


def test_pair_metrics_keep_full_reference_support_and_probabilities():
    a=dict(ages=np.array([60.,70.]),conditional=np.array([.25,.75]),P_1=.4)
    b=dict(ages=np.array([60.,70.]),conditional=np.array([.5,.5]),P_1=.6)
    meta=dict(auxiliary='LS',posterior='Logistic',intercept=False,Delta=1,
              rho_updates=3,initialization='identity',stage='test',priority=4)
    row,curve=paired_candidate(a,b,meta,'a.pkl','b.pkl',{50.:.5,60.:.5})
    assert row['mean_TV']==.625
    assert row['numerical_curve_TV']==.25
    assert row['P1_min']==.4 and row['P1_max']==.6
    np.testing.assert_array_equal(curve['check'],b['conditional'])


def test_budget_units_and_matching_knobs_are_explicit():
    assert 'epochs' in fitting_description('sgd',10,.1,123)
    assert 'iterations' in fitting_description('fista',30,1.,42)
    assert 'seed=123' in fitting_description('saga',3,1.,123)
    assert fitting_description('sgd',10,.1,123)!=fitting_description('sgd',10,1.,123)
