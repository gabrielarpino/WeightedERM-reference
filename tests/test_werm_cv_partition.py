"""Exhaustive oracle checks of optional endpoint constraints."""
from itertools import combinations

import numpy as np
import pytest

from tools.werm_cv_partition import additive_partition
from tools.m2_adjusted_experiment import squared_partition


@pytest.mark.parametrize('n,count,gap,end_min',[
    (12,4,2,1),(12,4,3,3),(13,3,4,2),(8,2,5,4),(7,1,3,4),
])
def test_partition_matches_exhaustive(n,count,gap,end_min):
    rng=np.random.default_rng(n+count)
    losses=rng.normal(size=(n,count))
    actual=additive_partition(losses,gap,end_min)
    valid=[]
    for cp in combinations(range(1,n),count-1):
        sizes=np.diff([0,*cp,n])
        if sizes[0]<end_min or sizes[-1]<end_min or np.any(sizes[1:-1]<gap):
            continue
        labels=np.searchsorted(cp,np.arange(n),side='right')
        valid.append((float(losses[np.arange(n),labels].sum()),cp))
    expected=min(valid,key=lambda item:item[0])[1]
    np.testing.assert_array_equal(actual,expected)


def test_default_endpoint_constraint_matches_canonical_dp():
    rng=np.random.default_rng(52)
    for n in (20,31):
        for count in range(1,6):
            theta=rng.normal(size=(n,count));y=rng.normal(size=n)
            np.testing.assert_array_equal(additive_partition((theta-y[:,None])**2,3),
                                          squared_partition(theta,y,3))
    np.testing.assert_array_equal(additive_partition(np.zeros((12,4)),3,3),[3,6,9])


def test_infeasible_constraints_raise():
    with pytest.raises(ValueError,match='Infeasible'):
        additive_partition(np.zeros((11,4)),3,3)
