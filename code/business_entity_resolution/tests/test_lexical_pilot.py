import numpy as np
import pytest
from src.evaluation import entity_f05
from src.models.lexical_pilot import text_features,select_threshold


def test_official_precision_example():
    assert entity_f05({'a','b'},{'a','b','c'})==pytest.approx(5/7)


def test_threshold_credits_missing_candidate_singleton_and_penalizes_false_merge():
    # Entity 0 singleton has no candidates, entity 1 has one correct candidate,
    # entity 2 singleton has an incorrect candidate. Exact optimum excludes FP.
    threshold,trials=select_threshold(np.array([.8,.6]),np.array([1,2]),np.array([1,0]),np.array([0,1,0]))
    assert .6<threshold<=.8
    assert max(x['macro_f0_5'] for x in trials)==1


def test_missing_fields_do_not_agree():
    assert text_features('','')==[0]*13
    assert text_features('531 main street','532 main street')[0]==0
