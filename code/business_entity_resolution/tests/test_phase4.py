"""Phase4 validation invariants independent of competition records."""
import numpy as np
from src.models.phase4_oof import fitting_mask
from src.models.lexical_pilot import select_threshold

def test_owned_heldout_negative_never_fits():
    q=np.array([1,1,1,1,2,3]);owner=np.array([1,2,4,-1,2,3])
    assert fitting_mask(q,owner,[1]).tolist()==[True,False,False,True,False,False]
    assert fitting_mask(q,owner,[1,3]).tolist()==[True,False,False,True,False,True]

def test_threshold_counts_unretrieved_truth_and_empty_queries():
    # First entity has two true links, one retrieved; second is a singleton;
    # third has an unretrieved link. At a threshold > .2 only one TP remains.
    threshold,trials=select_threshold(np.array([.8,.2]),np.array([0,1]),np.array([1,0]),np.array([2,0,1]))
    assert .2 < threshold <= .8
    assert abs(max(t['macro_f0_5'] for t in trials)-(1.25/1.5+1)/3)<1e-12

def test_locked_fold_cannot_fit_idf(tmp_path):
    import pytest
    from src.blocking.char_retrieval import retrieve
    with pytest.raises(ValueError,match='locked fold4'):
        retrieve(None,[],'name',3,{'fit_owner_folds':[4]},tmp_path)

def test_joint_empty_threshold_keeps_missing_entities():
    from src.models.phase4_decisions import empty_search
    (pair,empty),trials=empty_search(np.array([.8,.3]),np.array([0,1]),np.array([1,0]),np.array([1,0,1]))
    assert empty>=pair
    assert max(r['macro_f0_5'] for r in trials)==2/3
