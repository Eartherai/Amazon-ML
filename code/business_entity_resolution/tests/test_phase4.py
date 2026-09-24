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

def test_fused_topk_resolves_boundary_ties_lexically():
    from scipy.sparse import csr_matrix
    from src.blocking.char_retrieval import fused_top_k_rows
    queries=csr_matrix(np.array([[1.,0.],[0.,0.]],dtype=np.float32))
    targets=csr_matrix(np.array([[1.,0.],[1.,0.],[1.,0.],[0.,1.]],dtype=np.float32))
    result=fused_top_k_rows(queries,targets,np.array(['z','a','b','c']),2)
    assert result[0][0].tolist()==['a','b']
    assert result[1][0].tolist()==[]

def test_query_context_is_local_and_zero_safe():
    import polars as pl
    from src.models.phase4_ablation import add_query_context,CONTEXT_BASE
    f=pl.DataFrame({'source1_entity_id':['a','a','b'],**{n:[.2,.8,0.] for n in CONTEXT_BASE}})
    result=add_query_context(f)
    assert result['name_jw_query_max'].to_list()==[.8,.8,0.]
    assert result['name_jw_query_relative'].to_list()==[.25,1.,0.]
    assert result['name_jw_query_gap'].to_list()[1:]==[0.,0.]

def test_transliteration_is_symmetric_for_accented_source():
    from src.transliteration_features import compare_transliterated
    mapping={'café du port':'cafe du port'}
    assert compare_transliterated('café du port','cafe du port',mapping)==[1.,1.,1.,1.]
    assert compare_transliterated('cafe du port','café du port',mapping)==[1.,1.,1.,1.]
    assert compare_transliterated('','café du port',mapping)==[0.,0.,0.,0.]

def test_transliteration_missing_nonascii_mapping_fails_loudly():
    import pytest
    from src.transliteration_features import compare_transliterated
    with pytest.raises(ValueError,match='missing'):
        compare_transliterated('école','ecole',{})

def test_canonical_numeric_is_alternative_not_raw_rewrite():
    from src.models.phase4_numeric import canonical_numbers,compare_numbers
    assert canonical_numbers('flat ०१२३ unit 000')==['123','0']
    assert compare_numbers('206 main','0206 main')==[1.,1.,0.,1.,1.,0.]
    assert compare_numbers('','')==[0.,0.,0.,0.,0.,0.]

def test_xgboost_native_serialization_preserves_probability(tmp_path):
    import xgboost as xgb
    from src.models.family_adapter import Adapter
    rng=np.random.default_rng(17);x=rng.normal(size=(80,4)).astype(np.float32);y=(x[:,0]>0).astype(int)
    m=Adapter('xgboost');m.model.set_params(n_estimators=5,min_child_weight=1);m.fit(x,y,feature_name=['a','b','c','d']);before=m.predict(x);m.save_model(tmp_path/'model.txt')
    restored=xgb.Booster(model_file=tmp_path/'model.ubj');after=restored.predict(xgb.DMatrix(x))
    np.testing.assert_allclose(before,after,rtol=0,atol=0)
    assert np.all((before>0)&(before<1))
