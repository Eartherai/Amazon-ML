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


def test_feature_sql_excludes_heldout_targets_from_training(tmp_path):
    import duckdb,polars as pl
    from src.models.lexical_pilot import materialize
    con=duckdb.connect()
    con.execute('CREATE TABLE positive_pairs(source1_entity_id VARCHAR,target_id VARCHAR)')
    con.execute("INSERT INTO positive_pairs VALUES ('S1-1','S2-1')")
    con.execute('CREATE TABLE target_ownership(target_id VARCHAR,owner_fold INTEGER)')
    con.execute("INSERT INTO target_ownership VALUES ('S2-1',1),('S3-2',0)")
    con.execute('CREATE TABLE targets_normalized(entity_id VARCHAR,country VARCHAR,n VARCHAR,a VARCHAR)')
    con.execute("INSERT INTO targets_normalized VALUES ('S2-1','Unseen','alpha','1 road'),('S3-2','Unseen','alpha','2 road')")
    query=tmp_path/'queries.parquet';pl.DataFrame({'entity_id':['S1-1'],'country':['Unseen'],'n':['alpha'],'a':['1 road']}).write_parquet(query)
    for field in ['name','address']:
        pl.DataFrame({'source1_entity_id':['S1-1','S1-1'],'target_id':['S2-1','S3-2'],'route':[field+'_char3']*2,'route_score':[1.,.9],'route_rank':[1,2]}).write_parquet(tmp_path/(field+'_char3.parquet'))
    train=materialize(con,query,tmp_path,tmp_path/'train.parquet',True)
    assert train['target_id'].to_list()==['S2-1']
    assert train['label'].to_list()==[1]
    assert train['retrieval_route_count'].to_list()==[2.]
    dev=materialize(con,query,tmp_path,tmp_path/'dev.parquet',False)
    assert len(dev)==2
