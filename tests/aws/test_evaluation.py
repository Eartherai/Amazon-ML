"""Test macro oracle, duplicate route union, full coverage, and Fold4 gate."""
import json,subprocess,sys
import polars as pl
import pytest

@pytest.mark.parametrize('locked',[False,True])
def test_evaluation(tmp_path,locked):
    c=tmp_path/'c';c.mkdir();labels=tmp_path/'labels';labels.mkdir()
    pl.DataFrame({'entity_id':['q1','q2','q3','q4'],'country':['X']*4,'shard':[0]*4}).write_parquet(c/'query_coverage.parquet')
    pl.DataFrame({'source1_entity_id':['q1','q1','q3','q3','q4'],'target_id':['S2-a','S2-a','S2-c','S2-z','S2-d']}).write_parquet(c/'name-c000-s000-b000.parquet')
    pl.DataFrame({'source1_entity_id':['q1','q2','q3'],'country':['X']*3,'fold':[4 if locked else 1,2,3],'n_matches':[2,0,1]}).write_parquet(labels/'queries.parquet')
    pl.DataFrame({'source1_entity_id':['q1','q1','q3'],'target_id':['S2-a','S2-b','S2-c']}).write_parquet(labels/'truth.parquet')
    out=tmp_path/'out';r=subprocess.run([sys.executable,'scripts/aws/evaluate_retrieval.py','--candidates',str(c),'--labels',str(labels),'--output',str(out)],capture_output=True,text=True)
    if locked:
        assert r.returncode!=0 and 'Locked/invalid labels' in r.stderr
    else:
        assert r.returncode==0,r.stderr
        result=json.loads((out/'metrics.json').read_text())
        assert result['evaluation']['link_recall']==pytest.approx(2/3)
        assert result['evaluation']['oracle_macro_f0_5']==pytest.approx((1.25/1.5+1+1)/3)
        assert result['unlabeled_full_coverage']['full_s1_count']==4
        assert result['unlabeled_full_coverage']['total_candidate_pairs']==4
