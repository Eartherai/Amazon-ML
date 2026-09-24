"""Exercise persistent shard merging against dense reference with unknown countries."""
import json,os,subprocess,sys
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from src.blocking.char_retrieval import top_k


def test_index_roundtrip_and_merge(tmp_path):
    inp=tmp_path/'input';inp.mkdir()
    targets=pl.DataFrame({'entity_id':[f'S{2+i%2}-{i:04d}' for i in range(130)],'country':['Unseen-É']*130,'n':['identical company']*120+['different name']*10,'a':['12 street']*100+['42 road']*30})
    targets.write_parquet(inp/'targets.parquet')
    queries=pl.DataFrame({'entity_id':['S1-q'],'country':['Unseen-É'],'n':['identical company'],'a':['12 street']});queries.write_parquet(inp/'queries.parquet')
    for field,col in [('name','n'),('address','a')]:
        v=TfidfVectorizer(analyzer='char',ngram_range=(3,3),lowercase=False,dtype=np.float32).fit(targets[col].to_list())
        np.savez(inp/f'{field}_char3_idf.npz',idf=v.idf_,vocabulary=v.get_feature_names_out())
        score=(v.transform(queries[col].to_list())@v.transform(targets[col].to_list()).T).toarray()[0]
        ids,_=top_k(targets['entity_id'].to_numpy(),score,100)
        pl.DataFrame({'source1_entity_id':['S1-q']*len(ids),'target_id':ids}).write_parquet(inp/f'{field}_char3.parquet')
    (inp/'manifest.json').write_text('{}')
    out=tmp_path/'out'
    proc=subprocess.run([sys.executable,'scripts/aws/benchmark_indexes.py','--input',str(inp),'--output',str(out),'--chunk-rows','17'],env=os.environ,capture_output=True,text=True)
    assert proc.returncode==0,proc.stderr
    result=json.loads((out/'metrics.json').read_text())
    assert len(result['routes'])==2
    assert all(t['candidate_set_mismatches']==0 for r in result['routes'] for t in r['trials'])
    assert all(r['country']=='Unseen-É' for r in result['routes'])
