"""Phase4 natural-sample retrieval with IDF disjoint from OOF folds1-3.

Only unowned and fold0-owned training text fits vocabulary/IDF. All training
sources remain deployment-realistic distractors. Labels are read after retrieval.
"""
from pathlib import Path
import argparse,hashlib,json,resource,time
import duckdb,polars as pl
from src.blocking.char_retrieval import retrieve
from src.blocking.token_candidates import summarize_candidates

def main():
    p=argparse.ArgumentParser();p.add_argument('--sample',default='A');p.add_argument('--kernel',choices=['reference','fused'],default='reference');p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    querypath=Path(f'artifacts/validation/phase4-v001/P4-SAMPLE-{a.sample}.parquet')
    cfg={'kernel':a.kernel,'kernel_threads':2,'queries_path':str(querypath),'fit_owner_folds':[-1,0],'max_features':200000,'chunk_rows':25000,'query_block_rows':50,'top_k_per_route':100,'route_runtime_cap_seconds':7200,'routes':[['name',3],['address',3]],'query_sha256':hashlib.sha256(querypath.read_bytes()).hexdigest(),'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'retriever_sha256':hashlib.sha256(Path('code/business_entity_resolution/src/blocking/char_retrieval.py').read_bytes()).hexdigest()}
    (out/'config.json').write_text(json.dumps(cfg,indent=2))
    frame=pl.read_parquet(querypath)
    assert set(frame['fold'])<= {1,2,3}
    queries=frame.select('entity_id','country',pl.col('n').alias('name'),pl.col('a').alias('address')).to_dicts()
    db=duckdb.connect('artifacts/audit.duckdb',read_only=True,config={'threads':2,'memory_limit':'2GB'})
    routes={};union={q['entity_id']:set() for q in queries}
    for field in ['name','address']:
        rows,meta=retrieve(db,queries,field,3,cfg,out)
        route=pl.DataFrame(rows,schema=['source1_entity_id','target_id','route','route_score','route_rank','representation','target_source'],orient='row')
        route.write_parquet(out/f'{field}_char3.parquet',compression='zstd')
        for q,t,*_ in rows:union[q].add(t)
        routes[field]=meta
        (out/'progress.json').write_text(json.dumps(routes,indent=2));del rows,route
    truth={q['entity_id']:set() for q in queries}
    for q,t in db.execute('SELECT p.source1_entity_id,p.target_id FROM positive_pairs p JOIN read_parquet(?) q ON q.entity_id=p.source1_entity_id',[str(querypath)]).fetchall():truth[q].add(t)
    metric=summarize_candidates(queries,truth,union,10320219)
    report={'config':cfg,'routes':routes,'metrics':metric,'seconds':time.perf_counter()-start,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3,'scope':'Natural prevalence folds1-3; full training target pool; IDF excludes every OOF-owned target; fixed external text fit partition fold0 plus unowned; fold4 closed'}
    (out/'metrics.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
