"""Evaluate completed candidate routes, incremental unions, removals and slices.

Labels are loaded only for evaluation. All metrics refer to the same frozen
1000-S1 development pilot with the full training target pool.
"""
from pathlib import Path
import argparse,json,time
import duckdb
import numpy as np
import polars as pl
from src.blocking.token_candidates import summarize_candidates


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--translit',type=Path,default=Path('outputs/candidates/TRANS-002/name_translit_char3.parquet'));a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();db=duckdb.connect('artifacts/audit.duckdb',read_only=True,config={'threads':2,'memory_limit':'2GB'})
    querypath='outputs/candidates/TOKEN-001/run-002/pilot_queries.parquet';qdf=pl.read_parquet(querypath)
    queries=[dict(zip(('entity_id','country','name','address'),r)) for r in qdf.select('entity_id','country','n','a').iter_rows()];truth={q['entity_id']:set() for q in queries}
    db.execute('CREATE TEMP TABLE q AS SELECT * FROM read_parquet(?)',[querypath]);cursor=db.execute('''SELECT p.source1_entity_id,p.target_id,f.nonascii_target_name,f.missing_address,f.name_jw,f.address_jw,f.numeric_both_present,f.numeric_overlap FROM positive_pairs p JOIN q ON q.entity_id=p.source1_entity_id JOIN positive_features f USING(source1_entity_id,target_id)''')
    columns=[c[0] for c in cursor.description];positive=[dict(zip(columns,r)) for r in cursor.fetchall()]
    for r in positive:truth[r['source1_entity_id']].add(r['target_id'])
    paths={'token_union':Path('outputs/candidates/TOKEN-001/run-002/final_candidates.parquet')}
    for field,n in [('name',3),('address',3),('name',4),('address',4),('name',5),('address',5)]:
        path=Path(f'outputs/candidates/CHAR-001/{field}_char{n}.parquet')
        if path.exists():paths[field+f'_char{n}']=path
    if a.translit.exists():paths['name_translit_char3']=a.translit
    routes={};provenance=[]
    for route,path in paths.items():
        frame=pl.read_parquet(path);routepairs=set(frame.select('source1_entity_id','target_id').iter_rows());routes[route]=routepairs
        for q,t in routepairs:provenance.append((q,t,route))
    def metric(pairs):
        sets={q:set() for q in truth}
        for q,t in pairs:sets[q].add(t)
        result=summarize_candidates(queries,truth,sets,10320219)
        result['candidate_quantiles']['p90']=float(np.quantile([len(x) for x in sets.values()],.9))
        return result,sets
    incremental=[];union=set()
    for route,pairs in routes.items():
        before=len(union);oldhits=sum((q,t) in union for q,ts in truth.items() for t in ts);union|=pairs;m,_=metric(union)
        incremental.append({'added_route':route,'new_pairs':len(union)-before,'new_true_links':m['retrieved_links']-oldhits,**m})
    final,sets=metric(union);removals=[]
    for route in routes:
        other=set().union(*(pairs for name,pairs in routes.items() if name!=route));m,_=metric(other);removals.append({'removed_route':route,'lost_true_links':final['retrieved_links']-m['retrieved_links'],'removed_pairs':len(union)-len(other),**m})
    slices=[]
    conditions={
      'singleton':lambda q:len(truth[q['entity_id']])==0,
      '1_match':lambda q:len(truth[q['entity_id']])==1,
      '2_matches':lambda q:len(truth[q['entity_id']])==2,
      '3_to_5_matches':lambda q:3<=len(truth[q['entity_id']])<=5,
      '6_plus_matches':lambda q:len(truth[q['entity_id']])>=6,
      'short_name_le15':lambda q:len(q['name'])<=15,
      'long_name_ge35':lambda q:len(q['name'])>=35,
      'short_address_le30':lambda q:len(q['address'])<=30,
      'long_address_ge80':lambda q:len(q['address'])>=80,
      'missing_query_name':lambda q:not q['name'],
      'missing_query_address':lambda q:not q['address']}
    for key,predicate in conditions.items():
        qs=[q for q in queries if predicate(q)];ids={q['entity_id'] for q in qs}
        if not ids:slices.append({'slice':key,'query_count':0});continue
        slices.append({'slice':key,**summarize_candidates(qs,{q:truth[q] for q in ids},{q:sets[q] for q in ids},10320219)})
    pairslices=[]
    for key,predicate in {'ascii_target_name':lambda r:not r['nonascii_target_name'],'nonascii_target_name':lambda r:r['nonascii_target_name'],'missing_pair_address':lambda r:r['missing_address'],'hard_both_jw_lt_07':lambda r:r['name_jw']<.7 and r['address_jw']<.7,'numeric_conflict':lambda r:r['numeric_both_present'] and not r['numeric_overlap']}.items():
        subset=[r for r in positive if predicate(r)];hits=sum((r['source1_entity_id'],r['target_id']) in union for r in subset);pairslices.append({'slice':key,'true_links':len(subset),'retrieved_links':hits,'link_recall':hits/len(subset) if subset else None})
    # Stratified cluster bootstrap: entities, not pairs, are resampled; sample composition stays 50/50 country.
    rng=np.random.default_rng(20260925);bootstrap=[]
    for _ in range(1000):
        ids=[]
        for country in sorted({q['country'] for q in queries}):
            group=[q['entity_id'] for q in queries if q['country']==country];ids.extend(rng.choice(group,len(group),replace=True))
        denominator=sum(len(truth[q]) for q in ids);bootstrap.append(sum(len(truth[q]&sets[q]) for q in ids)/denominator)
    unionframe=pl.DataFrame(sorted(union),schema=['source1_entity_id','target_id'],orient='row');unionframe.write_parquet(a.output/'candidates.parquet')
    pl.DataFrame(provenance,schema=['source1_entity_id','target_id','route'],orient='row').write_parquet(a.output/'route_membership.parquet')
    report={'query_scope':'1000 balanced-country dev fold0 entities; no heldout score','target_pool':10320219,'routes':{name:str(path) for name,path in paths.items()},'final':final,'incremental':incremental,'leave_one_route_out':removals,'entity_slices':slices,'positive_link_slices':pairslices,'link_recall_cluster_bootstrap_95pct':np.quantile(bootstrap,[.025,.975]).tolist(),'bootstrap_scope':'1000 stratified entity resamples; diagnostic uncertainty only, ignores adaptive architecture selection and country population shift','runtime_seconds':time.perf_counter()-start}
    (a.output/'metrics.json').write_text(json.dumps(report,indent=2));db.close()
    print(json.dumps({'routes':list(routes),'recall':final['link_recall'],'oracle':final['oracle_macro_f0_5'],'complete':final['positive_entity_all_coverage'],'pairs':len(union)}))
if __name__=='__main__':main()
