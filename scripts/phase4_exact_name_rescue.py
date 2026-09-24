"""Observable name-collision rescue: rerank within exact-name groups by address."""
from pathlib import Path
import json,time
import duckdb,polars as pl
from rapidfuzz.fuzz import WRatio
from src.blocking.token_candidates import summarize_candidates
out=Path('outputs/candidates/P4-RESCUE-001');out.mkdir(parents=True,exist_ok=False);start=time.perf_counter();db=duckdb.connect('artifacts/audit.duckdb',read_only=True,config={'threads':2,'memory_limit':'1GB'});qpath='artifacts/validation/phase4-v001/P4-SAMPLE-A.parquet';q=pl.read_parquet(qpath);db.execute('CREATE TEMP TABLE q AS SELECT * FROM read_parquet(?)',[qpath]);cur=db.execute("SELECT q.entity_id,t.entity_id target_id,q.a,t.a target_address FROM q JOIN targets_normalized t ON t.n=q.n AND t.country=q.country WHERE q.n<>'' ORDER BY q.entity_id,t.entity_id")
groups={};raw=0
while rows:=cur.fetchmany(25000):
 for sid,tid,a,b in rows:groups.setdefault(sid,[]).append((tid,WRatio(a,b)/100 if a and b else 0.));raw+=1
rescue=[];triggered=0
for sid,rows in groups.items():
 if len(rows)<=100:continue
 triggered+=1
 for rank,(tid,score) in enumerate(sorted(rows,key=lambda r:(-r[1],r[0]))[:20]):rescue.append((sid,tid,'exact_name_address_rescue',score,rank+1))
rf=pl.DataFrame(rescue,schema=['source1_entity_id','target_id','route','route_score','route_rank'],orient='row');rf.write_parquet(out/'rescue.parquet');base=pl.concat([pl.read_parquet(f'outputs/candidates/P4-A-001/{field}_char3.parquet').select('source1_entity_id','target_id') for field in ['name','address']]).unique();extra=rf.select('source1_entity_id','target_id').join(base,on=['source1_entity_id','target_id'],how='anti');extra.write_parquet(out/'new_candidates.parquet');union=pl.concat([base,extra]);candidates={i:set() for i in q['entity_id']};truth={i:set() for i in q['entity_id']}
for s,t in union.iter_rows():candidates[s].add(t)
for s,t in db.execute('SELECT p.source1_entity_id,p.target_id FROM positive_pairs p JOIN q ON q.entity_id=p.source1_entity_id').fetchall():truth[s].add(t)
metrics=summarize_candidates(q.select('entity_id','country',pl.col('n').alias('name'),pl.col('a').alias('address')).to_dicts(),truth,candidates,10320219);old=json.loads(Path('outputs/candidates/P4-A-001/metrics.json').read_text())['metrics'];report={'trigger':'More than100 full-pool exact normalized name+country targets; no labels used','ranking':'address RapidFuzz WRatio descending, target ID tie-break; top20','queries_triggered':triggered,'exact_name_pairs_scored':raw,'new_candidates':len(extra),'new_true_links':metrics['retrieved_links']-old['retrieved_links'],'new_complete_entities':round((metrics['positive_entity_all_coverage']-old['positive_entity_all_coverage'])*(len(q)-old['singleton_count'])),'metrics':metrics,'seconds':time.perf_counter()-start,'scope':'Development hypothesis derived from observed5k misses; not independent evaluation and not matcher result'};(out/'metrics.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
