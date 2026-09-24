"""Independent full-target transliterated-name character retrieval route."""
import json,time,resource,hashlib,subprocess
from pathlib import Path
import duckdb,numpy as np,polars as pl
from src.blocking.char_retrieval import retrieve
from src.blocking.token_candidates import summarize_candidates
from src.preprocessing import normalize,BASE


def main():
    out=Path('outputs/candidates/TRANS-002');out.mkdir(parents=True,exist_ok=False)
    config=json.loads(Path('configs/blocking/CHAR-001.json').read_text());config['target_table']='translit_targets';config['routes']=[['name',3]];config['cache_manifest_sha256']=hashlib.sha256(Path('artifacts/transliteration/TRANS-001/manifest.json').read_bytes()).hexdigest()
    con=duckdb.connect('artifacts/audit.duckdb',read_only=True,config={'threads':2,'memory_limit':'2GB'})
    con.execute("CREATE TEMP VIEW translit_targets AS SELECT t.entity_id,t.country,coalesce(m.transliterated,t.n) n,t.a FROM targets_normalized t LEFT JOIN read_parquet('artifacts/transliteration/TRANS-001/name_map.parquet') m ON t.n=m.n")
    qr=pl.read_parquet(config['queries_path']);queries=[dict(zip(('entity_id','country','name','address'),r)) for r in qr.select('entity_id','country','n','a').iter_rows()]
    strings=[q['name'] for q in queries];r=subprocess.run([str(Path('artifacts/transliterate_probe').resolve())],input=json.dumps(strings)+'\n',capture_output=True,text=True,check=True)
    for q,t in zip(queries,json.loads(r.stdout)):q['name']=normalize(t,BASE)
    start=time.perf_counter();rows,metadata=retrieve(con,queries,'name',3,config,out)
    data=pl.DataFrame(rows,schema=['source1_entity_id','target_id','route','route_score','route_rank','representation','target_source'],orient='row').with_columns(pl.lit('name_translit_char3').alias('route'),pl.lit('foundation_any_latin_latin_ascii').alias('representation'))
    data.write_parquet(out/'name_translit_char3.parquet')
    con.execute('CREATE TEMP TABLE q AS SELECT entity_id FROM read_parquet(?)',[config['queries_path']]);truth={q['entity_id']:set() for q in queries}
    for q,t in con.execute('SELECT source1_entity_id,target_id FROM positive_pairs JOIN q ON source1_entity_id=q.entity_id').fetchall():truth[q].add(t)
    candidates={q:set() for q in truth}
    for q,t in data.select('source1_entity_id','target_id').iter_rows():candidates[q].add(t)
    report={'config':config,'metadata':metadata,'metrics':summarize_candidates(queries,truth,candidates,10320219),'runtime_seconds':time.perf_counter()-start,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3,'limitations':['Same 1000 balanced-country development queries; no matcher score','Mac Foundation transform pinned; Linux parity not yet verified','Full native view retained; this independent route uses generic script conversion only']}
    (out/'metrics.json').write_text(json.dumps(report,indent=2));print(json.dumps({'link_recall':report['metrics']['link_recall'],'runtime':report['runtime_seconds']}))
if __name__=='__main__':main()
