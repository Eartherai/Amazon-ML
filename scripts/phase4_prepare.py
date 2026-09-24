"""Freeze Phase3 and create nested natural-prevalence Phase4 samples."""
from pathlib import Path
import hashlib,json,subprocess
from datetime import datetime,timezone
import duckdb,polars as pl

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    root=Path('artifacts/validation/phase4-v001');root.mkdir(parents=True,exist_ok=False)
    metrics=json.loads(Path('outputs/experiments/GBDT-004/metrics.json').read_text())
    paths=['manifests/MANIFEST_RAW.json','artifacts/validation/v1/manifest.json','configs/preprocessing/FOUNDATION-001.json','code/business_entity_resolution/src/preprocessing.py','code/business_entity_resolution/src/models/lexical_pilot.py','code/business_entity_resolution/src/models/refine_lexical_pilot.py','code/business_entity_resolution/src/blocking/char_retrieval.py','outputs/experiments/GBDT-004/multiview.txt','artifacts/transliteration/TRANS-001/name_map.parquet']
    paths += [str(p) for p in Path('outputs/candidates/CHAR-001').glob('*char3_idf.npz')]
    baseline={'id':'BASELINE-P4-001','git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'created_at':datetime.now(timezone.utc).isoformat(),'data_version':'raw-v001','files_sha256':{p:sha(p) for p in paths},'preprocessing':'Legacy database NFC/lowercase/Unicode L M N spacing; native preserved; Mac Foundation transliteration feature only. FOUNDATION-001 is stored but not the original model normalization.','feature_version':'GBDT-004/45-features','features':metrics['features'],'candidate_version':'CHAR-001/name_char3+address_char3','candidates':{'top_k_per_route':100,'maximum_union':200,'country_filter':True,'fit_owner_folds':[-1,1,2,3],'fit_sample':'sha256 target ID first byte 00..03','min_df':2,'max_features':200000,'dtype':'float32','norm':'l2'},'model_version':'LightGBM 4.7.0 GBDT-004 multiview','hyperparameters':{'n_estimators':250,'num_leaves':31,'min_child_samples':30,'learning_rate':.06,'reg_lambda':2,'random_state':20260925,'n_jobs':2,'deterministic':True,'force_col_wise':True,'verbosity':-1},'decision':{'pair_threshold':.58,'predict_empty_if_no_pair_passes':True,'hard_negative_weight':1},'score_scope':'505 repeatedly inspected fold0 entities; not OOF or expected leaderboard performance','phase4_required_change':'Fit IDF excluding all OOF entities; original IDF unsuitable for folds1-3 OOF','fold4':'CLOSED'}
    # JSON is a strict subset of YAML 1.2; retain exact typed metadata without dependency.
    dest=Path('configs/baselines/BASELINE-P4-001.yaml')
    if not dest.exists():
        with dest.open('x') as f:json.dump(baseline,f,indent=2)
    db=duckdb.connect('artifacts/audit.duckdb',read_only=True,config={'threads':2,'memory_limit':'2GB'})
    db.execute("""CREATE TEMP TABLE population AS SELECT s.*,f.fold,f.n_matches,l.has_s2,l.has_s3,
       CASE WHEN f.n_matches=0 THEN '0' WHEN f.n_matches=1 THEN '1' WHEN f.n_matches=2 THEN '2' WHEN f.n_matches<=5 THEN '3-5' ELSE '6+' END match_bucket,
       CASE WHEN l.has_s2 AND l.has_s3 THEN 'both' WHEN l.has_s2 THEN 'S2' WHEN l.has_s3 THEN 'S3' ELSE 'none' END source_mix,
       NOT regexp_full_match(s.n,'[\\x00-\\x7F]*') nonascii,
       s.n='' name_missing,s.a='' address_missing,
       CASE WHEN length(s.n)<10 THEN 'short' WHEN length(s.n)>40 THEN 'long' ELSE 'medium' END name_length_bucket,
       CASE WHEN length(s.a)<20 THEN 'short' WHEN length(s.a)>80 THEN 'long' ELSE 'medium' END address_length_bucket,
       sha256('phase4-natural-v1|'||s.entity_id) sample_hash
       FROM s1_normalized s JOIN validation_folds f ON f.source1_entity_id=s.entity_id
       JOIN label_counts l ON l.source1_entity_id=s.entity_id WHERE f.fold IN (1,2,3)""")
    db.execute('CREATE TEMP TABLE ranked AS SELECT *,row_number() OVER(ORDER BY sample_hash,entity_id) sample_rank FROM population')
    report={'selection':'Uniform deterministic SHA256 rank in folds1-3; nested samples, no balancing; labels only for descriptive strata','oof_folds':[1,2,3],'fold4':'CLOSED','samples':{},'files_sha256':{}}
    for name,size in [('A',5000),('B',20000),('C',50000),('D',100000)]:
        path=root/f'P4-SAMPLE-{name}.parquet'
        db.execute(f'COPY (SELECT * FROM ranked WHERE sample_rank<={int(size)} ORDER BY sample_rank) TO ? (FORMAT PARQUET,COMPRESSION ZSTD)',[str(path)])
        frame=pl.read_parquet(path)
        report['samples'][name]={'entities':len(frame),'singleton_rate':frame.filter(pl.col('n_matches')==0).height/len(frame),'strata':{c:frame.group_by(c).len().sort(c).to_dicts() for c in ['fold','country','match_bucket','source_mix','nonascii','name_missing','address_missing','name_length_bucket','address_length_bucket']}}
        report['files_sha256'][str(path)]=sha(path)
    path=root/'P4-DIAGNOSTIC.parquet'
    db.execute("""COPY (SELECT * EXCLUDE(stratum_rank) FROM (SELECT *,row_number() OVER(PARTITION BY country,match_bucket,source_mix,nonascii,name_missing,address_missing,name_length_bucket,address_length_bucket ORDER BY sample_hash) stratum_rank FROM population) WHERE stratum_rank<=20) TO ? (FORMAT PARQUET,COMPRESSION ZSTD)""",[str(path)])
    report['diagnostic']={'entities':pl.read_parquet(path).height,'purpose':'Up to20 per observed joint stratum; biased diagnostics, never headline score; can overlap natural samples'}
    report['files_sha256'][str(path)]=sha(path)
    (root/'manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:{a:b for a,b in v.items() if a!='strata'} for k,v in report['samples'].items()}))
if __name__=='__main__':main()
