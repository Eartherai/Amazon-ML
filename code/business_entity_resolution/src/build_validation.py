"""Freeze deterministic entity folds and target ownership; no models are fit."""
import argparse,hashlib,json
from pathlib import Path
from src.audit_data import connect,rows,sqlstr,sha256

def fold_for(entity_id:str,seed:int=20260925,folds:int=5)->int:
    if folds<2: raise ValueError('Need at least two folds')
    return int(hashlib.sha256(f'{seed}|{entity_id}'.encode()).hexdigest()[:8],16)%folds

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--database',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--seed',type=int,default=20260925)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False);c=connect(a.database)
    shared=c.execute('SELECT count(*) FROM (SELECT target_id FROM positive_pairs GROUP BY 1 HAVING count(DISTINCT source1_entity_id)>1)').fetchone()[0]
    if shared:raise ValueError('Shared targets found; implement connected components before creating entity folds')
    c.execute(f'''CREATE TABLE validation_folds AS SELECT s.entity_id source1_entity_id,s.country,l.n_matches,
    ('0x'||substr(sha256('{a.seed}|'||s.entity_id),1,8))::UBIGINT%5 fold FROM train_source1 s JOIN label_counts l ON s.entity_id=l.source1_entity_id''')
    # Persist every target's owner fold; -1 means no S1 positive ownership in labels.
    c.execute('''CREATE TABLE target_ownership AS SELECT t.entity_id target_id,p.source1_entity_id,coalesce(f.fold,-1) owner_fold FROM train_targets t LEFT JOIN positive_pairs p ON p.target_id=t.entity_id LEFT JOIN validation_folds f USING(source1_entity_id)''')
    for table in ['validation_folds','target_ownership']:
        c.execute(f"COPY (SELECT * FROM {table} ORDER BY 1) TO {sqlstr(a.output_dir/(table+'.parquet'))} (FORMAT PARQUET,COMPRESSION ZSTD)")
    # A paired training example must have training S1 and target whose owner is training/unowned.
    c.execute('''CREATE VIEW permitted_training_targets AS SELECT * FROM target_ownership WHERE owner_fold IN (-1,1,2,3)''')
    report={'seed':a.seed,'fold_algorithm':'int(sha256(str(seed)+"|"+entity_id)[:8],16)%5',
      'fold_roles':{'0':'development and threshold selection','1,2,3':'initial training','4':'locked final evaluation; not for tuning'},
      'shared_targets':shared,
      'distribution':rows(c,'SELECT fold,country,count(*) entities,sum(n_matches) links,count(*) FILTER(WHERE n_matches=0) singletons FROM validation_folds GROUP BY ALL ORDER BY ALL'),
      'unowned_targets':c.execute('SELECT count(*) FROM target_ownership WHERE owner_fold=-1').fetchone()[0],
      'training_target_leakage_check':c.execute('SELECT count(*) FROM permitted_training_targets WHERE owner_fold IN (0,4)').fetchone()[0],
      'files':{x.name:sha256(x) for x in a.output_dir.glob('*.parquet')}}
    samples=c.execute('SELECT source1_entity_id,fold FROM validation_folds LIMIT 100').fetchall()
    assert all(fold_for(k,a.seed)==f for k,f in samples)
    (a.output_dir/'manifest.json').write_text(json.dumps(report,indent=2));c.execute('CHECKPOINT');print(json.dumps(report,indent=2))
if __name__=='__main__': main()
