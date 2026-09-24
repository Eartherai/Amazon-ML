"""Add independent transliteration features and entity-OOF hard-negative weights.

Selection remains calibration-only; fold0 check is descriptive, fold4 closed.
"""
import hashlib,json,resource,time,subprocess
from pathlib import Path
import duckdb,numpy as np,polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler,Levenshtein
from src.models.lexical_pilot import FEATURES,select_threshold
from src.evaluation import evaluate


def main():
    import lightgbm as lgb
    out=Path('outputs/experiments/GBDT-004');out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    db=duckdb.connect('artifacts/audit.duckdb',read_only=True,config={'threads':2,'memory_limit':'2GB'})
    frames={}
    extras=['name_translit_jw','name_translit_ratio','name_translit_tokensort','name_translit_exact','address_first_numeric_equal','address_first_numeric_conflict']
    for split in ['train','dev']:
        original=Path('outputs/experiments/GBDT-002')/(split+'_features.parquet')
        db.execute('CREATE OR REPLACE TEMP TABLE f AS SELECT * FROM read_parquet(?)',[str(original)])
        cursor=db.execute("""SELECT f.source1_entity_id,f.target_id,s.n source_name,coalesce(m.transliterated,t.n) target_name,
          regexp_extract(s.a,'[0-9]+') first_a,regexp_extract(t.a,'[0-9]+') first_b,o.source1_entity_id AS owner_id
          FROM f JOIN s1_normalized s ON s.entity_id=f.source1_entity_id JOIN targets_normalized t ON t.entity_id=f.target_id
          JOIN target_ownership o ON o.target_id=t.entity_id
          LEFT JOIN read_parquet('artifacts/transliteration/TRANS-001/name_map.parquet') m ON m.n=t.n""")
        rows=[]
        for q,t,a,b,na,nb,owner in cursor.fetchall():rows.append((q,t,JaroWinkler.normalized_similarity(a,b) if a and b else 0.,fuzz.ratio(a,b)/100 if a and b else 0.,fuzz.token_sort_ratio(a,b)/100 if a and b else 0.,float(bool(a) and a==b),float(bool(na and nb) and na==nb),float(bool(na and nb) and na!=nb),owner))
        additions=pl.DataFrame(rows,schema=['source1_entity_id','target_id',*extras,'owner'],orient='row')
        frame=pl.read_parquet(original).join(additions,on=['source1_entity_id','target_id'],validate='1:1').sort('source1_entity_id','target_id');frame.write_parquet(out/(split+'_features.parquet'));frames[split]=frame
    train,dev=frames['train'],frames['dev'];names=FEATURES+extras
    x=train.select(names).to_numpy();y=train['label'].to_numpy();xd=dev.select(names).to_numpy()
    fold=lambda q:int(hashlib.sha256(('inner-oof-v1|'+q).encode()).hexdigest()[:8],16)%2
    foldids=np.array([fold(q) for q in train['source1_entity_id']]);queryset=set(train['source1_entity_id'])
    ownerfold=np.array([fold(owner) if owner in queryset else -1 for owner in train['owner']])
    def model():return lgb.LGBMClassifier(n_estimators=250,num_leaves=31,min_child_samples=30,learning_rate=.06,reg_lambda=2,random_state=20260925,n_jobs=2,deterministic=True,force_col_wise=True,verbosity=-1)
    oof=np.zeros(len(train));counts=[]
    for held in [0,1]:
        fit=(foldids!=held)&(ownerfold!=held);check=foldids==held
        if np.any((y==1)&(foldids!=held)&(ownerfold==held)):raise AssertionError('Positive owner leakage')
        m=model();m.fit(x[fit],y[fit]);oof[check]=m.booster_.predict(x[check]);counts.append({'held_subfold':held,'fit_pairs':int(fit.sum()),'oof_pairs':int(check.sum()),'excluded_owned_pairs':int(((foldids!=held)&(ownerfold==held)).sum())})
    hard=(y==0)&(oof>=.2)
    train.select('source1_entity_id','target_id','label').with_columns(pl.Series('oof_probability',oof),pl.Series('hard_negative',hard)).write_parquet(out/'oof_hard_negatives.parquet')
    queries=pl.read_parquet('outputs/candidates/TOKEN-001/run-002/pilot_queries.parquet').to_dicts();truth={q['entity_id']:set() for q in queries};db.execute('CREATE TEMP TABLE q AS SELECT entity_id FROM read_parquet(?)',['outputs/candidates/TOKEN-001/run-002/pilot_queries.parquet'])
    for q,t in db.execute('SELECT source1_entity_id,target_id FROM positive_pairs JOIN q ON q.entity_id=source1_entity_id').fetchall():truth[q].add(t)
    calib={q for q in truth if int(hashlib.sha256(('calib-v1|'+q).encode()).hexdigest()[:8],16)%2==0};check=set(truth)-calib;ci={q:i for i,q in enumerate(sorted(calib))};mask=np.array([q in calib for q in dev['source1_entity_id']]);index=np.array([ci[q] for q in dev['source1_entity_id'] if q in calib]);actual=np.array([len(truth[q]) for q in sorted(calib)]);labels=dev['label'].to_numpy()
    results=[];probabilities={}
    for experiment,weight in [('multiview',1.),('multiview_hardneg3',3.)]:
        weights=np.where(hard,weight,1.);m=model();m.fit(x,y,sample_weight=weights);prob=m.booster_.predict(xd);threshold,trials=select_threshold(prob[mask],index,labels[mask],actual);pred={q:set() for q in truth}
        for q,t,p in zip(dev['source1_entity_id'],dev['target_id'],prob):
            if p>=threshold:pred[q].add(t)
        scores={}
        groups={'calibration':calib,'development_check':check,'all_development':set(truth)}
        groups.update({country:{q['entity_id'] for q in queries if q['country']==country}&check for country in ['India','US']})
        for key,ids in groups.items():scores[key]=evaluate({q:truth[q] for q in ids},{q:pred[q] for q in ids})
        m.booster_.save_model(str(out/(experiment+'.txt')));pl.DataFrame({'feature':names,'gain':m.booster_.feature_importance('gain')}).sort('gain',descending=True).write_csv(out/(experiment+'_importance.csv'))
        results.append({'model':experiment,'hard_negative_weight':weight,'threshold':threshold,'scores':scores,'threshold_trials':trials});probabilities[experiment]=prob
        print(json.dumps({'model':experiment,'threshold':threshold,'calibration':scores['calibration']['macro_f0_5'],'check':scores['development_check']}),flush=True)
    selected=max(results,key=lambda r:r['scores']['calibration']['macro_f0_5'])
    dev.select('source1_entity_id','target_id','country','label').with_columns([pl.Series(name,p) for name,p in probabilities.items()]).write_parquet(out/'dev_predictions.parquet')
    report={'scope':'Same 1000 fold1 training queries and fold0 495 calibration/505 check queries as GBDT-002. Two-fold inner entity-OOF training predictions mine hard negatives; owners of inner heldout S1 excluded from fit negatives. Final benchmark is not OOF; fold4 remains closed. Check is not untouched architecture validation.','models':results,'selected_model':selected['model'],'selection':'calibration macro F0.5 only','hard_negatives':int(hard.sum()),'hard_negative_definition':'label=0 and inner-entity-OOF predicted probability>=0.20','inner_oof_counts':counts,'features':names,'runtime_seconds':time.perf_counter()-start,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3,'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'module_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'limitations':['Small pilot; no final submission','Sparse retrieval definition remains name_char3 + address_char3, top100 each','Transliteration backend currently Mac Foundation only','Raw model scores are not calibrated probabilities; threshold chosen against exact macro F0.5']}
    (out/'metrics.json').write_text(json.dumps(report,indent=2));print(json.dumps({'selected':selected['model'],'hard_negatives':int(hard.sum())}))
if __name__=='__main__':main()
