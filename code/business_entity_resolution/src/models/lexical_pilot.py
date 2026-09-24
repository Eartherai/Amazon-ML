"""Small entity-separated lexical GBDT benchmark on actual retrieval negatives.

Fit S1 fold1; select model/threshold on a fixed half of sampled fold0; report the
other half separately. This is a development pilot, NOT OOF or final holdout.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os,resource,time,subprocess
from pathlib import Path
import duckdb
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler,Levenshtein
from src.evaluation import evaluate
from src.blocking.token_candidates import summarize_candidates


def text_features(a:str,b:str)->list[float]:
    if not a or not b:return [0.]*13
    aa,bb=set(a.split()),set(b.split());n=len(aa&bb)
    return [float(a==b),JaroWinkler.normalized_similarity(a,b),Levenshtein.normalized_similarity(a,b),
       fuzz.ratio(a,b)/100,fuzz.token_sort_ratio(a,b)/100,fuzz.token_set_ratio(a,b)/100,fuzz.partial_ratio(a,b)/100,
       n/len(aa|bb),2*n/(len(aa)+len(bb)),n/min(len(aa),len(bb)),min(len(a),len(b))/max(len(a),len(b)),
       float(a.split()[0]==b.split()[0]),float(a.split()[-1]==b.split()[-1])]


TEXT_NAMES=['exact','jw','levenshtein','ratio','token_sort','token_set','partial','jaccard','dice','containment','length_ratio','first_equal','last_equal']
FEATURES=[f'{field}_{name}' for field in ['name','address'] for name in TEXT_NAMES]+['name_x_address_jw','numeric_overlap','numeric_conflict','numeric_jaccard','query_address_missing','target_address_missing','country_equal','target_s2','name_retrieval_score','address_retrieval_score','name_reciprocal_rank','address_reciprocal_rank','retrieval_route_count']


def materialize(db,queryfile:Path,routedir:Path,dest:Path,training:bool):
    """Build labels only after actual candidates; exclude heldout-owner train negatives."""
    import re
    if dest.exists():return pl.read_parquet(dest)
    db.execute('CREATE OR REPLACE TEMP TABLE q AS SELECT * FROM read_parquet(?)',[str(queryfile)])
    db.execute('''CREATE OR REPLACE TEMP TABLE r AS SELECT source1_entity_id,target_id,
       max(CASE WHEN route='name_char3' THEN route_score ELSE 0 END) name_score,
       max(CASE WHEN route='address_char3' THEN route_score ELSE 0 END) address_score,
       max(CASE WHEN route='name_char3' THEN 1.0/route_rank ELSE 0 END) name_rank,
       max(CASE WHEN route='address_char3' THEN 1.0/route_rank ELSE 0 END) address_rank,
       count(DISTINCT route) routes FROM read_parquet(?) GROUP BY 1,2''',[[str(routedir/f'{field}_char3.parquet') for field in ['name','address']]])
    db.execute('''CREATE OR REPLACE TEMP TABLE labeled AS SELECT r.*,p.target_id IS NOT NULL AS is_label,o.owner_fold
      FROM r LEFT JOIN positive_pairs p USING(source1_entity_id,target_id)
      JOIN target_ownership o ON o.target_id=r.target_id''')
    condition='WHERE owner_fold IN (-1,1,2,3)' if training else ''
    if training and db.execute('SELECT count(*) FROM labeled WHERE is_label AND owner_fold<>1').fetchone()[0]:raise AssertionError('Unexpected positive owner in training')
    cursor=db.execute(f'''SELECT l.*,q.n,q.a,t.n target_name,t.a target_address,q.country,t.country target_country
       FROM labeled l JOIN q ON q.entity_id=l.source1_entity_id JOIN targets_normalized t ON t.entity_id=l.target_id {condition} ORDER BY l.source1_entity_id,l.target_id''')
    cols=[x[0] for x in cursor.description];rows=cursor.fetchall();result=[]
    for i,values in enumerate(rows):
        r=dict(zip(cols,values));n=text_features(r['n'],r['target_name']);a=text_features(r['a'],r['target_address'])
        na,nb=set(re.findall(r'\d+',r['a'])),set(re.findall(r'\d+',r['target_address']));shared=len(na&nb)
        features=n+a+[n[1]*a[1],float(shared>0),float(bool(na and nb) and not shared),shared/len(na|nb) if na or nb else 0.,float(not r['a']),float(not r['target_address']),float(r['country']==r['target_country']),float(r['target_id'].startswith('S2-')),r['name_score'],r['address_score'],r['name_rank'],r['address_rank'],r['routes']]
        result.append((r['source1_entity_id'],r['target_id'],r['country'],int(r['is_label']),*features))
        if i and i%50000==0:print(json.dumps({'feature_rows':i,'training':training}),flush=True)
    frame=pl.DataFrame(result,schema=['source1_entity_id','target_id','country','label']+FEATURES,orient='row').with_columns(pl.col(FEATURES).cast(pl.Float32));frame.write_parquet(dest,compression='zstd');return frame


def select_threshold(prob:np.ndarray,entity_index:np.ndarray,labels:np.ndarray,truth_counts:np.ndarray)->tuple[float,list]:
    """Optimize exact macro F0.5, including entities with no candidate rows."""
    trials=[]
    for threshold in np.linspace(.01,.99,99):
        keep=prob>=threshold;pred=np.bincount(entity_index[keep],minlength=len(truth_counts));tp=np.bincount(entity_index[keep],weights=labels[keep],minlength=len(truth_counts));den=pred+.25*truth_counts
        scores=np.divide(1.25*tp,den,out=np.ones(len(den)),where=den>0);trials.append({'threshold':float(threshold),'macro_f0_5':float(scores.mean())})
    best=max(trials,key=lambda r:(r['macro_f0_5'],r['threshold']));return best['threshold'],trials


def main():
    import lightgbm as lgb
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('outputs/experiments/GBDT-001'));a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();db=duckdb.connect('artifacts/audit.duckdb',read_only=True,config={'threads':2,'memory_limit':'2GB'})
    trainq=Path('outputs/candidates/TRAIN-001/queries.parquet');devq=Path('outputs/candidates/TOKEN-001/run-002/pilot_queries.parquet')
    train=materialize(db,trainq,Path('outputs/candidates/TRAIN-001/char'),a.output/'train_features.parquet',True)
    dev=materialize(db,devq,Path('outputs/candidates/CHAR-001'),a.output/'dev_features.parquet',False)
    queries=pl.read_parquet(devq).select('entity_id','country','n','a').to_dicts();truth={q['entity_id']:set() for q in queries}
    for q,t in db.execute('SELECT source1_entity_id,target_id FROM positive_pairs WHERE source1_entity_id IN (SELECT entity_id FROM read_parquet(?))',[str(devq)]).fetchall():truth[q].add(t)
    calib={q for q in truth if int(hashlib.sha256(('calib-v1|'+q).encode()).hexdigest()[:8],16)%2==0};eval_ids=set(truth)-calib
    ids=sorted(calib);index={q:i for i,q in enumerate(ids)};mask=np.array([q in calib for q in dev['source1_entity_id']]);idx=np.array([index[q] for q in dev['source1_entity_id'] if q in calib]);counts=np.array([len(truth[q]) for q in ids]);labels=dev['label'].to_numpy()
    xtrain=train.select(FEATURES).to_numpy();xdev=dev.select(FEATURES).to_numpy();ytrain=train['label'].to_numpy();results=[];all_probs={}
    for name,leaves,minleaf in [('gbdt31',31,30),('gbdt15',15,50)]:
        begin=time.perf_counter();model=lgb.LGBMClassifier(n_estimators=250,num_leaves=leaves,min_child_samples=minleaf,learning_rate=.06,reg_lambda=2,random_state=20260925,n_jobs=2,deterministic=True,force_col_wise=True,verbosity=-1)
        model.fit(xtrain,ytrain,feature_name=FEATURES);prob=model.predict_proba(xdev)[:,1];threshold,trials=select_threshold(prob[mask],idx,labels[mask],counts)
        predictions={q:set() for q in truth}
        for q,t,score in zip(dev['source1_entity_id'],dev['target_id'],prob):
            if score>=threshold:predictions[q].add(t)
        scores={}
        for part,selected in [('calibration',calib),('development_check',eval_ids),('all_development',set(truth))]:
            scores[part]=evaluate({q:truth[q] for q in selected},{q:predictions[q] for q in selected})
        slices={}
        for country in ['India','US']:
            selected={q['entity_id'] for q in queries if q['country']==country}&eval_ids
            slices[country]=evaluate({q:truth[q] for q in selected},{q:predictions[q] for q in selected})
        for target_source in ['S2-','S3-']:
            slices[target_source[:2]]=evaluate({q:{t for t in truth[q] if t.startswith(target_source)} for q in eval_ids},{q:{t for t in predictions[q] if t.startswith(target_source)} for q in eval_ids})
        for family,predicate in [('singleton',lambda n:n==0),('1_match',lambda n:n==1),('2_matches',lambda n:n==2),('3_to_5_matches',lambda n:3<=n<=5),('6_plus_matches',lambda n:n>=6)]:
            selected={q for q in eval_ids if predicate(len(truth[q]))}
            if selected:slices[family]=evaluate({q:truth[q] for q in selected},{q:predictions[q] for q in selected})
        model.booster_.save_model(str(a.output/(name+'.txt')));pl.DataFrame({'feature':FEATURES,'gain':model.booster_.feature_importance('gain')}).sort('gain',descending=True).write_csv(a.output/(name+'_feature_importance.csv'))
        results.append({'name':name,'threshold':threshold,'scores':scores,'development_check_slices':slices,'threshold_trials':trials,'fit_predict_seconds':time.perf_counter()-begin,'hyperparameters':model.get_params()});all_probs[name]=prob
        print(json.dumps({'model':name,'threshold':threshold,'calibration_macro':scores['calibration']['macro_f0_5'],'development_check':scores['development_check']}),flush=True)
    best=max(results,key=lambda r:r['scores']['calibration']['macro_f0_5'])
    prediction_frame=dev.select('source1_entity_id','target_id','country','label').with_columns([pl.Series(name,prob) for name,prob in all_probs.items()]);prediction_frame.write_parquet(a.output/'dev_pair_predictions.parquet')
    selected_scores=all_probs[best['name']];chosen=dev.with_columns(pl.Series('probability',selected_scores)).with_columns((pl.col('probability')>=best['threshold']).alias('predicted'))
    errors=chosen.filter((pl.col('label')==0)&pl.col('predicted')).sort('probability',descending=True)
    errors.head(200).write_csv(a.output/'highest_confidence_dev_false_positives.csv')
    chosen.filter((pl.col('label')==1)&~pl.col('predicted')).sort('probability').head(200).write_csv(a.output/'lowest_confidence_retrieved_dev_positives.csv')
    families={'false_positive_pairs':len(errors),'exact_name':int(errors['name_exact'].sum()),'name_strong_address_weak':len(errors.filter((pl.col('name_jw')>=.9)&(pl.col('address_jw')<.7))),'address_strong_name_weak':len(errors.filter((pl.col('address_jw')>=.9)&(pl.col('name_jw')<.7))),'numeric_conflict':int(errors['numeric_conflict'].sum()),'singleton_queries':len({q for q in errors['source1_entity_id'] if not truth[q]})}
    (a.output/'error_families.json').write_text(json.dumps({'scope':'all sampled development, overlapping diagnostic families','counts':families},indent=2))
    # In-sample hard negatives are diagnostic examples only; a later round must use entity-OOF mining.
    report={'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'module_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'Small first matcher; 1000 fold1 train entities, 1000 fold0 dev entities split by fixed hash into calibration/check. Not OOF; fold4 remains closed. Candidate architecture was selected on fold0, so check is not an untouched architecture holdout.','train_pairs':len(train),'train_positive_pairs':int(ytrain.sum()),'dev_pairs':len(dev),'calibration_entities':len(calib),'development_check_entities':len(eval_ids),'selected_model':best['name'],'selection_rule':'Maximum calibration macro F0.5 only; tie favors first model','models':results,'features':FEATURES,'runtime_seconds':time.perf_counter()-start,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3,'aws_compute_cost_usd':0,'lightgbm_version':lgb.__version__,'limitations':['No final test predictions or submission','No OOF hard-negative mining yet','Only 1000 training entities; expand before selection','Two char3 routes, not full candidate union; current score cannot be assigned to other candidate configurations','No guarantee of calibrated probabilities from raw GBDT scores']}
    (a.output/'metrics.json').write_text(json.dumps(report,indent=2));print(json.dumps({'selected':best['name'],'runtime':report['runtime_seconds']}))
if __name__=='__main__':main()
