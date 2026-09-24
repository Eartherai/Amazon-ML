"""Three outer entity folds with nested two-fold threshold selection.

Frozen external IDF fits only fold0/unowned targets; no outer validation text fits
IDF. Each model excludes targets owned by any entity outside its fit folds.
All outer predictions retain the full target pool. No fold4 evaluation.
"""
from pathlib import Path
import argparse,hashlib,json,time,resource,re,subprocess
import duckdb,numpy as np,polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from src.models.lexical_pilot import FEATURES,text_features,select_threshold
from src.evaluation import evaluate
EXTRAS=['name_translit_jw','name_translit_ratio','name_translit_tokensort','name_translit_exact','address_first_numeric_equal','address_first_numeric_conflict']
NAMES=FEATURES+EXTRAS

def fitting_mask(query_folds,owner_folds,allowed):
    """Exclude every held-out owner's target, even when it is a negative pair."""
    return np.isin(query_folds,allowed)&np.isin(owner_folds,[-1,*allowed])

def features(db,qpath,routes,out):
    """Stream joined candidates into bounded feature batches; preserve all scores."""
    db.execute('CREATE TEMP TABLE q AS SELECT * FROM read_parquet(?)',[str(qpath)])
    db.execute('''CREATE TEMP TABLE r AS SELECT source1_entity_id,target_id,
      max(CASE WHEN route='name_char3' THEN route_score ELSE 0 END) ns,
      max(CASE WHEN route='address_char3' THEN route_score ELSE 0 END) ads,
      max(CASE WHEN route='name_char3' THEN 1.0/route_rank ELSE 0 END) nr,
      max(CASE WHEN route='address_char3' THEN 1.0/route_rank ELSE 0 END) ar,
      count(*) routes FROM read_parquet(?) GROUP BY 1,2''',[[str(routes/f'{f}_char3.parquet') for f in ['name','address']]])
    cursor=db.execute('''SELECT r.*,q.country,q.fold,q.n,q.a,t.n tn,t.a ta,t.country tc,
       p.target_id IS NOT NULL AS positive,o.owner_fold,coalesce(m.transliterated,t.n) translit
       FROM r JOIN q ON q.entity_id=r.source1_entity_id JOIN targets_normalized t ON t.entity_id=r.target_id
       JOIN target_ownership o ON o.target_id=r.target_id
       LEFT JOIN positive_pairs p ON p.source1_entity_id=r.source1_entity_id AND p.target_id=r.target_id
       LEFT JOIN read_parquet('artifacts/transliteration/TRANS-001/name_map.parquet') m ON m.n=t.n''')
    cols=[c[0] for c in cursor.description];part=0
    while rows:=cursor.fetchmany(25000):
        built=[]
        for values in rows:
            r=dict(zip(cols,values));n=text_features(r['n'],r['tn']);a=text_features(r['a'],r['ta']);na,nb=set(re.findall(r'\d+',r['a'])),set(re.findall(r'\d+',r['ta']));shared=len(na&nb)
            basic=n+a+[n[1]*a[1],float(shared>0),float(bool(na and nb) and not shared),shared/len(na|nb) if na or nb else 0.,float(not r['a']),float(not r['ta']),float(r['country']==r['tc']),float(r['target_id'].startswith('S2-')),r['ns'],r['ads'],r['nr'],r['ar'],r['routes']]
            left,right=r['n'],r['translit'];firsta=re.search('[0-9]+',r['a']);firstb=re.search('[0-9]+',r['ta']);fa=firsta.group() if firsta else '';fb=firstb.group() if firstb else ''
            extra=[JaroWinkler.normalized_similarity(left,right) if left and right else 0.,fuzz.ratio(left,right)/100 if left and right else 0.,fuzz.token_sort_ratio(left,right)/100 if left and right else 0.,float(bool(left) and left==right),float(bool(fa and fb) and fa==fb),float(bool(fa and fb) and fa!=fb)]
            built.append((r['source1_entity_id'],r['target_id'],r['country'],r['fold'],r['owner_fold'],int(r['positive']),*basic,*extra))
        pl.DataFrame(built,schema=['source1_entity_id','target_id','country','fold','owner_fold','label',*NAMES],orient='row').with_columns(pl.col(NAMES).cast(pl.Float32)).write_parquet(out/f'part-{part:04}.parquet',compression='zstd');part+=1
        if part%10==0:print(json.dumps({'feature_rows':part*25000}),flush=True)

def scores(frame,prob,threshold,truth,queryframe):
    prediction={q:set() for q in truth}
    for q,t,p in zip(frame['source1_entity_id'],frame['target_id'],prob):
        if p>=threshold:prediction[q].add(t)
    result={'overall':evaluate(truth,prediction)}
    for country in sorted(set(queryframe['country'])):
        ids=set(queryframe.filter(pl.col('country')==country)['entity_id'])
        result[country]=evaluate({q:truth[q] for q in ids},{q:prediction[q] for q in ids})
    for source in ['S2','S3']:
        result[source]=evaluate({q:{t for t in ts if t.startswith(source+'-')} for q,ts in truth.items()},{q:{t for t in ts if t.startswith(source+'-')} for q,ts in prediction.items()})
    return result,prediction

def main():
    import lightgbm as lgb
    p=argparse.ArgumentParser();p.add_argument('--sample',default='A');p.add_argument('--routes',required=True);p.add_argument('--output',required=True);a=p.parse_args();start=time.perf_counter()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);cache=out/'features';cache.mkdir()
    qpath=Path(f'artifacts/validation/phase4-v001/P4-SAMPLE-{a.sample}.parquet');q=pl.read_parquet(qpath)
    assert set(q['fold'])=={1,2,3}
    routepath=Path(a.routes);rcfg=json.loads((routepath/'config.json').read_text());assert rcfg['fit_owner_folds']==[-1,0];assert rcfg['query_sha256']==hashlib.sha256(qpath.read_bytes()).hexdigest()
    db=duckdb.connect('artifacts/audit.duckdb',read_only=True,config={'threads':2,'memory_limit':'2GB'})
    features(db,qpath,routepath,cache)
    frame=pl.read_parquet(str(cache/'*.parquet')).sort('source1_entity_id','target_id')
    assert frame.select('source1_entity_id','target_id').unique().height==len(frame)
    truth={i:set() for i in q['entity_id']}
    for s,t in db.execute('SELECT p.source1_entity_id,p.target_id FROM positive_pairs p JOIN q ON q.entity_id=p.source1_entity_id').fetchall():truth[s].add(t)
    db.close();x=frame.select(NAMES).to_numpy();y=frame['label'].to_numpy();fold=frame['fold'].to_numpy();owner=frame['owner_fold'].to_numpy();ids=frame['source1_entity_id'].to_numpy()
    params=json.loads(Path('configs/baselines/BASELINE-P4-001.yaml').read_text())['hyperparameters']
    def fit(allowed):
        mask=fitting_mask(fold,owner,allowed)
        if np.any((y==1)&np.isin(fold,allowed)&~mask):raise AssertionError('Positive has inconsistent ownership')
        model=lgb.LGBMClassifier(**params);model.fit(x[mask],y[mask],feature_name=NAMES)
        return model,int(mask.sum())
    allpred={};allprob=np.zeros(len(frame));reports=[];entityscores=[]
    for held in [1,2,3]:
        target=out/f'FOLD-{held}';target.mkdir();allowed=[f for f in [1,2,3] if f!=held];inner=np.full(len(frame),np.nan);innercounts=[]
        for innerheld in allowed:
            innerfit=[f for f in allowed if f!=innerheld];m,count=fit(innerfit);mask=fold==innerheld;inner[mask]=m.booster_.predict(x[mask]);innercounts.append({'fit_folds':innerfit,'held':innerheld,'fit_pairs':count})
        trainids=sorted(q.filter(pl.col('fold').is_in(allowed))['entity_id']);lookup={s:i for i,s in enumerate(trainids)};mask=np.isin(fold,allowed)
        threshold,trials=select_threshold(inner[mask],np.array([lookup[s] for s in ids[mask]]),y[mask],np.array([len(truth[s]) for s in trainids]))
        pl.DataFrame(trials).write_parquet(target/'threshold_curves.parquet')
        frame.filter(pl.col('fold').is_in(allowed)).select('source1_entity_id','target_id','label').with_columns(pl.Series('score',inner[mask])).write_parquet(target/'inner_pair_scores.parquet')
        m,count=fit(allowed);mask=fold==held;prob=m.booster_.predict(x[mask]);allprob[mask]=prob;m.booster_.save_model(str(target/'model.txt'))
        heldframe=frame.filter(pl.col('fold')==held);heldq=q.filter(pl.col('fold')==held);heldtruth={s:truth[s] for s in heldq['entity_id']}
        result,pred=scores(heldframe,prob,threshold,heldtruth,heldq);fixed,_=scores(heldframe,prob,.58,heldtruth,heldq);allpred.update(pred)
        for s in sorted(pred):
            tp=len(pred[s]&truth[s]);den=len(pred[s])+.25*len(truth[s]);entityscores.append(1.25*tp/den if den else 1.)
        heldframe.select('source1_entity_id','target_id','country','label').with_columns(pl.Series('score',prob)).write_parquet(target/'pair_scores.parquet')
        pl.DataFrame({'source1_entity_id':sorted(pred),'matched_entity_ids':[','.join(sorted(pred[s])) for s in sorted(pred)]}).write_csv(target/'entity_predictions.tsv',separator='\t')
        report={'fold':held,'entities':len(heldq),'singleton_rate':len([s for s in heldtruth if not heldtruth[s]])/len(heldtruth),'fit_pairs':count,'threshold':threshold,'threshold_source':'Two inner entity folds wholly inside outer training partition','scores':result,'fixed_0_58_scores':fixed,'inner_folds':innercounts};reports.append(report);(target/'metrics.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    rng=np.random.default_rng(20260925);values=np.array(entityscores);boots=np.array([rng.choice(values,len(values),replace=True).mean() for _ in range(2000)])
    macros=[r['scores']['overall']['macro_f0_5'] for r in reports]
    report={'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'baseline_sha256':hashlib.sha256(Path('configs/baselines/BASELINE-P4-001.yaml').read_bytes()).hexdigest(),'retrieval_config':rcfg,'feature_file_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(cache.glob('*.parquet'))},'folds':reports,'pooled':evaluate(truth,allpred),'fold_macro_mean':float(np.mean(macros)),'fold_macro_std':float(np.std(macros,ddof=1)),'fold_macro_min':min(macros),'fold_macro_max':max(macros),'bootstrap_95_ci':np.quantile(boots,[.025,.975]).tolist(),'bootstrap_unit':'S1 entity; conditional on this training/split procedure, not model refit uncertainty','entities':len(q),'pairs':len(frame),'runtime_seconds':time.perf_counter()-start,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3,'fold4':'CLOSED','features':NAMES,'query_sha256':hashlib.sha256(qpath.read_bytes()).hexdigest(),'module_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'limitations':['Development OOF used for subsequent architecture selection is not a pristine final holdout','IDF is fixed using fold0 plus unowned text; differs from Phase3 baseline for leakage safety','Nested threshold models train on one outer fold; final outer models train on two; assess threshold transfer','No learned probability calibration or singleton model yet']}
    frame.select('source1_entity_id','target_id','country','fold','label').with_columns(pl.Series('score',allprob)).write_parquet(out/'pair_scores.parquet');(out/'metrics.json').write_text(json.dumps(report,indent=2));print(json.dumps({'pooled':report['pooled'],'ci':report['bootstrap_95_ci']}),flush=True)
if __name__=='__main__':main()
