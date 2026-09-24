"""Feature ablations on identical cached candidates and nested entity folds."""
import argparse,json,time,hashlib,subprocess,shutil,importlib.metadata
from pathlib import Path
import duckdb,numpy as np,polars as pl
from src.models.phase4_oof import NAMES,fitting_mask,scores
from src.models.lexical_pilot import select_threshold
from src.evaluation import evaluate,entity_f05

CONTEXT_BASE=['name_jw','address_jw','name_x_address_jw','name_retrieval_score','address_retrieval_score']
CONTEXT_NAMES=[f'{n}_{suffix}' for n in CONTEXT_BASE for suffix in ['query_max','query_gap','query_relative']]

def add_query_context(frame):
    """Deterministic per-query unlabeled context, recomputable at inference."""
    for n in CONTEXT_BASE:
        frame=frame.with_columns(pl.col(n).max().over('source1_entity_id').alias(n+'_query_max'))
        frame=frame.with_columns((pl.col(n)-pl.col(n+'_query_max')).alias(n+'_query_gap'),pl.when(pl.col(n+'_query_max')>0).then(pl.col(n)/pl.col(n+'_query_max')).otherwise(0.).alias(n+'_query_relative'))
    return frame

def retained(variant):
    if variant=='baseline':return NAMES
    if variant=='with_query_context':return NAMES+CONTEXT_NAMES
    if variant=='with_canonical_numeric':
        from src.models.phase4_numeric import NAMES as numeric_names
        return NAMES+numeric_names
    def remove(n):
        if variant=='without_translit':return 'translit' in n
        if variant=='without_numeric':return 'numeric' in n
        if variant=='without_address':return n.startswith('address_') or 'numeric' in n or 'address_missing' in n or n=='name_x_address_jw'
        if variant=='without_name':return n.startswith('name_')
        if variant=='without_retrieval':return 'retrieval' in n or 'reciprocal_rank' in n
        if variant=='without_route_count':return n=='retrieval_route_count'
        if variant=='without_token_similarity':return n.startswith(('name_','address_')) and any(k in n for k in ['token','jaccard','dice','containment','first_equal','last_equal'])
        if variant=='without_char_similarity':return n.startswith(('name_','address_')) and any(k in n for k in ['_jw','levenshtein','_ratio','_partial','retrieval_score'])
        raise ValueError(variant)
    return [n for n in NAMES if not remove(n)]

def main():
    import lightgbm as lgb
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output');p.add_argument('--extra-features');p.add_argument('--country-balanced',action='store_true');p.add_argument('--reference-predictions');p.add_argument('--ownership-ipw',action='store_true');p.add_argument('--family',choices=['lightgbm','catboost','xgboost'],default='lightgbm');p.add_argument('--sample',default='A');p.add_argument('--variants',nargs='+',default=['without_translit','without_numeric','without_address','without_name','without_retrieval','without_route_count','without_token_similarity','without_char_similarity']);a=p.parse_args();root=Path(a.run);out=Path(a.output) if a.output else root/'ablations';out.mkdir(parents=True,exist_ok=False)
    snapshot=out/'source_snapshot';snapshot.mkdir()
    source_root=Path('code/business_entity_resolution/src')
    hashes={}
    for source in source_root.rglob('*.py'):
        relative=source.relative_to(source_root);destination=snapshot/relative;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,destination);hashes[str(relative)]=hashlib.sha256(destination.read_bytes()).hexdigest()
    (out/'provenance.json').write_text(json.dumps({'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'source_sha256':hashes,'arguments':vars(a),'baseline_sha256':hashlib.sha256(Path('configs/baselines/BASELINE-P4-001.yaml').read_bytes()).hexdigest(),'versions':{name:importlib.metadata.version(name) for name in ['numpy','polars','lightgbm','scikit-learn','rapidfuzz']}},indent=2))
    frame=pl.read_parquet(str(root/'features/part-*.parquet')).sort('source1_entity_id','target_id');q=pl.read_parquet(f'artifacts/validation/phase4-v001/P4-SAMPLE-{a.sample}.parquet');db=duckdb.connect('artifacts/audit.duckdb',read_only=True,config={'threads':2,'memory_limit':'1GB'});truth={i:set() for i in q['entity_id']}
    for s,t in db.execute('SELECT p.source1_entity_id,p.target_id FROM positive_pairs p JOIN read_parquet(?) q ON q.entity_id=p.source1_entity_id',[f'artifacts/validation/phase4-v001/P4-SAMPLE-{a.sample}.parquet']).fetchall():truth[s].add(t)
    db.close();
    if 'with_query_context' in a.variants:frame=add_query_context(frame)
    if a.extra_features:
        extra=pl.read_parquet(a.extra_features);frame=frame.join(extra,on=['source1_entity_id','target_id'],how='left',validate='1:1')
        if frame.select(pl.any_horizontal(pl.all().is_null()).any()).item():raise ValueError('Missing feature join')
    fold=frame['fold'].to_numpy();owner=frame['owner_fold'].to_numpy();y=frame['label'].to_numpy();ids=frame['source1_entity_id'].to_numpy();params=json.loads(Path('configs/baselines/BASELINE-P4-001.yaml').read_text())['hyperparameters'];basepred={}
    for held in [1,2,3]:
        for s,ts in pl.read_csv(root/f'FOLD-{held}'/'entity_predictions.tsv',separator='\t').iter_rows():basepred[s]=set(ts.split(',')) if ts else set()
    if a.reference_predictions:
        basepred={s:set(ts.split(',')) if ts else set() for s,ts in pl.read_csv(a.reference_predictions,separator='\t').iter_rows()}
        if set(basepred)!=set(truth):raise ValueError('Reference prediction IDs mismatch')
    reports=[]
    for variant in a.variants:
        start=time.perf_counter();dest=out/variant;dest.mkdir();names=retained(variant);x=frame.select(names).to_numpy();predicted={};details=[]
        def fit(allowed):
            mask=fitting_mask(fold,owner,allowed);
            if a.family=='lightgbm':m=lgb.LGBMClassifier(**params)
            else:
                from src.models.family_adapter import Adapter
                m=Adapter(a.family)
            fit_kwargs={}
            if a.ownership_ipw:
                if a.family!='lightgbm':raise ValueError('IPW probe currently LightGBM only')
                fit_kwargs['sample_weight']=np.where((y[mask]==0)&(owner[mask]>=0),5/len(allowed),1.)
            if a.country_balanced:
                from src.models.country_weights import country_pair_weights
                if a.family!='lightgbm':raise ValueError('Country balance probe currently LightGBM only')
                weights=country_pair_weights(q,ids[mask],allowed)
                fit_kwargs['sample_weight']=weights*fit_kwargs.get('sample_weight',1.)
            m.fit(x[mask],y[mask],feature_name=names,**fit_kwargs);return m
        for held in [1,2,3]:
            allowed=[f for f in [1,2,3] if f!=held];inner=np.zeros(len(frame))
            for val in allowed:
                model=fit([f for f in allowed if f!=val]);mask=fold==val;inner[mask]=model.booster_.predict(x[mask])
            trainids=sorted(q.filter(pl.col('fold').is_in(allowed))['entity_id']);lookup={s:i for i,s in enumerate(trainids)};mask=np.isin(fold,allowed);threshold,trials=select_threshold(inner[mask],np.array([lookup[s] for s in ids[mask]]),y[mask],np.array([len(truth[s]) for s in trainids]));model=fit(allowed);mask=fold==held;prob=model.booster_.predict(x[mask]);vq=q.filter(pl.col('fold')==held);vt={s:truth[s] for s in vq['entity_id']};report,pred=scores(frame.filter(pl.col('fold')==held),prob,threshold,vt,vq);predicted.update(pred);details.append({'fold':held,'threshold':threshold,'scores':report});model.booster_.save_model(str(dest/f'fold-{held}.txt'))
        keys=sorted(truth);delta=np.array([entity_f05(truth[s],predicted[s])-entity_f05(truth[s],basepred[s]) for s in keys]);rng=np.random.default_rng(20260925);ci=np.quantile([rng.choice(delta,len(delta),replace=True).mean() for _ in range(1000)],[.025,.975]).tolist();result={'country_balanced':a.country_balanced,'reference_predictions':a.reference_predictions,'ownership_ipw':a.ownership_ipw,'family':a.family,'variant':variant,'features':names,'folds':details,'pooled':evaluate(truth,predicted),'paired_macro_delta':float(delta.mean()),'paired_entity_bootstrap_95_ci':ci,'seconds':time.perf_counter()-start};reports.append(result);(dest/'metrics.json').write_text(json.dumps(result,indent=2));pl.DataFrame({'source1_entity_id':keys,'matched_entity_ids':[','.join(sorted(predicted[s])) for s in keys]}).write_csv(dest/'entity_predictions.tsv',separator='\t');(out/'metrics.json').write_text(json.dumps({'variants':reports,'scope':'Development comparisons; same candidates/outer folds, nested thresholds per variant. Raw and separate deduplicated preprocessing views absent from baseline, not claimed as tested.'},indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['features','folds']}),flush=True)
if __name__=='__main__':main()
