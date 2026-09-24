"""Nested decision and calibration comparisons, using saved inner OOF scores.

No outer-fold label selects a threshold, calibrator, or emptiness threshold.
These are development comparisons; model selection consumes outer OOF evidence.
"""
import argparse,json
from pathlib import Path
import numpy as np,polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from src.models.lexical_pilot import select_threshold
from src.models.phase4_oof import scores

def arrays(frame,truth):
    ids=sorted(truth);lookup={q:i for i,q in enumerate(ids)}
    idx=np.array([lookup[q] for q in frame['source1_entity_id']]);counts=np.array([len(truth[q]) for q in ids]);return idx,counts

def empty_search(prob,idx,label,counts):
    """Joint threshold optimization on training-only inner OOF entities."""
    top=np.zeros(len(counts));np.maximum.at(top,idx,prob);best=(-1.,0.,0.);trials=[]
    for pair in np.arange(.2,.951,.025):
        keep=prob>=pair;p=np.bincount(idx[keep],minlength=len(counts));tp=np.bincount(idx[keep],weights=label[keep],minlength=len(counts))
        for empty in np.arange(pair,.976,.025):
            active=top>=empty;den=p*active+.25*counts;s=np.divide(1.25*tp*active,den,out=np.ones(len(den)),where=den>0);score=float(s.mean());trials.append({'pair_threshold':float(pair),'empty_threshold':float(empty),'macro_f0_5':score})
            best=max(best,(score,float(pair),float(empty)))
    return best[1:],trials

def main():
    import duckdb
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--sample',default='A');a=p.parse_args();root=Path(a.run);out=root/'decisions';out.mkdir(exist_ok=False)
    q=pl.read_parquet(f'artifacts/validation/phase4-v001/P4-SAMPLE-{a.sample}.parquet');db=duckdb.connect('artifacts/audit.duckdb',read_only=True,config={'threads':2,'memory_limit':'1GB'});db.execute('CREATE TEMP TABLE selected_ids AS SELECT entity_id FROM read_parquet(?)',[f'artifacts/validation/phase4-v001/P4-SAMPLE-{a.sample}.parquet'])
    truth={i:set() for i in q['entity_id']}
    for s,t in db.execute('SELECT p.source1_entity_id,p.target_id FROM positive_pairs p JOIN selected_ids q ON q.entity_id=p.source1_entity_id').fetchall():truth[s].add(t)
    db.close();results=[];pooled={}
    for held in [1,2,3]:
        d=root/f'FOLD-{held}';inner=pl.read_parquet(d/'inner_pair_scores.parquet');outer=pl.read_parquet(d/'pair_scores.parquet');tq=q.filter(pl.col('fold')!=held);vq=q.filter(pl.col('fold')==held);tt={i:truth[i] for i in tq['entity_id']};vt={i:truth[i] for i in vq['entity_id']};idx,counts=arrays(inner,tt);labels=inner['label'].to_numpy();raw=inner['score'].to_numpy();rawout=outer['score'].to_numpy()
        transformed={'raw':(raw,rawout)}
        logit=lambda x:np.log(np.clip(x,1e-6,1-1e-6)/(1-np.clip(x,1e-6,1-1e-6))).reshape(-1,1)
        platt=LogisticRegression(C=1.,solver='lbfgs',max_iter=200);platt.fit(logit(raw),labels);transformed['platt']=(platt.predict_proba(logit(raw))[:,1],platt.predict_proba(logit(rawout))[:,1]);iso=IsotonicRegression(out_of_bounds='clip');iso.fit(raw,labels);transformed['isotonic']=(iso.predict(raw),iso.predict(rawout))
        serialized={'platt_coef':platt.coef_.tolist(),'platt_intercept':platt.intercept_.tolist(),'isotonic_x':iso.X_thresholds_.tolist(),'isotonic_y':iso.y_thresholds_.tolist()};(out/f'fold-{held}-calibrators.json').write_text(json.dumps(serialized))
        for name,(trainprob,testprob) in transformed.items():
            threshold,trials=select_threshold(trainprob,idx,labels,counts);report,pred=scores(outer,testprob,threshold,vt,vq);pooled.setdefault(name,{}).update(pred)
            results.append({'fold':held,'method':name,'threshold':threshold,'brier':float(np.mean((testprob-outer['label'].to_numpy())**2)),'scores':report});pl.DataFrame(trials).write_parquet(out/f'fold-{held}-{name}-thresholds.parquet')
            pl.DataFrame({'score':testprob,'label':outer['label']}).with_columns((pl.col('score')*10).floor().clip(0,9).alias('bin')).group_by('bin').agg(pl.len().alias('pairs'),pl.col('score').mean().alias('mean_score'),pl.col('label').mean().alias('positive_rate')).sort('bin').write_csv(out/f'fold-{held}-{name}-reliability.csv')
        (pair,empty),trials=empty_search(raw,idx,labels,counts);oi,oc=arrays(outer,vt);top=np.zeros(len(oc));np.maximum.at(top,oi,rawout);adjusted=np.where(top[oi]>=empty,rawout,-1.);report,pred=scores(outer,adjusted,pair,vt,vq);pooled.setdefault('pair_plus_empty',{}).update(pred);results.append({'fold':held,'method':'pair_plus_empty','threshold':pair,'empty_threshold':empty,'scores':report});pl.DataFrame(trials).write_parquet(out/f'fold-{held}-empty-grid.parquet')
        print(json.dumps({'fold':held,'decisions_complete':True}),flush=True)
    from src.evaluation import evaluate
    report={'results':results,'pooled':{name:evaluate(truth,pred) for name,pred in pooled.items()},'scope':'Outer evaluation; all calibrators/thresholds fit only inner OOF from outer training folds. Comparing methods uses development evidence, not locked holdout.'};(out/'metrics.json').write_text(json.dumps(report,indent=2));print(json.dumps(report['pooled']))
if __name__=='__main__':main()
