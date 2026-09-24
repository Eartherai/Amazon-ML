"""Country label-transfer stress test; generic text IDF remains fixed externally.

This is not a strict held-country-text experiment: fold0 IDF contains both known
countries. No evaluation-owned text or labels fit the model or threshold.
"""
from pathlib import Path
import json,time
import duckdb,numpy as np,polars as pl
from src.models.phase4_oof import NAMES,fitting_mask,scores
from src.models.lexical_pilot import select_threshold

def main():
 import lightgbm as lgb
 root=Path('outputs/oof/P4-A-001');out=root/'country_transfer';out.mkdir(exist_ok=False);start=time.perf_counter();f=pl.read_parquet(str(root/'features/part-*.parquet')).sort('source1_entity_id','target_id');q=pl.read_parquet('artifacts/validation/phase4-v001/P4-SAMPLE-A.parquet');truth={s:set() for s in q['entity_id']};db=duckdb.connect('artifacts/audit.duckdb',read_only=True,config={'threads':2,'memory_limit':'1GB'})
 for s,t in db.execute("SELECT p.source1_entity_id,p.target_id FROM positive_pairs p JOIN read_parquet('artifacts/validation/phase4-v001/P4-SAMPLE-A.parquet') q ON q.entity_id=p.source1_entity_id").fetchall():truth[s].add(t)
 db.close();x=f.select(NAMES).to_numpy();y=f['label'].to_numpy();fold=f['fold'].to_numpy();owner=f['owner_fold'].to_numpy();country=f['country'].to_numpy();ids=f['source1_entity_id'].to_numpy();params=json.loads(Path('configs/baselines/BASELINE-P4-001.yaml').read_text())['hyperparameters'];reports=[]
 for train_country in sorted(set(q['country'])):
  domain=country==train_country;inner=np.zeros(len(f))
  def fit(allowed):
   mask=domain&fitting_mask(fold,owner,allowed);m=lgb.LGBMClassifier(**params);m.fit(x[mask],y[mask],feature_name=NAMES);return m
  for held in [1,2,3]:
   m=fit([i for i in [1,2,3] if i!=held]);mask=domain&(fold==held);inner[mask]=m.booster_.predict(x[mask])
  trainids=sorted(q.filter(pl.col('country')==train_country)['entity_id']);lookup={s:i for i,s in enumerate(trainids)};threshold,trials=select_threshold(inner[domain],np.array([lookup[s] for s in ids[domain]]),y[domain],np.array([len(truth[s]) for s in trainids]));m=fit([1,2,3]);mask=~domain;test=f.filter(pl.col('country')!=train_country);tq=q.filter(pl.col('country')!=train_country);tt={s:truth[s] for s in tq['entity_id']};prob=m.booster_.predict(x[mask]);result,pred=scores(test,prob,threshold,tt,tq);m.booster_.save_model(str(out/f'fit-{train_country}.txt'));test.select('source1_entity_id','target_id','label').with_columns(pl.Series('score',prob)).write_parquet(out/f'fit-{train_country}-pair_scores.parquet');pl.DataFrame(trials).write_parquet(out/f'fit-{train_country}-threshold_curve.parquet');reports.append({'train_country':train_country,'train_entities':len(trainids),'threshold_from_training_country_oof':threshold,'heldout_countries':sorted(set(tq['country'])),'scores':result});print(json.dumps(reports[-1]),flush=True)
 (out/'metrics.json').write_text(json.dumps({'experiments':reports,'seconds':time.perf_counter()-start,'scope':'Country label-transfer stress test, no held-country labels select threshold. Fixed external fold0/unowned IDF contains known-country text; not a fully unseen-country-text test and not a France performance forecast. Fold4 closed.'},indent=2))
if __name__=='__main__':main()
