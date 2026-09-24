"""Compare numeric feature addition on nested20k and its newly added15k entities."""
from pathlib import Path
import json
import duckdb,numpy as np,polars as pl
from src.evaluation import evaluate,entity_f05
root=Path('outputs/experiments/P4-NUMERIC-B-001');out=root/'confirmation';out.mkdir(exist_ok=False)
q=pl.read_parquet('artifacts/validation/phase4-v001/P4-SAMPLE-B.parquet');old=set(pl.read_parquet('artifacts/validation/phase4-v001/P4-SAMPLE-A.parquet')['entity_id']);truth={s:set() for s in q['entity_id']};db=duckdb.connect('artifacts/audit.duckdb',read_only=True,config={'threads':2,'memory_limit':'1GB'})
for s,t in db.execute("SELECT p.source1_entity_id,p.target_id FROM positive_pairs p JOIN read_parquet('artifacts/validation/phase4-v001/P4-SAMPLE-B.parquet') q ON q.entity_id=p.source1_entity_id").fetchall():truth[s].add(t)
def read(path):
 return {s:set(ts.split(',')) if ts else set() for s,ts in pl.read_csv(path,separator='\t').iter_rows()}
baseline={}
for fold in [1,2,3]:baseline.update(read(Path(f'outputs/oof/P4-B-001/FOLD-{fold}/entity_predictions.tsv')))
new=read(root/'with_canonical_numeric/entity_predictions.tsv');assert baseline.keys()==new.keys()==truth.keys();report={}
for name,ids in [('all20k',set(truth)),('new15k',set(truth)-old),('previous5k',old)]:
 keys=sorted(ids);t={s:truth[s] for s in keys};base=evaluate(t,{s:baseline[s] for s in keys});updated=evaluate(t,{s:new[s] for s in keys});delta=np.array([entity_f05(truth[s],new[s])-entity_f05(truth[s],baseline[s]) for s in keys]);rng=np.random.default_rng(20260925);ci=np.quantile([rng.choice(delta,len(delta),replace=True).mean() for _ in range(2000)],[.025,.975]);report[name]={'baseline':base,'numeric':updated,'paired_delta':float(delta.mean()),'paired_bootstrap_95_ci':ci.tolist()}
report['scope']='Development confirmation; new15k excludes previously inspected5k entities, but same country distribution and shared training pool. Not fold4 or France evidence.';(out/'metrics.json').write_text(json.dumps(report,indent=2))
lines=['# Numeric preprocessing confirmation','',report['scope'],'','| Population | Baseline macro | Numeric macro | Paired delta | 95% CI |','|---|---:|---:|---:|---|']
for name in ['all20k','new15k','previous5k']:
 r=report[name];lines.append(f'| {name} | {r["baseline"]["macro_f0_5"]:.6f} | {r["numeric"]["macro_f0_5"]:.6f} | {r["paired_delta"]:+.6f} | {r["paired_bootstrap_95_ci"]} |')
lines+=['','Canonical digits are alternative comparison features. Raw values and original comparisons remain; numeric equality never automatically accepts a match. Thresholds are reselected inside each outer training partition. Country-transfer gains were modest and unseen-France robustness remains unresolved.']
Path('docs/NUMERIC_CONFIRMATION.md').write_text('\n'.join(lines)+'\n');print(json.dumps(report))
