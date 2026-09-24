"""Append completed Phase4-A experiments without overwriting historical rows."""
from pathlib import Path
import csv,json
from datetime import datetime,timezone
path=Path('EXPERIMENTS.csv');old=list(csv.DictReader(path.open()));fields=list(old[0]);existing={r['experiment_id'] for r in old}
m=json.loads(Path('outputs/oof/P4-A-001/metrics.json').read_text());r=json.loads(Path('outputs/candidates/P4-A-001/metrics.json').read_text());d=json.loads(Path('outputs/oof/P4-A-001/decisions/metrics.json').read_text());now=datetime.now(timezone.utc).isoformat()
entries=[]
for eid,model,score,runtime,ram,notes in [('EXP-006','retrieval oracle',{},r['seconds'],r['peak_rss_gib'],'5k natural prevalence; full10.32M target pool; IDF fits only fold0/unowned; oracle not model score.'),('EXP-007','LightGBM multiview nested OOF',m['pooled'],m['runtime_seconds'],m['peak_rss_gib'],'5k/3outer folds; threshold selected on2inner folds per outer; CI '+str(m['bootstrap_95_ci'])+'; fold4 CLOSED.'),('EXP-008','Nested calibration/emptiness comparison',d['pooled']['raw'],'','','No promotion: raw .9234065; Platt .9227854; isotonic .9234973; pair+empty .9231134. Development comparisons, not final holdout.')]:
 if eid in existing:continue
 row={k:'' for k in fields};row.update(experiment_id=eid,timestamp=now,git_commit=m['git_commit'],data_version='raw-v001; phase4-v001/A',validation_split='Natural S1 folds1,2,3; fold4 closed',blocking_version='P4-A-001 name/address char3; fitfolds -1,0',candidate_recall=r['metrics']['link_recall'],candidate_pairs=r['metrics']['candidate_pairs'],features_version='GBDT-004 45 features',model=model,model_version='LightGBM4.7.0',threshold='inner OOF:0.83,0.82,0.81',precision=score.get('micro_precision',''),recall=score.get('micro_recall',''),f0_5=score.get('macro_f0_5',''),singleton_f0_5=score.get('singleton_f0_5',''),non_singleton_f0_5=score.get('non_singleton_f0_5',''),runtime_seconds=runtime,peak_ram_gb=ram,compute_device='M5 CPU;2threads',aws_cost_estimate=0,notes=notes,status='completed_development');entries.append(row)
with path.open('a',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=fields);writer.writerows(entries)
for filename,data in [('P4-A-001-retrieval.json',r['config']),('P4-A-001-oof.json',{'query_sha256':m['query_sha256'],'git_commit':m['git_commit'],'baseline':'configs/baselines/BASELINE-P4-001.yaml','outer_folds':[1,2,3],'inner_folds':2,'threshold_selection':'inner-only','candidate_path':'outputs/candidates/P4-A-001','fold4':'CLOSED'})]:
 target=Path('configs/phase4')/filename
 if not target.exists():target.write_text(json.dumps(data,indent=2))
print('Recorded',len(entries),'new rows')
