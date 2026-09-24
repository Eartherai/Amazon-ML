"""Publish a measured Phase4 checkpoint from immutable completed run artifacts."""
from pathlib import Path
import argparse,json,csv,subprocess
from datetime import datetime,timezone
import numpy as np,polars as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--run',default='outputs/oof/P4-A-001');p.add_argument('--retrieval',default='outputs/candidates/P4-A-001');a=p.parse_args();root=Path(a.run);m=json.loads((root/'metrics.json').read_text());r=json.loads((Path(a.retrieval)/'metrics.json').read_text());dec=json.loads((root/'decisions/metrics.json').read_text()) if (root/'decisions/metrics.json').exists() else None
lines=['# Phase4 measured OOF checkpoint','',f'Run: {root}. {m["entities"]:,} natural-prevalence entities, three original entity folds; fold4 CLOSED. Candidate retrieval used all10,320,219 training targets. IDF excluded all OOF-owned targets. Thresholds were selected from inner OOF only.','', '| Outer fold | S1 | Singleton rate | Threshold | Macro F0.5 | Precision | Recall | Singleton F0.5 | India | US |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for f in m['folds']:
 s=f['scores'];o=s['overall'];lines.append(f'| {f["fold"]} | {f["entities"]} | {f["singleton_rate"]:.4f} | {f["threshold"]:.2f} | {o["macro_f0_5"]:.6f} | {o["micro_precision"]:.6f} | {o["micro_recall"]:.6f} | {o["singleton_f0_5"]:.6f} | {s["India"]["macro_f0_5"]:.6f} | {s["US"]["macro_f0_5"]:.6f} |')
lines+=['',f'Pooled macro: **{m["pooled"]["macro_f0_5"]:.6f}**. Fold mean/std: {m["fold_macro_mean"]:.6f} / {m["fold_macro_std"]:.6f}; min/max: {m["fold_macro_min"]:.6f} / {m["fold_macro_max"]:.6f}. S1 bootstrap95% interval: {m["bootstrap_95_ci"]}. The interval is conditional on the fitted procedure, not model-refit uncertainty.','', 'Source-specific macro scores (all S1 entities retained):']
for f in m['folds']:lines.append(f'- Fold{f["fold"]}: S2 {f["scores"]["S2"]["macro_f0_5"]:.6f}; S3 {f["scores"]["S3"]["macro_f0_5"]:.6f}.')
lines+=['','## Candidate coverage','', '```json',json.dumps(r['metrics'],indent=2),'```','', '## Nested decision comparisons','']
if dec:
 lines+=['| Method | Pooled macro | Precision | Recall | Singleton |','|---|---:|---:|---:|---:|']
 for name,s in dec['pooled'].items():lines.append(f'| {name} | {s["macro_f0_5"]:.6f} | {s["micro_precision"]:.6f} | {s["micro_recall"]:.6f} | {s["singleton_f0_5"]:.6f} |')
lines+=['', 'These are development OOF comparisons. Selecting a method consumes this evidence; fold4 remains the final one-time check after the whole decision procedure is frozen. No leaderboard submission is justified by this checkpoint alone. Larger coverage, candidate rescue, ablations, training-scale and final test inference are still pending.']
Path('docs/PHASE4_CHECKPOINT.md').write_text('\n'.join(lines)+'\n')
with Path('docs/OOF_VALIDATION.md').open('a') as f:f.write('\n## Completed measured run\n\nSee [Phase4 checkpoint](PHASE4_CHECKPOINT.md) for per-fold natural prevalence, macro/P/R, source/country scores and bootstrap interval.\n')
fig,ax=plt.subplots(1,2,figsize=(12,4.5));fs=m['folds'];ax[0].bar([str(f['fold']) for f in fs],[f['scores']['overall']['macro_f0_5'] for f in fs],color='#266e93');ax[0].axhline(.9,color='#b25626',ls='--',label='Investigation target0.90');ax[0].set_ylim(.75,1);ax[0].set_xlabel('Outer entity fold');ax[0].set_ylabel('Macro F0.5');ax[0].legend();ax[0].set_title('Nested thresholds; natural prevalence')
for f in fs:
 curve=pl.read_parquet(root/f'FOLD-{f["fold"]}'/'threshold_curves.parquet');ax[1].plot(curve['threshold'],curve['macro_f0_5'],label=f'Outer{f["fold"]}: inner only')
ax[1].set_xlabel('Pair threshold');ax[1].set_ylabel('Inner OOF macro F0.5');ax[1].legend();ax[1].set_title('Threshold selection never uses outer labels');fig.tight_layout();fig.savefig('docs/figures/phase4_oof.png',dpi=160);plt.close(fig)
now=datetime.now(timezone.utc).isoformat()
with Path('TIMELINE.md').open('a') as f:f.write(f'\n## {now} — Codex measured Phase4 checkpoint\n- Run {root}; pooled nested OOF macro {m["pooled"]["macro_f0_5"]:.6f}; CI {m["bootstrap_95_ci"]}; entities {m["entities"]}; folds3.\n- Retrieval {r["metrics"]}.\n- Decision comparisons recorded in docs/PHASE4_CHECKPOINT.md. Fold4 CLOSED, no submission or paid compute.\n- Next: larger natural sample, error-driven ablations and rescue; retain all outputs.\n')
print(json.dumps({'pooled':m['pooled'],'ci':m['bootstrap_95_ci'],'retrieval':r['metrics']}))
