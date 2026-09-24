"""Summarize completed feature ablations with paired entity uncertainty."""
from pathlib import Path
import json,hashlib
import numpy as np,polars as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('outputs/oof/P4-A-001');m=json.loads((root/'ablations/metrics.json').read_text());rows=m['variants'];assert len(rows)==8,'Do not publish a partial ablation tournament as complete'
text=['# Phase4 feature ablations','', '5,000 natural-prevalence entities, same full-pool candidates, same three outer folds. Each ablation independently selects thresholds from its two inner folds. These are development comparisons of whole decision procedures.','', '| Removed group | Macro F0.5 | Delta | Paired95% CI |','|---|---:|---:|---|']
for r in rows:text.append(f'| {r["variant"]} | {r["pooled"]["macro_f0_5"]:.6f} | {r["paired_macro_delta"]:+.6f} | {r["paired_entity_bootstrap_95_ci"]} |')
text+=['','No raw-text feature group or separate aggressive/accent/deduplicated preprocessing view exists in the frozen45-feature model. Their absence is not an ablation result. Token-sort/set comparisons are implemented and removed in the token-similarity ablation. Char-similarity removal affects pair similarities and retrieval-score features, while candidate generation remains identical.','', 'Route-count removal has weak evidence if its interval crosses zero; retaining its negligible implementation cost is reasonable pending larger validation. Preserve beneficial groups. Do not interpret each conditional paired bootstrap as a multiple-comparison-corrected significance test.']
Path('docs/PHASE4_ABLATIONS.md').write_text('\n'.join(text)+'\n')
(root/'ablations/provenance.json').write_text(json.dumps({'module_sha256':hashlib.sha256(Path('code/business_entity_resolution/src/models/phase4_ablation.py').read_bytes()).hexdigest(),'parent_metrics_sha256':hashlib.sha256((root/'metrics.json').read_bytes()).hexdigest(),'baseline_sha256':hashlib.sha256(Path('configs/baselines/BASELINE-P4-001.yaml').read_bytes()).hexdigest()},indent=2))
fig,ax=plt.subplots(1,2,figsize=(14,5));ys=np.arange(len(rows));d=np.array([r['paired_macro_delta'] for r in rows]);low=np.array([r['paired_entity_bootstrap_95_ci'][0] for r in rows]);high=np.array([r['paired_entity_bootstrap_95_ci'][1] for r in rows]);ax[0].errorbar(d,ys,xerr=np.stack([d-low,high-d]),fmt='o',color='#266e93',capsize=3);ax[0].set_yticks(ys,[r['variant'].replace('without_','') for r in rows]);ax[0].axvline(0,color='grey',ls='--');ax[0].set_xlabel('Macro F0.5 change after removal');ax[0].set_title('Paired S1 bootstrap95% intervals')
for method in ['raw','platt','isotonic']:
 frames=[pl.read_csv(root/f'decisions/fold-{f}-{method}-reliability.csv') for f in [1,2,3]];f=pl.concat(frames).group_by('bin').agg(((pl.col('mean_score')*pl.col('pairs')).sum()/pl.col('pairs').sum()).alias('x'),((pl.col('positive_rate')*pl.col('pairs')).sum()/pl.col('pairs').sum()).alias('y')).sort('bin');ax[1].plot(f['x'],f['y'],marker='o',label=method)
ax[1].plot([0,1],[0,1],color='grey',ls='--');ax[1].set_xlabel('Mean predicted score in bin');ax[1].set_ylabel('Observed positive fraction');ax[1].legend();ax[1].set_title('Outer OOF reliability; calibrators fit inner only');fig.tight_layout();fig.savefig('docs/figures/phase4_ablations_calibration.png',dpi=160);plt.close(fig)
print(json.dumps([{k:r[k] for k in ['variant','paired_macro_delta','paired_entity_bootstrap_95_ci']} for r in rows]))
