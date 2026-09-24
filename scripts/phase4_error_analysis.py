"""Analyze all false merges from nested-threshold outer OOF predictions."""
from pathlib import Path
import argparse,json
import polars as pl
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();root=Path(a.run);out=root/'error_analysis';out.mkdir(exist_ok=False)
metrics=json.loads((root/'metrics.json').read_text());threshold={r['fold']:r['threshold'] for r in metrics['folds']}
f=pl.read_parquet(str(root/'features/part-*.parquet'));scores=pl.read_parquet(root/'pair_scores.parquet').select('source1_entity_id','target_id','score')
f=f.join(scores,on=['source1_entity_id','target_id'],validate='1:1').with_columns(pl.col('fold').replace_strict(threshold).alias('threshold'))
fp=f.filter((pl.col('label')==0)&(pl.col('score')>=pl.col('threshold'))).with_columns(
 ((pl.col('name_exact')==1)&(pl.col('address_exact')==0)).alias('exact_name_different_address'),
 ((pl.col('address_exact')==1)&(pl.col('name_exact')==0)).alias('exact_address_different_name'),
 (pl.col('numeric_conflict')==1).alias('numeric_disagreement'),
 ((pl.col('name_jw')>=.95)&(pl.col('name_exact')==0)).alias('near_identical_name'),
 ((pl.col('name_translit_exact')==1)&(pl.col('name_exact')==0)).alias('transliteration_collision'),
 ((pl.col('query_address_missing')==1)|(pl.col('target_address_missing')==1)).alias('missing_address'),
 (pl.col('name_token_set')==1).alias('name_token_containment'),
 (pl.col('owner_fold')>=0).alias('different_known_owner'))
fp.sort('score',descending=True).write_parquet(out/'all_false_positive_pairs.parquet');fp.filter(pl.col('score')>=.9).sort('score',descending=True).write_csv(out/'high_confidence_false_positives.csv')
patterns=['exact_name_different_address','exact_address_different_name','numeric_disagreement','near_identical_name','transliteration_collision','missing_address','name_token_containment','different_known_owner'];rows=[]
for pattern in patterns:
 s=fp.filter(pl.col(pattern));rows.append({'pattern':pattern,'pairs':len(s),'score_min':s['score'].min(),'score_median':s['score'].median(),'score_max':s['score'].max(),'India':s.filter(pl.col('country')=='India').height,'US':s.filter(pl.col('country')=='US').height,'S2':s.filter(pl.col('target_s2')==1).height,'S3':s.filter(pl.col('target_s2')==0).height})
pl.DataFrame(rows).write_csv(out/'pattern_counts.csv');summary={'all_false_positive_pairs':len(fp),'high_confidence_score_ge_0_9':fp.filter(pl.col('score')>=.9).height,'patterns':rows,'scope':'Nested-threshold outer OOF; overlapping descriptive patterns, not proven causes'};(out/'metrics.json').write_text(json.dumps(summary,indent=2))
text=['# False-merge analysis','',f'Run: {root}. All {len(fp)} false-positive pairs are preserved; {summary["high_confidence_score_ge_0_9"]} have raw model score >=0.90. Scores are not calibrated probabilities. Predictions use each outer fold’s inner-selected threshold.','', '| Observable pattern | Pairs | Median score | India | US |','|---|---:|---:|---:|---:|']
for r in sorted(rows,key=lambda r:r['pairs'],reverse=True):text.append(f'| {r["pattern"]} | {r["pairs"]} | {r["score_median"]} | {r["India"]} | {r["US"]} |')
text+=['','Patterns overlap. Different-known-owner means a target has a different labeled owner; it does not justify imposing a global ownership constraint without validation. Chain/franchise, generic business, and legal-suffix identity causes cannot be established reliably by these flags and remain unclassified.','', 'Next controlled hypotheses: nested emptiness decision, numeric/name-address feature ablations, targeted negatives by observed pattern. No blanket rejection rules have been introduced.']
Path('docs/FALSE_MERGE_ANALYSIS.md').write_text('\n'.join(text)+'\n');print(json.dumps(summary))
