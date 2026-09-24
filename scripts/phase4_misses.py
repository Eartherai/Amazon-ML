"""Audit all unretrieved positives in a completed natural-prevalence run."""
from pathlib import Path
import argparse,json
import duckdb,polars as pl
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--sample',default='A');a=p.parse_args();root=Path(a.run);out=root/'missed_links';out.mkdir(exist_ok=False)
db=duckdb.connect('artifacts/audit.duckdb',read_only=True,config={'threads':2,'memory_limit':'2GB'})
db.execute('CREATE TEMP TABLE q AS SELECT * FROM read_parquet(?)',[f'artifacts/validation/phase4-v001/P4-SAMPLE-{a.sample}.parquet']);db.execute('CREATE TEMP TABLE candidates AS SELECT source1_entity_id,target_id FROM read_parquet(?)',[str(root/'pair_scores.parquet')]);cur=db.execute('''SELECT p.*,s.n source_name,s.a source_address,t.n target_name,t.a target_address
FROM positive_features p JOIN q ON q.entity_id=p.source1_entity_id JOIN s1_normalized s ON s.entity_id=p.source1_entity_id JOIN targets_normalized t ON t.entity_id=p.target_id
ANTI JOIN candidates c ON c.source1_entity_id=p.source1_entity_id AND c.target_id=p.target_id''');f=pl.DataFrame(cur.fetchall(),schema=[c[0] for c in cur.description],orient='row').with_columns((pl.col('name_jw')<.7).alias('weak_name'),(pl.col('address_jw')<.7).alias('weak_address'),((pl.col('name_jw')<.7)&(pl.col('address_jw')<.7)).alias('both_weak'),(pl.col('numeric_both_present')&~pl.col('numeric_overlap')).alias('numeric_disagreement'));f.write_parquet(out/'all_missed_links.parquet')
patterns=['weak_name','weak_address','both_weak','numeric_disagreement','nonascii_target_name','missing_address','acronym_equal','exact_name','exact_address'];report={'missed_links':len(f),'patterns':{c:int(f[c].sum()) for c in patterns},'countries':f.group_by('country').len().to_dicts(),'scope':'Natural-prevalence nested OOF candidate misses; overlapping descriptive bins, not inferred business causes'};(out/'metrics.json').write_text(json.dumps(report,indent=2))
with Path('docs/MISSED_LINK_ANALYSIS.md').open('a') as d:
 d.write(f'\n## Phase4 {a.sample}: natural-prevalence candidate misses\n\nRun {root}: {len(f)} missed true links. All preserved in missed_links/all_missed_links.parquet.\n\n| Observable pattern | Links |\n|---|---:|\n');d.writelines(f'| {k} | {v} |\n' for k,v in report['patterns'].items());d.write('\nThe same pattern definitions and unresolved-cause cautions above apply. Next retrieval hypothesis: transliteration-aware names and address rescue, evaluated by marginal coverage on this development population; candidate-cap cause still requires deeper rankings.\n')
print(json.dumps(report))
