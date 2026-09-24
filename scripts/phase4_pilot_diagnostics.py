"""Characterize frozen pilot misses and marginal retrieval costs, not OOF evidence."""
from pathlib import Path
import json
import duckdb,polars as pl
out=Path('outputs/analysis/P4-PILOT-001');out.mkdir(parents=True,exist_ok=False)
db=duckdb.connect('artifacts/audit.duckdb',read_only=True,config={'threads':2,'memory_limit':'2GB'})
qpath='outputs/candidates/TOKEN-001/run-002/pilot_queries.parquet'
db.execute('CREATE TEMP TABLE q AS SELECT * FROM read_parquet(?)',[qpath])
db.execute("CREATE TEMP TABLE candidates AS SELECT * FROM read_parquet('outputs/candidates/UNION-003/candidates.parquet')")
miss=db.execute('''SELECT p.*,s.n source_name,s.a source_address,t.n target_name,t.a target_address
 FROM positive_features p JOIN q ON q.entity_id=p.source1_entity_id
 JOIN s1_normalized s ON s.entity_id=p.source1_entity_id JOIN targets_normalized t ON t.entity_id=p.target_id
 ANTI JOIN candidates c ON c.source1_entity_id=p.source1_entity_id AND c.target_id=p.target_id''')
miss=pl.DataFrame(miss.fetchall(),schema=[c[0] for c in miss.description],orient='row')
miss=miss.with_columns((pl.col('name_jw')<.7).alias('weak_name'),(pl.col('address_jw')<.7).alias('weak_address'),((pl.col('name_jw')<.7)&(pl.col('address_jw')<.7)).alias('both_weak'),(pl.col('numeric_both_present')&~pl.col('numeric_overlap')).alias('numeric_disagreement'))
miss.write_parquet(out/'missed_links.parquet')
cols=['weak_name','weak_address','both_weak','numeric_disagreement','nonascii_target_name','missing_address','acronym_equal','accent_fold_equal','exact_name','exact_address']
counts={c:int(miss[c].sum()) for c in cols}
truth={i:set() for i in pl.read_parquet(qpath)['entity_id']}
for q,t in db.execute('SELECT p.source1_entity_id,p.target_id FROM positive_pairs p JOIN q ON q.entity_id=p.source1_entity_id').fetchall():truth[q].add(t)
members=pl.read_parquet('outputs/candidates/UNION-003/route_membership.parquet');order=['name_char3','address_char3','token_union','name_translit_char3','name_char4','address_char4','name_char5','address_char5','token_expanded']
actual=set(members['route']);order=[r for r in order if r in actual]+sorted(actual-set(order));found={q:set() for q in truth};rows=[];tp=complete=total=0
for route in order:
 for q,t in members.filter(pl.col('route')==route).select('source1_entity_id','target_id').iter_rows():found[q].add(t)
 ntp=sum(len(ts&found[q]) for q,ts in truth.items());nc=sum(bool(ts) and ts<=found[q] for q,ts in truth.items());n=sum(map(len,found.values()));gain=ntp-tp
 rows.append({'route':route,'new_true_links':gain,'new_complete_entities':nc-complete,'new_candidates':n-total,'candidates_per_new_link':(n-total)/gain if gain else None,'cumulative_link_recall':ntp/sum(map(len,truth.values())),'cumulative_candidates':n})
 tp,complete,total=ntp,nc,n
pl.DataFrame(rows).write_csv(out/'route_utility.csv');(out/'metrics.json').write_text(json.dumps({'scope':'Repeated Phase3 development pilot, 1000 balanced-country queries, not OOF','missed_links':len(miss),'overlapping_pattern_counts':counts,'route_order':order},indent=2))
text=['# Missed-link analysis','',f'Frozen Phase3 broad union misses {len(miss)} of 3,449 labeled links. This is the repeatedly inspected 1,000-query pilot, not fresh Phase4 OOF. Pattern flags overlap; they are not proven causes.','', '| Observable pattern | Missed links |','|---|---:|']+[f'| {k} | {v} |' for k,v in counts.items()]
text += ['', 'Weak means Jaro-Winkler <0.70, a descriptive bin, not a decision rule. Numeric disagreement means both addresses contain numbers but no shared numeric token. Non-ASCII is an observable script flag, not proof of transliteration failure.', '', 'Cap truncation versus retrieval-score failure remains unresolved: top100 artifacts omit deeper rankings. No chain/franchise identity or business-name-change claim can be established from these features alone. Review raw competition text locally and test wider retrieval only on a controlled subset.', '', 'The name/address char3-first incremental table is in outputs/analysis/P4-PILOT-001/route_utility.csv. Ordering affects marginal attribution; compare route removals as well. Large-scale Phase4 misses will be appended separately.']
Path('docs/MISSED_LINK_ANALYSIS.md').write_text('\n'.join(text)+'\n')
print(json.dumps({'misses':len(miss),'counts':counts,'routes':rows}))
