"""Explain missed positive links after a frozen full-pool candidate pilot."""
import json
from pathlib import Path
import duckdb
import polars as pl

base=Path('outputs/candidates/TOKEN-001/run-002')
c=duckdb.connect('artifacts/audit.duckdb',read_only=True,config={'threads':2,'memory_limit':'2GB'})
c.execute('CREATE TEMP TABLE q AS SELECT * FROM read_parquet(?)',[str(base/'pilot_queries.parquet')])
c.execute('CREATE TEMP TABLE cand AS SELECT * FROM read_parquet(?)',[str(base/'final_candidates.parquet')])
cur=c.execute('''SELECT p.source1_entity_id,p.target_id,q.country,
 c.target_id IS NOT NULL retrieved,coalesce(c.candidate_rank,0) candidate_rank,
 f.name_jw,f.address_jw,f.nonascii_target_name,f.missing_address,f.numeric_overlap,
 q.n source_name,t.n target_name,q.a source_address,t.a target_address
 FROM positive_pairs p JOIN q ON q.entity_id=p.source1_entity_id
 JOIN positive_features f USING(source1_entity_id,target_id)
 JOIN targets_normalized t ON t.entity_id=p.target_id
 LEFT JOIN cand c USING(source1_entity_id,target_id)''')
cols=[x[0] for x in cur.description];frame=pl.DataFrame(cur.fetchall(),schema=cols,orient='row')
frame.write_parquet(base/'positive_retrieval_diagnostics.parquet')
summary=frame.group_by('country','retrieved').agg(pl.len(),pl.col('nonascii_target_name').mean(),pl.col('missing_address').mean(),pl.col('numeric_overlap').mean(),pl.col('name_jw').mean(),pl.col('address_jw').mean()).sort('country','retrieved')
summary.write_csv(base/'missed_link_summary.csv')
frame.filter(~pl.col('retrieved')).sort('name_jw','address_jw').head(50).write_csv(base/'missed_link_examples.tsv',separator='\t')
report=json.loads((base/'metrics.json').read_text());lines=['# Full-target candidate pilot: TOKEN-001','','1,000 fixed fold-0 S1 queries (500 each India/US), all 10,320,219 training targets. The sample has 3,449 true links and 50 singletons. No model fitting. Frequency ranking excludes targets owned by folds 0/4; full-pool frequency is used only as an inference fanout guard.','','| Final candidate cap | Pairs | Link recall | Oracle macro F0.5 ceiling |','|---|---:|---:|---:|']
for key,v in report['metrics'].items():lines.append(f"| {key} | {v['candidate_pairs']:,} | {v['link_recall']:.2%} | {v['oracle_macro_f0_5']:.5f} |")
lines+=['',f"Runtime {report['runtime_seconds']:.2f}s; peak process RSS {report['peak_process_rss_gib']:.2f} GiB. CPU only.",'','The best recall here (77.33%) is inadequate for final candidate generation. The oracle is an impossible perfect matcher restricted to these candidates; it is **not an achieved model score**. Full-pool target coverage does not make 1,000 queries a complete validation run.','','## Implications','','- Only 464/1,000 queries have an eligible rare-name key and 705 have an eligible rare-address key under the current DF≤1,000 cap. Relax fanout selectively and add fuzzy character retrieval.','- The numeric route promotes already retrieved name hits; it does not independently rescue missing candidates.','- Top-100 fusion loses 130 true links compared with the route-capped union. Keep broader candidates until a validated ranker can reduce volume safely.','- 47/50 true singletons still receive candidates. Candidate existence is not evidence of a match.','- India union recall is 80.68%; US is 73.92%. This reflects this particular lexical blocker and balanced sample, not a claim that India is the easier overall task.','','## Failure and repair','','The first query plan exceeded its deliberate 1 GiB scratch cap when correlated UNNEST expanded the full target pool. Preserved failure metadata/source in `outputs/candidates/TOKEN-001/`. The successful `run-002` streams scalar UNNEST through country/source partitions and filters query keys before aggregation. It retained the full target pool and used 1.26 GiB peak RSS.','','## Missed-link morphology','','| Country | Retrieved | Links | Non-ASCII name | Missing address | Numeric overlap | Mean name JW | Mean address JW |','|---|---|---:|---:|---:|---:|---:|---:|']
for r in summary.to_dicts():lines.append(f"| {r['country']} | {r['retrieved']} | {r['len']} | {r['nonascii_target_name']:.1%} | {r['missing_address']:.1%} | {r['numeric_overlap']:.1%} | {r['name_jw']:.3f} | {r['address_jw']:.3f} |")
lines+=['','## Reproduce','','```sh','PYTHONPATH=code/business_entity_resolution .venv/bin/python -m src.blocking.token_candidates --database artifacts/audit.duckdb --config configs/blocking/TOKEN-001.json --output-dir outputs/candidates/TOKEN-001/run-003','```','','Always select a new output directory. The current CLI is a development-pilot scaffold; a configurable test/full-query streaming mode and persistent indexes remain to be built.']
Path('docs/CANDIDATE_PILOT.md').write_text('\n'.join(lines)+'\n');print(summary)
