"""Export retrieval-evaluation labels only for unlocked folds0–3."""
from pathlib import Path
import json
import duckdb
from upload_verified import sha
root=Path('artifacts/cloud/phase5/retrieval-eval-v001');root.mkdir(parents=True,exist_ok=False)
con=duckdb.connect('artifacts/audit.duckdb',read_only=True,config={'threads':2,'memory_limit':'2GB'})
con.execute('COPY (SELECT v.source1_entity_id, v.country, v.fold, v.n_matches FROM validation_folds v WHERE v.fold IN (0,1,2,3) ORDER BY v.source1_entity_id) TO ? (FORMAT PARQUET,COMPRESSION ZSTD)',[str(root/'queries.parquet')])
con.execute('COPY (SELECT p.source1_entity_id,p.target_id FROM positive_pairs p JOIN validation_folds v USING(source1_entity_id) WHERE v.fold IN (0,1,2,3) ORDER BY p.source1_entity_id,p.target_id) TO ? (FORMAT PARQUET,COMPRESSION ZSTD)',[str(root/'truth.parquet')])
manifest={'scope':'Unlocked folds0–3 ONLY. Fold4 labels intentionally excluded until PRE-FOLD4-FINAL. Full-target pools are retained.','query_count':con.execute('SELECT count(*) FROM validation_folds WHERE fold IN (0,1,2,3)').fetchone()[0],'files':[{'name':p.name,'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(root.iterdir())]}
(root/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest))
