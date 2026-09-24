"""Export ALL training S1 text with no labels, plus known-good frozen IDF."""
from pathlib import Path
import json,shutil
import duckdb
from upload_verified import sha
root=Path('artifacts/cloud/phase5/full-query-v001');root.mkdir(parents=True,exist_ok=False)
con=duckdb.connect('artifacts/audit.duckdb',read_only=True,config={'threads':2,'memory_limit':'2GB'})
con.execute('COPY (SELECT entity_id,country,n,a FROM s1_normalized ORDER BY entity_id) TO ? (FORMAT PARQUET,COMPRESSION ZSTD)',[str(root/'queries.parquet')])
for name in ['name_char3_idf.npz','address_char3_idf.npz']:shutil.copy2(Path('outputs/candidates/P4-A-001')/name,root/name)
counts=con.execute('SELECT country,count(*) FROM s1_normalized GROUP BY country ORDER BY country').fetchall()
manifest={'scope':'All training S1 text ONLY; no ground truth, no match counts, no Fold4 label metrics','queries_by_country':dict(counts),'files':[{'name':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(root.iterdir())]}
(root/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest))
