"""Export unlabeled target text and frozen retrieval references for EC2 parity."""
from pathlib import Path
import hashlib,json,shutil
import duckdb
root=Path('artifacts/cloud/phase5/index-input-v001')
root.mkdir(parents=True,exist_ok=False)
con=duckdb.connect('artifacts/audit.duckdb',read_only=True,config={'threads':2,'memory_limit':'2GB'})
con.execute('COPY (SELECT entity_id,country,n,a FROM targets_normalized ORDER BY country,entity_id) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)',[str(root/'targets.parquet')])
for name in ['name_char3_idf.npz','address_char3_idf.npz','name_char3.parquet','address_char3.parquet']:
 shutil.copy2(Path('outputs/candidates/P4-A-001')/name,root/name)
con.execute("COPY (SELECT entity_id,country,n,a FROM read_parquet('artifacts/validation/phase4-v001/P4-SAMPLE-A.parquet') ORDER BY entity_id) TO ? (FORMAT PARQUET,COMPRESSION ZSTD)",[str(root/'queries.parquet')])
manifest={'scope':'All training targets; unlabeled development query text; frozen IDF [-1,0]. No ground truth or Fold4 labels.','files':[]}
for p in sorted(root.iterdir()):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 manifest['files'].append({'name':p.name,'sha256':h.hexdigest(),'bytes':p.stat().st_size})
(root/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest))
