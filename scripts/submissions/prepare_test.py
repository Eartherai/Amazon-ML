"""Export frozen legacy-normalized test sources without labels or external data."""
from pathlib import Path
import json,hashlib,time,shutil
from datetime import datetime,timezone
import duckdb

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()

def main():
 out=Path('artifacts/cloud/phase5/test-input-v001');out.mkdir(parents=True,exist_ok=False)
 if shutil.disk_usage('.').free<9*1024**3:raise RuntimeError('Insufficient disk reserve')
 con=duckdb.connect('artifacts/audit.duckdb',read_only=True,config={'threads':4,'memory_limit':'3GB'})
 con.execute("CREATE TEMP MACRO norm(x) AS trim(regexp_replace(lower(nfc_normalize(coalesce(x,''))), '[^\\p{L}\\p{M}\\p{N}]+', ' ', 'g'))")
 start=time.perf_counter();report={'created_utc':datetime.now(timezone.utc).isoformat(),'representation':'Frozen audit_data.norm: NFC lowercase LMN punctuation/space; arbitrary country string','labels':'none','sources':{}}
 for kind,table in [('queries','test_source1'),('target2','test_source2'),('target3','test_source3')]:
  file=out/f'{kind}.parquet'
  con.execute(f'COPY (SELECT entity_id,country,norm(business_name) AS n,norm(business_address) AS a FROM {table} ORDER BY entity_id) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)',[str(file)])
  row=con.execute(f'SELECT count(*) FROM {table}').fetchone()[0]
  report['sources'][kind]={'rows':row,'sha256':sha(file),'bytes':file.stat().st_size}
  print(json.dumps({'source':kind,'rows':row,'seconds':time.perf_counter()-start}),flush=True)
 con.close();report['elapsed_seconds']=time.perf_counter()-start
 (out/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
