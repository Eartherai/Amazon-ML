"""Unsupervised generic Foundation transliteration cache for all test names."""
from pathlib import Path
import hashlib,json,platform,subprocess,time
from datetime import datetime,timezone
import duckdb,polars as pl
from src.preprocessing import BASE,normalize

def main():
 root=Path('artifacts/cloud/phase5/test-translit-v001');root.mkdir(parents=True,exist_ok=False)
 binary=Path('artifacts/transliterate_probe').resolve();start=time.perf_counter()
 con=duckdb.connect(config={'threads':2,'memory_limit':'2GB'})
 files=[str(p)for p in Path('artifacts/cloud/phase5/test-input-v001').glob('*.parquet')]
 cur=con.execute("SELECT DISTINCT n FROM read_parquet(?) WHERE regexp_matches(n,'[^\\x00-\\x7F]') ORDER BY n",[files])
 parts=[];rows=0
 while batch:=cur.fetchmany(25000):
  names=[r[0]for r in batch]
  r=subprocess.run([str(binary)],input=json.dumps(names,ensure_ascii=False)+'\n',capture_output=True,text=True,check=True)
  mapped=json.loads(r.stdout)
  if len(mapped)!=len(names):raise ValueError('Transliteration row mismatch')
  path=root/f'part-{len(parts):05d}.parquet'
  pl.DataFrame({'n':names,'transliterated':[normalize(x,BASE)for x in mapped]}).write_parquet(path,compression='zstd')
  parts.append(path);rows+=len(names)
  if rows%100000<25000:print(json.dumps({'names':rows,'seconds':time.perf_counter()-start}),flush=True)
 con.close()
 pl.scan_parquet([str(x)for x in parts]).sink_parquet(root/'name_map.parquet',compression='zstd')
 manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'count':rows,'backend':'macOS Foundation Any-Latin; Latin-ASCII','platform':platform.platform(),'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'normalization':BASE,'files':[{'name':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}for p in [root/'name_map.parquet']],'seconds':time.perf_counter()-start,'labels':'none'}
 (root/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest))
if __name__=='__main__':main()
