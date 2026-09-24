"""Create a generic library-only transliteration view for all non-ASCII target names.

No labels, dictionaries or outside records are consulted. Native view is retained.
The cached keys use the recorded legacy NFC/lower normalization; transformed text
uses the explicit new BASE recipe. Backend/system metadata prevents silent reuse.
"""
from pathlib import Path
import argparse,hashlib,json,platform,resource,subprocess,time,unicodedata
from datetime import datetime,timezone
import duckdb
import polars as pl
from src.preprocessing import BASE,normalize,config_hash


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--binary',type=Path,default=Path('artifacts/transliterate_probe'));a=p.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=True)
    if (a.output_dir/'manifest.json').exists():raise FileExistsError('Choose a fresh cache version')
    started=time.perf_counter();db=duckdb.connect('artifacts/audit.duckdb',read_only=True,config={'threads':2,'memory_limit':'2GB'})
    cur=db.execute("SELECT DISTINCT n FROM targets_normalized WHERE regexp_matches(n,'[^\\x00-\\x7F]') ORDER BY n")
    files=[];total=0
    while batch:=cur.fetchmany(25000):
        names=[x[0] for x in batch]
        result=subprocess.run([str(a.binary.resolve())],input=json.dumps(names,ensure_ascii=False)+'\n',capture_output=True,text=True,check=True)
        transformed=json.loads(result.stdout)
        if len(transformed)!=len(names):raise ValueError('Transliteration length mismatch')
        f=pl.DataFrame({'n':names,'transliterated':[normalize(x,BASE) for x in transformed]})
        path=a.output_dir/f'part-{len(files):05d}.parquet'
        if path.exists():raise FileExistsError(path)
        f.write_parquet(path,compression='zstd');files.append({'file':path.name,'rows':len(f),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()});total+=len(f)
        print(json.dumps({'unique_names':total,'elapsed':round(time.perf_counter()-started,2)}),flush=True)
    db.close()
    # Source shards persist; consolidated artifact is a convenience for bounded joins.
    pl.scan_parquet([str(a.output_dir/x['file']) for x in files]).sink_parquet(a.output_dir/'name_map.parquet',compression='zstd')
    manifest={'created_at':datetime.now(timezone.utc).isoformat(),'scope':'All distinct non-ASCII light-normalized names across full training S2/S3; no labels used; ASCII names pass through unchanged','backend':'macOS Foundation applyingTransform Any-Latin; Latin-ASCII','platform':platform.platform(),'unicode_python':unicodedata.unidata_version,'binary_sha256':hashlib.sha256(a.binary.read_bytes()).hexdigest(),'config_hash':config_hash({'operations':BASE,'transliteration':'Any-Latin; Latin-ASCII'}),'rows':total,'parts':files,'runtime_seconds':time.perf_counter()-started,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3,'linux_equivalence':'Not yet verified; keep platform/backend pinned','view':'name_transliterated; never replaces n'}
    (a.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest))
if __name__=='__main__':main()
