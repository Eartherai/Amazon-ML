"""Compare bounded Parquet shard sizes using the same measured record prefix."""
from pathlib import Path
import json,time,resource
import polars as pl
out=Path('artifacts/processed/shard-benchmark-v001');out.mkdir(parents=True,exist_ok=False)
frame=pl.read_parquet(['artifacts/processed/prep-v001/train_source1/part-00000.parquet','artifacts/processed/prep-v001/train_source1/part-00001.parquet'])
rows=[]
for n in [375000,750000,1500000]:
 x=frame.head(n);path=out/f'{n}.parquet';t=time.perf_counter();x.write_parquet(path,compression='zstd',compression_level=3,row_group_size=100000);write=time.perf_counter()-t;t=time.perf_counter();read=pl.read_parquet(path);seconds=time.perf_counter()-t
 assert x.equals(read)
 rows.append({'rows':n,'logical_mib':x.estimated_size()/1024**2,'compressed_mib':path.stat().st_size/1024**2,'write_seconds':write,'read_seconds':seconds,'roundtrip_equal':True})
 # Scratch benchmark output is deliberately ephemeral; the source foundation is never deleted.
 path.unlink()
report={'scope':'Same first 1.5M train S1 foundation records; varying prefix sizes; one repetition, OS caches warm; not proof of whole-pipeline optimum','measurements':rows,'peak_process_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3,'decision':'Keep existing immutable 750k version for bounded-memory jobs; 1.5M/approximately264MiB logical is a valid larger-shard alternative for future versions, no reason to rewrite completed data just for fewer objects'}
(out/'metrics.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
