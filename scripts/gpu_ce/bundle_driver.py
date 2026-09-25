"""Concatenate many small result parquet parts into a few bundles (runs as a CPU SageMaker job)."""
import glob, json, os
from pathlib import Path
import pandas as pd
cfg = json.loads(Path("/opt/ml/input/data/code/config.json").read_text())
out = Path("/opt/ml/output/data"); out.mkdir(parents=True, exist_ok=True)
import boto3
s3 = boto3.client("s3"); b, k = cfg["s3_out"].replace("s3://", "").split("/", 1)
for name, spec in cfg["bundles"].items():
    files = sorted(f for f in glob.glob(f"/opt/ml/input/data/{spec['channel']}/*.parquet") if spec.get("exclude", "@@") not in os.path.basename(f))
    df = pd.concat([pd.read_parquet(f, columns=spec["cols"]) for f in files], ignore_index=True)
    p = out / f"{name}.parquet"; df.to_parquet(p, index=False, compression="zstd")
    s3.upload_file(str(p), b, k.rstrip("/") + f"/{name}.parquet")
    print(json.dumps({"bundle": name, "files": len(files), "rows": len(df), "bytes": p.stat().st_size}), flush=True)
