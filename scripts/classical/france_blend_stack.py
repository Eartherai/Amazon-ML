"""CL-066: France stack probability = W * (self-trained FULL stack) + (1 - W) * (self-trained no-CE stack), both already
logit-shifted so their own fold-3 thresholds map to 0.72 (france_selftrain.py outputs). India/US rows unchanged.
Simulator (US<->India, round 1): W=0.25 beats the production-like stack by +0.0100 / +0.0256. Usage: france_blend_stack.py W"""
import sys
from pathlib import Path
import polars as pl
ROOT = Path(__file__).resolve().parents[2]; w = float(sys.argv[1]); O = ROOT / "outputs/experiments/CL-064"
f = pl.read_parquet(O / "france_p_0.9_0.02.parquet").select("q", "t", pl.col("p_new").alias("pf"))
n = pl.read_parquet(O / "france_p_0.9_0.02_NOCE.parquet").select("q", "t", pl.col("p_new").alias("pn"))
b = f.join(n, on=["q", "t"], how="inner"); assert b.height == f.height == n.height
b = b.select("q", "t", (w * pl.col("pf") + (1 - w) * pl.col("pn")).alias("p_new"))
st = pl.read_parquet(ROOT / "outputs/experiments/CL-014/test-stack-CL-044-v2qbag-stack.parquet").unique(["q", "t"], keep="first")
out = st.join(b, on=["q", "t"], how="left").with_columns(pl.coalesce("p_new", "p").alias("p")).drop("p_new"); assert out.height == st.height
name = ROOT / f"outputs/experiments/CL-014/test-stack-CL-066-frblend-{w}.parquet"; out.write_parquet(name); print(name, b.height, float(b["p_new"].mean()))
