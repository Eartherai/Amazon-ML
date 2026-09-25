"""Union France rescue route outputs (max text-only p per pair) into one rescue parquet."""
import sys, polars as pl
out, srcs = sys.argv[1], sys.argv[2:]
df = pl.concat([pl.read_parquet(s).select("q", "t", "p") for s in srcs]).group_by("q", "t").agg(pl.col("p").max())
df.write_parquet(out)
print({"pairs": df.height, "ge_0.85": int((df["p"] >= 0.85).sum()), "ge_0.9": int((df["p"] >= 0.9).sum())})
