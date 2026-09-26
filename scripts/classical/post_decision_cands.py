"""Rewrite a dumped candidate table's decision column from a final matching TSV (after overrides), so further overrides can
be applied sequentially with apply_override_submission.py. Usage: post_decision_cands.py CANDS MATCHING_TSV OUT"""
import sys
import polars as pl
c = pl.read_parquet(sys.argv[1])
m = pl.read_csv(sys.argv[2], separator="\t", quote_char=None, infer_schema_length=0).with_columns(pl.col("matched_entity_ids").fill_null(""))
pairs = m.filter(pl.col("matched_entity_ids") != "").select(pl.col("source1_entity_id").alias("q"), pl.col("matched_entity_ids").str.split(",").alias("t")).explode("t").with_columns(pl.lit(1).alias("dec_new"))
out = c.join(pairs, on=["q", "t"], how="left").with_columns(pl.col("dec_new").fill_null(0).cast(pl.Int64).alias("decision")).drop("dec_new")
miss = pairs.join(c.select("q", "t"), on=["q", "t"], how="anti").height
assert miss == 0, f"{miss} matched pairs missing from candidate table"
out.write_parquet(sys.argv[3]); print(out.height, int(out["decision"].sum()), "changed", int((out["decision"] != c["decision"]).sum()))
