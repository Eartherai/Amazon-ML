"""Test-side population summary of CL-005 top-12 v2 stage-2 probabilities per country (no labels).

Inputs (read-only): outputs/experiments/CL-003/test_probs/CL-005-top12v2-v1.parquet (q, t, base, p; rows are the
sorted concatenation of outputs/experiments/CL-005/features-v2/*.npz, verified here row by row),
the feature shards (selected columns only), student_resource/dataset/test/test_source1.tsv.
Output: outputs/analysis/public_gap/test_summary.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/classical"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage2_features_v2 import NAMES  # noqa: E402
from common import FEATS, summarize  # noqa: E402

TEST = ROOT / "student_resource/dataset/test"
FEAT_DIR = ROOT / "outputs/experiments/CL-005/features-v2"
PROBS = ROOT / "outputs/experiments/CL-003/test_probs/CL-005-top12v2-v1.parquet"
OUT = ROOT / "outputs/analysis/public_gap"


def main():
    t0 = time.time()
    probs = pl.read_parquet(PROBS, columns=["q", "t", "base", "p"])
    s1 = pl.read_csv(TEST / "test_source1.tsv", separator="\t", quote_char=None, infer_schema_length=0,
                     columns=["entity_id", "country"])
    idx = [NAMES.index(f) for f in FEATS]
    shards = sorted(FEAT_DIR.glob("*.npz"))
    off = 0
    res = {"source": {"probs": str(PROBS.relative_to(ROOT)), "features": str(FEAT_DIR.relative_to(ROOT)),
                      "threshold": 0.67, "shards": len(shards), "probs_rows": len(probs)}, "countries": {}}
    per_country: dict[str, list] = {}
    mism = 0
    for path in shards:
        c = path.name.split("-s")[0]
        z = np.load(path)
        q, t = z["q"], z["t"]
        n = len(q)
        sl = probs.slice(off, n)
        ok = (sl["q"] == pl.Series(q)).all() and (sl["t"] == pl.Series(t)).all()
        base_ok = np.allclose(sl["base"].to_numpy(), z["X"][:, 0].astype(np.float64), atol=1e-6)
        if not (ok and base_ok):
            mism += 1
            raise AssertionError(f"alignment mismatch at {path.name}")
        per_country.setdefault(c, []).append((off, n, z["X"][:, idx].copy()))
        off += n
    assert off == len(probs), (off, len(probs))
    res["source"]["alignment"] = "all shards row-aligned with parquet (q, t, base)"
    for c in sorted(per_country):  # no pooled "All" here: keeps peak RSS low; targets are country-specific
        parts = per_country.pop(c)
        rows = np.concatenate([np.arange(o, o + n) for o, n, _ in parts])
        Xs = np.concatenate([x for _, _, x in parts])
        ids = s1.filter(pl.col("country") == c)["entity_id"].to_list()
        m = pl.DataFrame({"q": ids, "s1i": np.arange(len(ids), dtype=np.int64)})
        sub = pl.DataFrame({"row": rows, "q": probs["q"].gather(rows)}).join(m, on="q", how="left", maintain_order="left")
        assert sub["s1i"].null_count() == 0
        tcode = probs["t"].gather(rows).cast(pl.Categorical).to_physical().to_numpy().astype(np.int64)
        F = {f: Xs[:, j] for j, f in enumerate(FEATS)}
        r = summarize(sub["s1i"].to_numpy(), len(ids), probs["p"].gather(rows).to_numpy(), F, tcode)
        res["countries"][c] = r
        print(c, json.dumps({k: r[k] for k in ("s1", "pred_singleton_pct", "links_mean", "self_est_macro_top12",
                                             "conflict_link_share_pct")}), flush=True)
        del Xs, F, sub, parts
    res["seconds"] = time.time() - t0
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "test_summary.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
