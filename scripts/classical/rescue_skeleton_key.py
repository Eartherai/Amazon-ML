"""CL-007: exact transliterated core-name skeleton-key rescue route for links outside the top-12.

Key = sorted unique consonant-skeleton tokens of the legal/noise-stripped core name
(anyascii transliteration), blocked within country; blocks with more than CAP
targets are skipped. Only pairs outside the first-stage top-12 are rescue pairs.
A separate classical LightGBM scores rescue pairs; it is trained on the 194k
training S1 (eval S1 and eval-owned targets excluded) and evaluated as an
increment over CL-003 top-12 predictions on the untouched EXP-044 6k at the
pre-declared stage-2 threshold. Fold4 CLOSED (absent from the 200k store/eval).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage2_features_v2 as F  # noqa: E402
import pilot_stage2_exp044 as P  # noqa: E402

P.F = F
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/experiments/CL-007"
CAP = 20
OPTS = "delim='\t',header=true,quote='',escape='',all_varchar=true"
PARAMS = dict(objective="binary", n_estimators=500, learning_rate=0.04, num_leaves=31, min_child_samples=40,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=2.0, verbose=-1, n_jobs=9)
RESCUE_NAMES = [n for n in F.NAMES[9:]] + ["block_targets", "block_s1", "n_strong", "sib_name", "sib_addr", "sib_dup", "top1_base"]


def key_of(name: str) -> str:
    rec_core = F.core_tokens(F.norm(name or ""))
    toks = sorted({F.skeleton_token(t) for t in rec_core if len(t) >= 2})
    k = " ".join(toks)
    return k if len(k) >= 3 else ""


def keys_chunk(rows):
    return [(i, c, key_of(n)) for i, n, c in rows]


def build_keys(split: str) -> Path:
    dest = OUT / f"keys-{split}.parquet"
    if dest.exists():
        return dest
    d = ROOT / f"student_resource/dataset/{split}"
    con = duckdb.connect()
    rows = con.execute(f"""SELECT entity_id, business_name, country FROM read_csv('{d}/{split}_source1.tsv',{OPTS})
        UNION ALL SELECT entity_id, business_name, country FROM read_csv('{d}/{split}_source2.tsv',{OPTS})
        UNION ALL SELECT entity_id, business_name, country FROM read_csv('{d}/{split}_source3.tsv',{OPTS})""").fetchall()
    con.close()
    chunks = [rows[i:i + 200000] for i in range(0, len(rows), 200000)]
    out = []
    with mp.get_context("fork").Pool(9) as pool:
        for part in pool.imap(keys_chunk, chunks):
            out += part
    pl.DataFrame(out, schema=["id", "country", "key"], orient="row").filter(pl.col("key") != "").write_parquet(dest)
    return dest


def rescue_pairs(split: str, s1_ids: set | None, exclude: pl.DataFrame) -> pl.DataFrame:
    k = pl.read_parquet(build_keys(split))
    s1 = k.filter(pl.col("id").str.starts_with("S1-"))
    if s1_ids is not None:
        s1 = s1.filter(pl.col("id").is_in(list(s1_ids)))
    tg = k.filter(~pl.col("id").str.starts_with("S1-"))
    tsize = tg.group_by("country", "key").agg(pl.len().alias("block_targets"))
    ssize = k.filter(pl.col("id").str.starts_with("S1-")).group_by("country", "key").agg(pl.len().alias("block_s1"))
    tg = tg.join(tsize, on=["country", "key"]).filter(pl.col("block_targets") <= CAP)
    pairs = s1.rename({"id": "q"}).join(tg.rename({"id": "t"}), on=["country", "key"]).join(ssize, on=["country", "key"])
    pairs = pairs.join(exclude.select("q", "t"), on=["q", "t"], how="anti")
    return pairs.select("q", "t", "country", "block_targets", "block_s1")


G: dict = {}


def feat_chunk(items):
    rows = []
    for q, t, bt, bs in items:
        a, b = G["s1"][q], G["t"][t]
        v = F.pair_vector(a, b, t.startswith("S2-"), *G["idf"])
        sibs = G["strong"].get(q, [])
        sn = sa = sd = 0.0
        for sid in sibs:
            o = G["t"][sid]
            if b.core and o.core:
                sn = max(sn, F.JaroWinkler.normalized_similarity(" ".join(b.core), " ".join(o.core)))
            if b.addr and o.addr:
                sa = max(sa, F.fuzz.token_set_ratio(" ".join(b.addr), " ".join(o.addr)) / 100)
            if b.raw_name == o.raw_name and b.raw_addr == o.raw_addr:
                sd = 1.0
        rows.append(v + [float(bt), float(bs), float(len(sibs)), sn, sa, sd, G["top1"].get(q, 0.0)])
    return np.asarray(rows, dtype=np.float32)


def featurize(pairs: pl.DataFrame, top: pl.DataFrame, texts_s1: dict, texts_t: dict, idf) -> np.ndarray:
    strong = defaultdict(list)
    top1 = {}
    for q, t, b in top.select("q", "t", "base").iter_rows():
        if b >= 0.9:
            strong[q].append(t)
        top1[q] = max(top1.get(q, 0.0), b)
    G.update({"s1": texts_s1, "t": texts_t, "idf": idf, "strong": strong, "top1": top1})
    items = list(pairs.select("q", "t", "block_targets", "block_s1").iter_rows())
    chunks = [items[i::72] for i in range(72)]
    order = [i for c in range(72) for i in range(c, len(items), 72)]
    with mp.get_context("fork").Pool(9) as pool:
        mats = pool.map(feat_chunk, chunks)
    X = np.vstack([m for m in mats if len(m)])
    inv = np.empty(len(order), dtype=np.int64)
    inv[np.array(order)] = np.arange(len(order))
    return X[inv]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "test"], required=True)
    ap.add_argument("--test-top", type=Path, help="parquet with q,t,base (test top-12 or sidecar)")
    ap.add_argument("--threshold", type=float, default=0.66)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.mode == "train":
        top = pl.concat([pl.read_parquet(ROOT / f"outputs/experiments/CL-003/s3/results/top12-fold{k}.parquet") for k in (1, 2, 3)]) \
            .rename({"source1_entity_id": "q", "target_id": "t", "base_score": "base"})
        ev_truth = {k: set(v) for k, v in json.loads((P.EXP044 / "truth.json").read_text()).items()}
        owned_eval = {t for v in ev_truth.values() for t in v}
        pairs = rescue_pairs("train", set(top["q"].unique().to_list()), top)
        gt = pl.read_csv(P.TRAIN / "train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema_length=0)
        pos = gt.filter(pl.col("matched_entity_ids").is_not_null()).with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids") \
            .rename({"source1_entity_id": "q", "matched_entity_ids": "t"}).with_columns(pl.lit(1, dtype=pl.Int8).alias("y"))
        pairs = pairs.join(pos, on=["q", "t"], how="left").with_columns(pl.col("y").fill_null(0))
        is_eval = pairs["q"].is_in(list(ev_truth))
        tr = pairs.filter(~is_eval & ~pl.col("t").is_in(list(owned_eval)))
        ev = pairs.filter(is_eval)
        print(json.dumps({"rescue_pairs": len(pairs), "train_pairs": len(tr), "train_pos": int(tr["y"].sum()),
                          "eval_pairs": len(ev), "eval_pos": int(ev["y"].sum()), "s": round(time.time() - t0, 1)}), flush=True)
        s1r, tr_rec = P.load_texts(sorted(set(pairs["q"].to_list())), sorted(set(pairs["t"].to_list()) | set(top.filter(pl.col("base") >= 0.9)["t"].to_list())))
        idf = P.s1_idf(ROOT / "outputs/experiments/CL-001/train_s1_idf.json", P.TRAIN / "train_source1.tsv")
        top_rel = top.filter(pl.col("q").is_in(list(set(pairs["q"].to_list()))))
        Xtr = featurize(tr, top_rel, s1r, tr_rec, idf)
        Xev = featurize(ev, top_rel, s1r, tr_rec, idf)
        ytr = tr["y"].to_numpy()
        model = lgb.LGBMClassifier(**PARAMS).fit(Xtr, ytr)
        path = OUT / "rescue-model.txt"
        model.booster_.save_model(str(path))
        pev = model.predict_proba(Xev)[:, 1] if len(ev) else np.array([])
        # increment over CL-003 top12 eval predictions at pre-declared threshold
        z = np.load(ROOT / "outputs/experiments/CL-003/eval6k-top12.npz")
        thr12 = json.loads((ROOT / "outputs/experiments/CL-003/stage2-200k-top12.json").read_text())["threshold"]
        base_pred = defaultdict(set)
        for q, t, p in zip(z["q"].tolist(), z["t"].tolist(), z["p"]):
            if p >= thr12:
                base_pred[q].add(t)
        country = {p["q"]: p["country"] for p in map(json.loads, (P.EXP044 / "pairs.jsonl").open())}
        s1s = sorted(ev_truth)
        e0, per0 = P.evaluate(s1s, base_pred, ev_truth, country)
        res = {"base_top12": e0["macro_f0_5"]}
        for thr in (0.5, 0.6, args.threshold, 0.75, 0.85):
            pred = {s: set(v) for s, v in base_pred.items()}
            added = tp = 0
            for (q, t, y), p in zip(ev.select("q", "t", "y").iter_rows(), pev):
                if p >= thr:
                    pred.setdefault(q, set()).add(t)
                    added += 1
                    tp += y
            e1, per1 = P.evaluate(s1s, pred, ev_truth, country)
            d = np.array([per1[s] - per0[s] for s in s1s])
            rng = np.random.default_rng(0)
            b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(1000)]
            res[f"thr{thr:.2f}"] = {"macro": e1["macro_f0_5"], "delta": float(d.mean()), "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                                    "added": added, "added_true": int(tp)}
        meta = {"cap": CAP, "features": RESCUE_NAMES, "threshold_declared": args.threshold, "eval": res,
                "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "seconds": time.time() - t0, "fold4": "CLOSED"}
        (OUT / "rescue-train.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta["eval"], indent=1))
    else:
        top = pl.read_parquet(args.test_top).select("q", "t", "base")
        pairs = rescue_pairs("test", None, top)
        print(json.dumps({"test_rescue_pairs": len(pairs), "s": round(time.time() - t0, 1)}), flush=True)
        con = duckdb.connect()
        d = ROOT / "student_resource/dataset/test"
        need_t = set(pairs["t"].to_list()) | set(top.filter(pl.col("base") >= 0.9).filter(pl.col("q").is_in(list(set(pairs["q"].to_list()))))["t"].to_list())
        con.execute("CREATE TABLE w(id VARCHAR)")
        con.executemany("INSERT INTO w VALUES (?)", [(x,) for x in need_t | set(pairs["q"].to_list())])
        recs = {}
        for src in ("source1", "source2", "source3"):
            for i, n, a in con.execute(f"SELECT entity_id, business_name, business_address FROM read_csv('{d}/test_{src}.tsv',{OPTS}) JOIN w ON id=entity_id").fetchall():
                recs[i] = F.Record(n or "", a or "")
        idf = json.loads((ROOT / "outputs/experiments/CL-002/test_s1_idf.json").read_text())
        idf = (idf["name"], idf["name_default"], idf["addr"], idf["addr_default"])
        top_rel = top.filter(pl.col("q").is_in(list(set(pairs["q"].to_list()))))
        X = featurize(pairs, top_rel, recs, recs, idf)
        booster = lgb.Booster(model_file=str(OUT / "rescue-model.txt"))
        p = booster.predict(X, num_threads=8)
        pairs.with_columns(pl.Series("p", p)).write_parquet(OUT / "test-rescue-pairs.parquet")
        print(json.dumps({"test_rescue_pairs": len(pairs), "accepted": int((p >= args.threshold).sum()), "s": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
