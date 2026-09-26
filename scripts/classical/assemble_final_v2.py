"""CL-025: final test assembly = stage-2 + e5-base CE stack (fold-3 trained) on sparse top-12,
max-probability ownership, dense bi-encoder top-10 rescue (text-only stage-2 p >= DENSE_THR, unclaimed
targets, ownership among rescues), raw duplicate expansion, and a France policy switch.

Inputs are saved artifacts only (no retrieval or model training beyond the small stacker):
  --ce-held    fold-3 CE held logits parquet (q,t,logit)  [OOF for fold-3 S1]
  --ce-test    dir of CE test parts per shard (q,t,base,logit)
  --dense      parquet of test dense pairs scored by text-only stage-2 (q,t,cos,rank,p_text)
  --france     'new' (pipeline everywhere) or 'vsafe' (France rows copied from the VSAFE upload file)
Outputs matching TSV (test order), candidate additions parquet, provenance JSON.
"""
import argparse, glob, hashlib, json
from collections import defaultdict
from pathlib import Path
import duckdb, numpy as np, polars as pl, lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]
TEST = ROOT / "student_resource/dataset/test"
HEADER = "source1_entity_id\tmatched_entity_ids\n"
OPTS = "delim='\t',header=true,quote='',escape='',all_varchar=true"
PARAMS = dict(objective="binary", n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.9, verbose=-1, n_jobs=8, random_state=0, deterministic=True, force_col_wise=True)


def logit(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def ce_ctx(K, lv):
    by = defaultdict(list)
    for i, (q, _) in enumerate(K):
        if not np.isnan(lv[i]): by[q].append(i)
    rk = np.full(len(K), np.nan, np.float32); gp = np.full(len(K), np.nan, np.float32)
    for q, idx in by.items():
        v = lv[idx]; o = np.argsort(-v); rk[np.array(idx)[o]] = np.arange(1, len(idx) + 1); gp[idx] = v.max() - v
    return rk, gp


def f05(P, T):
    if not P and not T: return 1.0
    tp = len(P & T)
    if not tp: return 0.0
    a, r = tp / len(P), tp / len(T); return 1.25 * a * r / (0.25 * a + r)


def own_decide(K, prob, thr):
    best = {}
    for i, (q, t) in enumerate(K):
        if prob[i] >= thr and (t not in best or prob[i] > best[t][1]): best[t] = (q, prob[i])
    out = defaultdict(set)
    for t, (q, _) in best.items(): out[q].add(t)
    return out


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 23), b""): h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    for x in ("--ce-held", "--ce-test", "--dense", "--name"): ap.add_argument(x, required=True)
    ap.add_argument("--france", choices=["new", "vsafe"], default="new")
    ap.add_argument("--dense-thr", type=float, default=0.8)
    ap.add_argument("--salt", default="", help="stacker threshold cross-fit partition salt ('' reproduces CL-025)")
    ap.add_argument("--oof", default="train-oof-top12-v2.npy", help="stage-2 OOF file in CL-003 (v2 or v2q)")
    ap.add_argument("--probs", default="CL-005-top12v2-v1.parquet", help="stage-2 test probabilities in CL-003/test_probs")
    ap.add_argument("--vsafe", default=str(ROOT / "outputs/submissions/UPLOAD_FINAL_VSAFE/UPLOAD_FINAL_VSAFE_matching_results.tsv"))
    ap.add_argument("--france-rescue", default=str(ROOT / "outputs/experiments/CL-012/France-rescue-VSAFE.parquet"))
    a = ap.parse_args()
    # ---- stacker on fold-3 OOF (same features as stack_ce_fold3 'stack_all')
    z = np.load(ROOT / "outputs/experiments/CL-003/trainmat-top12-v2.npz", allow_pickle=True)
    keys, X = z["keys"], z["X"]; p2 = np.load(ROOT / "outputs/experiments/CL-003" / a.oof)
    top = pl.concat([pl.read_parquet(ROOT / f"outputs/experiments/CL-003/s3/results/top12-fold{k}.parquet") for k in (1, 2, 3)])
    fold = dict(zip(top["source1_entity_id"].to_list(), top["fold"].to_list()))
    m3 = np.array([fold[q] == 3 for q in keys[:, 0].tolist()]); keys, X, p2 = keys[m3], X[m3], p2[m3]
    K = list(map(tuple, keys.tolist()))
    lab = {(q, t): l for q, t, l in top.select("source1_entity_id", "target_id", "label").iter_rows()}
    y = np.array([lab[k] for k in K])
    ce = {(q, t): v for q, t, v in pl.read_parquet(a.ce_held).select("q", "t", "logit").iter_rows()}
    lv = np.array([ce.get(k, np.nan) for k in K], np.float32); rk, gp = ce_ctx(K, lv)
    Fm = np.column_stack([logit(p2), X[:, 0], X[:, 2], X[:, 3], X[:, 4], lv, rk, gp]).astype(np.float32)
    qs = sorted({k[0] for k in K})
    gt = pl.read_csv(ROOT / "student_resource/dataset/train/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema_length=0).filter(pl.col("source1_entity_id").is_in(qs))
    truth = {q: set(r.split(",")) if r else set() for q, r in gt.iter_rows()}
    # salted partition: decorrelated from the stage-2 OOF partition (sha256(q) % 3), see CL-043
    part = np.array([int(hashlib.sha256((a.salt + q).encode()).hexdigest(), 16) % 3 for q, _ in K])
    oof = np.zeros(len(K))
    for k in range(3):
        oof[part == k] = lgb.LGBMClassifier(**PARAMS).fit(Fm[part != k], y[part != k]).predict_proba(Fm[part == k])[:, 1]
    def macro(th):
        d = own_decide(K, oof, th); return float(np.mean([f05(d.get(q, set()), truth[q]) for q in qs]))
    grid = {float(th): macro(th) for th in np.arange(0.4, 0.95, 0.02)}
    thr = max(grid, key=grid.get); oof_macro = grid[thr]
    model = lgb.LGBMClassifier(**PARAMS).fit(Fm, y)
    print(json.dumps({"stacker_threshold": float(thr), "fold3_oof_macro_own": oof_macro}), flush=True)
    # ---- test stacked probabilities
    probs = pl.read_parquet(ROOT / "outputs/experiments/CL-003/test_probs" / a.probs).select("q", "t", "p").unique(subset=["q", "t"], keep="first")
    bundled = Path(a.ce_test).is_file()  # one concatenated parquet (q,t,logit) or a dir of per-shard parts
    ce_all = pl.read_parquet(a.ce_test, columns=["q", "t", "logit"]).unique(subset=["q", "t"], keep="first") if bundled else None
    ce_parts = {} if bundled else {Path(f).stem: f for f in glob.glob(str(Path(a.ce_test) / "*.parquet"))}
    TK, TP, missing, ce_hits = [], [], [], 0
    for f in sorted((ROOT / "outputs/experiments/CL-005/features-v2").glob("*.npz")):
        zz = np.load(f); Kt = list(zip(zz["q"].tolist(), zz["t"].tolist())); Xt = zz["X"]
        if bundled or f.stem in ce_parts:
            src = ce_all if bundled else pl.read_parquet(ce_parts[f.stem], columns=["q", "t", "logit"])
            j = pl.DataFrame({"q": zz["q"], "t": zz["t"]}).with_row_index("i").join(src, on=["q", "t"], how="left").sort("i")
            lvt = j["logit"].fill_null(np.nan).to_numpy().astype(np.float32)
        else:
            missing.append(f.stem); lvt = np.full(len(Kt), np.nan, np.float32)
        jp = pl.DataFrame({"q": zz["q"], "t": zz["t"]}).with_row_index("i").join(probs, on=["q", "t"], how="left").sort("i")
        assert jp.height == len(Kt) and jp["p"].null_count() == 0, f.stem
        p2t = jp["p"].to_numpy()
        ce_hits += int((~np.isnan(lvt)).sum()); rkt, gpt = ce_ctx(Kt, lvt)
        Ft = np.column_stack([logit(p2t), Xt[:, 0], Xt[:, 2], Xt[:, 3], Xt[:, 4], lvt, rkt, gpt]).astype(np.float32)
        TK += Kt; TP.append(model.predict_proba(Ft)[:, 1])
    if missing:
        raise SystemExit(f"CE test parts missing for {len(missing)} shards, e.g. {missing[:3]}")
    TP = np.concatenate(TP); print(json.dumps({"test_pairs": len(TK), "ce_hits": ce_hits}), flush=True)
    pl.DataFrame({"q": [k[0] for k in TK], "t": [k[1] for k in TK], "p": TP}).write_parquet(ROOT / f"outputs/experiments/CL-014/test-stack-{a.name}.parquet")
    chosen = own_decide(TK, TP, thr)
    claimed = {t for v in chosen.values() for t in v}
    # ---- dense (+ France rescue) additions: text-only p >= thr, unclaimed targets, ownership among additions
    dn = pl.read_parquet(a.dense).select("q", "t", "p_text")
    fr = pl.read_parquet(a.france_rescue).select("q", "t", pl.col("p").alias("p_text"))
    pool = pl.concat([dn, fr]).group_by("q", "t").agg(pl.col("p_text").max()).filter(pl.col("p_text") >= a.dense_thr).sort(["p_text", "q", "t"], descending=[True, False, False])
    best = {}
    for q, t, p in pool.iter_rows():
        if t in claimed: continue
        if t not in best or p > best[t][1]: best[t] = (q, p)
    added = [(q, t) for t, (q, _) in best.items()]
    for q, t in added: chosen[q].add(t)
    # ---- raw duplicate expansion
    con = duckdb.connect()
    groups = con.execute(f"""WITH t AS (SELECT entity_id, business_name n, business_address a, country c FROM read_csv('{TEST}/test_source2.tsv',{OPTS})
          UNION ALL SELECT entity_id, business_name, business_address, country FROM read_csv('{TEST}/test_source3.tsv',{OPTS})),
        g AS (SELECT n, a, c FROM t GROUP BY n, a, c HAVING count(*) > 1)
        SELECT t.entity_id, t.n, t.a, t.c FROM t JOIN g ON t.n IS NOT DISTINCT FROM g.n AND t.a IS NOT DISTINCT FROM g.a AND t.c=g.c""").fetchall()
    members, key_of = defaultdict(list), {}
    for i, n, ad, c in groups: members[(n, ad, c)].append(i); key_of[i] = (n, ad, c)
    allclaimed = {t for v in chosen.values() for t in v}; dup = []
    for q in list(chosen):
        for t in list(chosen[q]):
            if t in key_of:
                for o in members[key_of[t]]:
                    if o not in allclaimed: allclaimed.add(o); chosen[q].add(o); dup.append((q, o))
    # ---- write
    s1 = pl.read_csv(TEST / "test_source1.tsv", separator="\t", quote_char=None, infer_schema_length=0)
    order, country = s1["entity_id"].to_list(), dict(zip(s1["entity_id"].to_list(), s1["country"].to_list()))
    vs = {}
    with open(a.vsafe) as f:
        assert f.readline() == HEADER
        for line in f:
            q, _, raw = line.rstrip("\n").partition("\t"); vs[q] = raw
    dest = ROOT / "outputs/submissions" / a.name; dest.mkdir(parents=True, exist_ok=False)
    path = dest / "matching_results.tsv"; per = defaultdict(lambda: {"s1": 0, "links": 0, "empty": 0})
    with open(path, "w") as f:
        f.write(HEADER)
        for q in order:
            c = country[q]
            if a.france == "vsafe" and c == "France":
                raw = vs[q]
            else:
                raw = ",".join(sorted(chosen.get(q, set())))
            f.write(q + "\t" + raw + "\n")
            n = len(raw.split(",")) if raw else 0
            per[c]["s1"] += 1; per[c]["links"] += n; per[c]["empty"] += int(n == 0)
    add = pl.DataFrame(added + dup, schema=["q", "t"], orient="row")
    add.write_parquet(dest / "candidate_additions.parquet")
    prov = {"matching_sha256": sha(path), "args": vars(a), "stacker_threshold": float(thr), "fold3_oof_macro_own": oof_macro,
            "test_pairs": len(TK), "ce_hits": ce_hits, "dense_added": len(added), "dup_added": len(dup), "per_country": {k: dict(v) for k, v in per.items()}, "fold4": "CLOSED"}
    (dest / "provenance.json").write_text(json.dumps(prov, indent=2, default=str)); print(json.dumps(prov, indent=1, default=str))


if __name__ == "__main__":
    main()
