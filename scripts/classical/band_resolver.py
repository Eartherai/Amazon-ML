"""CL-077: invariant band resolver for an unseen country (France uncertain-band hypothesis), US<->India simulator.

Target predictions: production-like FULL stack trained on the source country (transfer_stack.py DUMP_SYSTEM=FULL):
pred_FULL_{src}_to_{tgt}.parquet (q, t, prob, thr). Resolver: LightGBM on country-invariant canonical text features only
(canonical name/address equality flags and similarities, number agreement, target source, address presence), trained on
the SOURCE country's labeled candidate pairs (candidates_mfu fold tables, all folds). Applied only to target pairs in the
uncertain band [LO, thr) whose target is not owned by the base decisions; adds only, highest resolver score first, one S1
per target. Exact per-S1 macro F0.5 on target fold-3 S1 (sparse decisions, full truth). Target labels only score.
Usage: band_resolver.py SRC
"""
import json, re, sys
from pathlib import Path
import numpy as np, polars as pl, lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "scripts/classical"))
from canon_rule import cname, caddr, base
from transfer_stack import f05, decide

REG = set("hauts de france pays la loire nouvelle aquitaine nord gironde atlantique pas calais".split())
P = dict(objective="binary", n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=100, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.9, reg_lambda=5.0, verbose=-1, n_jobs=6, random_state=0, deterministic=True)
FE = ["name_eq", "addr_eq", "t_noaddr", "q_noaddr", "name_jac", "name_c3", "name_contain", "addr_jac", "num_eq_first", "num_overlap", "num_conflict",
      "t_s3", "name_len_ratio", "addr_c3"]


def c3(a, b):
    A = {a[i:i + 3] for i in range(max(len(a) - 2, 1))}; B = {b[i:i + 3] for i in range(max(len(b) - 2, 1))}
    return len(A & B) / max(len(A | B), 1)


def feats(pairs, txt):
    rows = []
    for q, t in pairs:
        (qn, qa), (tn, ta) = txt[q], txt[t]
        cqn, ctn = cname(qn), cname(tn); cqa, cta = caddr(qa, REG), caddr(ta, REG)
        sq, st = set(cqn.split()), set(ctn.split()); aq, at = set(cqa.split()), set(cta.split())
        nq = [x for x in cqa.split() if x.isdigit()]; nt = [x for x in cta.split() if x.isdigit()]
        fq = re.findall(r"\d+", base(qa)); ft = re.findall(r"\d+", base(ta))
        rows.append((cqn == ctn, cqa == cta and cta != "", cta == "", cqa == "", len(sq & st) / max(len(sq | st), 1), c3(cqn, ctn),
                     (sq <= st or st <= sq) and bool(sq) and bool(st), len(aq & at) / max(len(aq | at), 1),
                     bool(fq) and bool(ft) and fq[0].lstrip("0") == ft[0].lstrip("0"), len(set(nq) & set(nt)), len(set(nq) ^ set(nt)) if nq and nt else -1,
                     t.startswith("S3"), len(ctn) / max(len(cqn), 1), c3(cqa, cta)))
    return pl.DataFrame(rows, schema=FE, orient="row").cast(pl.Float32)


def main():
    src = sys.argv[1]; tgt = "India" if src == "US" else "US"; LO = float(__import__("os").environ.get("LO", 0.2))
    T = ROOT / "student_resource/dataset/train"; txt = {}
    for i in (1, 2, 3):
        d = pl.read_csv(T / f"train_source{i}.tsv", separator="\t", quote_char=None, infer_schema_length=0).fill_null("")
        txt.update(dict(zip(d["entity_id"].to_list(), zip(d["business_name"].to_list(), d["business_address"].to_list()))))
    gt = pl.read_csv(T / "train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema_length=0)
    truth = {q: set(r.split(",")) if r else set() for q, r in gt.iter_rows()}
    # resolver training: source-country labeled pairs from the fold tables (in-domain probabilities not used)
    tr = pl.concat([pl.read_parquet(f"/Users/earther/Desktop/aml-shared/candidates_mfu_f{h}.parquet", columns=["q", "t", "label", "country", "prob"]) for h in (1, 2, 3)])
    tr = tr.filter((pl.col("country") == src) & (pl.col("prob") >= 0.005))
    Xtr = feats(tr.select("q", "t").iter_rows(), txt); m = lgb.LGBMClassifier(**P).fit(Xtr.to_numpy(), tr["label"].to_numpy())
    print(json.dumps({"event": "resolver_train", "src": src, "pairs": tr.height, "pos_rate": round(float(tr["label"].mean()), 4)}), flush=True)
    d = pl.read_parquet(ROOT / f"outputs/experiments/CL-034/pred_FULL_{src}_to_{tgt}.parquet"); thr = float(d["thr"][0])
    qa, ta, pa = d["q"].to_numpy(), d["t"].to_numpy(), d["prob"].to_numpy(); tq = sorted(set(qa.tolist()))
    dec = decide(qa, ta, pa, thr); owned = {t for v in dec.values() for t in v}
    base_m = float(np.mean([f05(dec.get(q, set()), truth[q]) for q in tq]))
    band = d.filter((pl.col("prob") >= LO) & (pl.col("prob") < thr) & ~pl.col("t").is_in(list(owned)))
    Xb = feats(band.select("q", "t").iter_rows(), txt); rp = m.predict_proba(Xb.to_numpy())[:, 1]
    yb = np.array([int(t in truth[q]) for q, t in band.select("q", "t").iter_rows()])
    print(json.dumps({"event": "band", "pairs": band.height, "per_s1": round(band.height / len(tq), 3), "band_precision": round(float(yb.mean()), 4), "base_macro": round(base_m, 6), "thr": thr}), flush=True)
    bq, bt = band["q"].to_list(), band["t"].to_list()
    for tau in (0.5, 0.7, 0.8, 0.9, 0.95, 0.98):
        add = {}
        for i in np.argsort(-rp, kind="stable"):
            if rp[i] < tau: break
            if bt[i] not in add: add[bt[i]] = bq[i]
        full = {q: set(v) for q, v in dec.items()}
        for t_, q_ in add.items(): full.setdefault(q_, set()).add(t_)
        mm = float(np.mean([f05(full.get(q, set()), truth[q]) for q in tq])); ntp = sum(int(t_ in truth[q_]) for t_, q_ in add.items())
        print(json.dumps({"src": src, "tgt": tgt, "tau": tau, "adds": len(add), "adds_true": ntp, "precision": round(ntp / max(len(add), 1), 4), "macro": round(mm, 6), "delta": round(mm - base_m, 6)}), flush=True)


if __name__ == "__main__":
    main()
