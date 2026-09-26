"""CL-069: CE-scored sparse below-band rescue. Pairs in the stage-1 top-12 with base < 0.02 (never CE-scored in
production) get fold-held e5-base CE logits (P5-PAIRSCORE-CE-F{h}SPLOW-001, CE trained without fold h). Add-only model
(LightGBM on CE logit + stack prob + stage-1 base + rank + S1/target context) per MF protocol via rx_rescue.py's
build/realize: fit on two folds, threshold by cross-fit inside them, apply once to the held fold. Fold4 CLOSED."""
import json, sys
from pathlib import Path
import numpy as np, polars as pl, lightgbm as lgb
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "scripts/classical"))
from rx_rescue import build, realize, f05, PS, SH
FE = ["ce", "score", "base", "srank", "q_nacc", "q_maxacc", "q_ncand", "t_ncomp"]
ONLYCE = len(sys.argv) > 1 and sys.argv[1] == "ceonly"


def main():
    tr_ = pl.read_parquet(SH / "truth_mf.parquet"); truth = {q: set(s.split(",")) - {""} for q, s in tr_.select("q", "targets").iter_rows()}
    qfold = dict(tr_.select("q", "fold").iter_rows()); data = {}
    for h in (1, 2, 3):
        x, pred = build(h, truth)
        ce = pl.read_parquet(ROOT / f"outputs/experiments/CL-069/ce-f{h}.parquet").select("q", "t", pl.col("logit").alias("ce_s"))
        x = x.filter(pl.col("b_sparse") == 1).drop("ce").join(ce, on=["q", "t"], how="inner").rename({"ce_s": "ce"})
        data[h] = (x, pred); print(json.dumps({"fold": h, "pairs": x.height, "pos": int(x["y"].sum())}), flush=True)
    fe = ["ce"] if ONLYCE else FE
    grid = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]; rows = []
    for h in (1, 2, 3):
        others = [k for k in (1, 2, 3) if k != h]
        tr = pl.concat([data[k][0] for k in others]); m = lgb.LGBMClassifier(**PS).fit(tr.select(fe).to_numpy().astype(np.float32), tr["y"].to_numpy())
        score = {g: 0.0 for g in grid}
        for k in others:
            kk = [z for z in others if z != k][0]; mk = lgb.LGBMClassifier(**PS).fit(data[kk][0].select(fe).to_numpy().astype(np.float32), data[kk][0]["y"].to_numpy())
            xk = data[k][0]; pk = mk.predict_proba(xk.select(fe).to_numpy().astype(np.float32))[:, 1]; qk = [q for q in truth if qfold[q] == k]
            for g in grid: score[g] += realize(xk, pk, g, data[k][1], qk, truth)[0]
        thr = max(score, key=score.get)
        x, pred0 = data[h]; ph = m.predict_proba(x.select(fe).to_numpy().astype(np.float32))[:, 1]; qh = [q for q in truth if qfold[q] == h]
        b = float(np.mean([f05(pred0.get(q, set()), truth[q]) for q in qh])); n, nadd, ntp = realize(x, ph, thr, pred0, qh, truth)
        rows.append({"fold": h, "thr": thr, "base": round(b, 6), "new": round(n, 6), "delta": round(n - b, 6), "adds": nadd, "adds_true": ntp, "precision": round(ntp / max(nadd, 1), 4), "n_s1": len(qh)})
        print(json.dumps(rows[-1]), flush=True)
        # write adds for combination
        add = {}
        for i in np.argsort(-ph, kind="stable"):
            if ph[i] < thr: break
            q, t = x["q"][int(i)], x["t"][int(i)]
            if t not in add: add[t] = q
        out = ROOT / f"outputs/experiments/CL-069/adds{'_ceonly' if ONLYCE else ''}"; out.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"q": list(add.values()), "t": list(add.keys())}, schema={"q": pl.String, "t": pl.String}).write_parquet(out / f"fold{h}.parquet")
    tot = sum(r["n_s1"] for r in rows); print(json.dumps({"pooled_delta": round(sum(r["delta"] * r["n_s1"] for r in rows) / tot, 6)}))


if __name__ == "__main__":
    main()
