"""Strict MF evaluation of stacked India/US correctors: base decisions candidates_mfu_f{h}; overrides applied in order
(adds only when the target is unowned or already owned by the same S1; removes release ownership); optional new-pair adds
(outside the tables). Exact macro F0.5 over all truth_mf S1 per fold. Usage: combo_eval.py NAME=DIR[:new] ..."""
import sys
from collections import defaultdict
import numpy as np, polars as pl
SH = "/Users/earther/Desktop/aml-shared"


def f05(P, T):
    if not P and not T: return 1.0
    tp = len(P & T)
    if not tp: return 0.0
    a, r = tp / len(P), tp / len(T); return 1.25 * a * r / (0.25 * a + r)


tr = pl.read_parquet(f"{SH}/truth_mf.parquet"); truth = {q: set(s.split(",")) - {""} for q, s in tr.select("q", "targets").iter_rows()}
qf = dict(tr.select("q", "fold").iter_rows()); co = {}
for h in (1, 2, 3):
    co.update(dict(pl.read_parquet(f"{SH}/candidates_mfu_f{h}.parquet").select("q", "country").unique("q").iter_rows()))
specs = [a.split("=", 1) for a in sys.argv[1:]]
tot = defaultdict(float); ntot = 0; per = defaultdict(lambda: defaultdict(float)); cnt = defaultdict(int)
for h in (1, 2, 3):
    c = pl.read_parquet(f"{SH}/candidates_mfu_f{h}.parquet").select("q", "t", "decision")
    pred, owner = defaultdict(set), {}
    for q, t, d in c.iter_rows():
        if d: pred[q].add(t); owner[t] = q
    qs = [q for q in truth if qf[q] == h]
    sc = lambda: np.array([f05(pred.get(q, set()), truth[q]) for q in qs])
    s = sc(); line = [f"fold{h} base {s.mean():.6f}"]; tot["base"] += s.sum(); prev = s
    for name, d in specs:
        new = d.endswith(":new"); d = d[:-4] if new else d
        o = pl.read_parquet(f"{d}/fold{h}.parquet")
        if "new_decision" not in o.columns: o = o.with_columns(pl.lit(1).alias("new_decision"))
        for q, t, v in o.select("q", "t", "new_decision").iter_rows():
            if v == 1 and owner.get(t, q) == q: pred[q].add(t); owner[t] = q
            elif v == 0 and t in pred.get(q, ()): pred[q].discard(t); owner.pop(t, None)
        s = sc(); line.append(f"+{name} {s.mean()-prev.mean():+.6f}"); tot[name] += s.sum(); prev = s
    for q, v in zip(qs, s): per[co.get(q, "?")]["f"] += v; cnt[co.get(q, "?")] += 1
    ntot += len(qs); print(" ".join(line), f"final {s.mean():.6f}", flush=True)
print("pooled base", round(tot["base"] / ntot, 6), " ".join(f"{k} {tot[k]/ntot:.6f}" for k, _ in specs), "| India", round(per["India"]["f"] / cnt["India"], 6), "US", round(per["US"]["f"] / cnt["US"], 6))
