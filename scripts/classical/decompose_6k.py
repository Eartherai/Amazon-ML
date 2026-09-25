"""Exact per-S1 loss decomposition on the fixed 6k for the CL-003 stage-2 predictions."""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pilot_stage2_exp044 as P
ROOT = Path(__file__).resolve().parents[2]
truth = {k: set(v) for k, v in json.loads((P.EXP044 / "truth.json").read_text()).items()}
pairs = [json.loads(l) for l in (P.EXP044 / "pairs.jsonl").open()]
top12 = defaultdict(set)
for x in pairs: top12[x["q"]].add(x["t"])
z = np.load(ROOT / "outputs/experiments/CL-003/eval6k-sidecar.npz")
thr = json.loads((ROOT / "outputs/experiments/CL-003/stage2-200k-sidecar.json").read_text())["threshold"]
side, pred = defaultdict(set), defaultdict(set)
for q, t, p in zip(z["q"].tolist(), z["t"].tolist(), z["p"]):
    side[q].add(t)
    if p >= thr: pred[q].add(t)
f = P.f05
tot = defaultdict(float); n = len(truth)
for s, T in truth.items():
    Pp = pred[s]; fp_free = Pp & T
    fcur = f(Pp, T)
    if not T:
        tot["singleton_false_merge"] += 1 - fcur; continue
    f_nofp = f(fp_free, T)
    f_side = f(fp_free | (T & side[s]), T)
    f_top = f(fp_free | (T & top12[s]), T)
    tot["false_positive_links"] += f_nofp - fcur
    tot["rejected_true_in_sidecar"] += f_side - f_nofp
    tot["true_outside_sidecar_in_top12"] += f_top - f_side
    tot["true_outside_top12_or_unretrieved"] += 1 - f_top
    k = "one" if len(T) == 1 else "multi"
    tot[f"loss_true_{k}"] += 1 - fcur
res = {k: round(v / n, 5) for k, v in tot.items()}
res["macro_f0_5"] = round(np.mean([f(pred[s], T) for s, T in truth.items()]), 6)
res["oracle_sidecar"] = round(np.mean([f(T & side[s], T) for s, T in truth.items()]), 6)
res["oracle_top12"] = round(np.mean([f(T & top12[s], T) for s, T in truth.items()]), 6)
print(json.dumps(res, indent=1))
