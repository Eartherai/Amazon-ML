"""CL-011: independent char-3 TF-IDF retrieval probe (name top-50 + address top-50, fit per split/country on targets).
Counts strong text-only-stage-2 pairs OUTSIDE the existing top-12 per S1, on test samples and on the labeled 6k."""
import sys, json, random, math
from collections import defaultdict
from pathlib import Path
import numpy as np, polars as pl, duckdb, lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
sys.path.insert(0, "scripts/classical"); import stage2_features_v2 as F
O = "delim='\t',header=true,quote='',escape='',all_varchar=true"
split, country, nq = sys.argv[1], sys.argv[2], int(sys.argv[3])
con = duckdb.connect(); D = f"student_resource/dataset/{split}"
tg = con.execute(f"""SELECT entity_id,business_name,business_address FROM read_csv('{D}/{split}_source2.tsv',{O}) WHERE country='{country}'
  UNION ALL SELECT entity_id,business_name,business_address FROM read_csv('{D}/{split}_source3.tsv',{O}) WHERE country='{country}'""").fetchall()
s1 = dict((r[0], (r[1] or "", r[2] or "")) for r in con.execute(f"SELECT entity_id,business_name,business_address FROM read_csv('{D}/{split}_source1.tsv',{O}) WHERE country='{country}'").fetchall())
if split == "train":
    ev = json.load(open("/Users/earther/.codex/worktrees/aml-neural-warroom/Amazon ML Challange/outputs/experiments/warroom_neural/EXP-044/truth.json"))
    pairs = [json.loads(l) for l in open("/Users/earther/.codex/worktrees/aml-neural-warroom/Amazon ML Challange/outputs/experiments/warroom_neural/EXP-044/pairs.jsonl")]
    qs = sorted({p["q"] for p in pairs if p["country"] == country})[:nq]
    top = {(p["q"], p["t"]) for p in pairs}
    idf = json.load(open("outputs/experiments/CL-001/train_s1_idf.json"))
else:
    random.seed(0); qs = random.sample(sorted(s1), nq)
    pr = pl.read_parquet("outputs/experiments/CL-003/test_probs/CL-005-top12v2-v1.parquet").filter(pl.col("q").is_in(qs))
    top = set(pr.select("q", "t").iter_rows())
    idf = json.load(open("outputs/experiments/CL-002/test_s1_idf.json"))
ids = [r[0] for r in tg]; recs_t = {r[0]: (r[1] or "", r[2] or "") for r in tg}
cand = defaultdict(set)
for col in (0, 1):
    vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), min_df=2, dtype=np.float32)
    M = vec.fit_transform([F.norm(r[col + 1] or "") for r in tg])
    Q = vec.transform([F.norm(s1[q][col]) for q in qs])
    for lo in range(0, len(qs), 200):
        S = (Q[lo:lo + 200] @ M.T).tocsr()
        for i in range(S.shape[0]):
            row = S.getrow(i); 
            if row.nnz == 0: continue
            k = min(50, row.nnz); idx = row.indices[np.argpartition(-row.data, k - 1)[:k]]
            for j in idx: cand[qs[lo + i]].add(ids[j])
new = [(q, t) for q in qs for t in cand[q] if (q, t) not in top]
model = lgb.Booster(model_file="outputs/experiments/CL-003/stage2-200k-top12-textonly.txt")
IDF = (idf["name"], idf["name_default"], idf["addr"], idf["addr_default"])
rows = [F.pair_vector(F.Record(*s1[q]), F.Record(*recs_t[t]), t.startswith("S2-"), *IDF) for q, t in new]
p = model.predict(np.asarray(rows, dtype=np.float32)) if rows else np.array([])
res = {"split": split, "country": country, "queries": len(qs), "new_pairs": len(new)}
for th in (0.5, 0.7, 0.8, 0.9):
    sel = [(q, t) for (q, t), pp in zip(new, p) if pp >= th]
    res[f"new_ge_{th}_per_s1"] = round(len(sel) / len(qs), 4)
    if split == "train":
        res[f"precision_ge_{th}"] = round(sum(t in set(ev[q]) for q, t in sel) / max(len(sel), 1), 4)
if split == "train":
    miss = sum(1 for q in qs for t in ev[q] if (q, t) not in top)
    got = sum(1 for q in qs for t in ev[q] if (q, t) not in top and t in cand[q])
    res["true_outside_top12_per_s1"] = round(miss / len(qs), 4); res["probe_recovers_per_s1"] = round(got / len(qs), 4)
print(json.dumps(res))
if len(sys.argv) > 4:
    ex = [(q, t, pp) for (q, t), pp in zip(new, p) if pp >= 0.8][:int(sys.argv[4])]
    for q, t, pp in ex: print(round(float(pp), 3), "|", s1[q], "\n      ", recs_t[t])
