"""CL-012: fresh per-country char-3 TF-IDF retrieval (fit on that country's test targets) to rescue links the
frozen train-fit retrieval missed (probe: France 0.2-0.28 strong new pairs/S1 vs India 0.01, US 0.005, train 0.03).
New pairs (outside existing top-12) are scored by the text-only stage-2 (no first-stage features)."""
import sys, json, time, multiprocessing as mp
from pathlib import Path
import numpy as np, polars as pl, duckdb, lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
sys.path.insert(0, "scripts/classical"); import stage2_features_v2 as F
O = "delim='\t',header=true,quote='',escape='',all_varchar=true"
country, K = sys.argv[1], int(sys.argv[2])
out = Path(f"outputs/experiments/CL-012/{country}-rescue.parquet"); out.parent.mkdir(parents=True, exist_ok=True)
t0 = time.time(); con = duckdb.connect(); D = "student_resource/dataset/test"
tg = con.execute(f"""SELECT entity_id,business_name,business_address FROM read_csv('{D}/test_source2.tsv',{O}) WHERE country='{country}'
  UNION ALL SELECT entity_id,business_name,business_address FROM read_csv('{D}/test_source3.tsv',{O}) WHERE country='{country}'""").fetchall()
s1 = con.execute(f"SELECT entity_id,business_name,business_address FROM read_csv('{D}/test_source1.tsv',{O}) WHERE country='{country}'").fetchall()
tid = np.array([r[0] for r in tg]); qid = np.array([r[0] for r in s1])
pairs = set()
for col in (1, 2):
    vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), min_df=2, dtype=np.float32)
    M = vec.fit_transform([F.norm(r[col] or "") for r in tg]).T.tocsr()
    Q = vec.transform([F.norm(r[col] or "") for r in s1]).tocsr()
    for lo in range(0, Q.shape[0], 20000):
        C = sp_matmul_topn(Q[lo:lo + 20000], M, top_n=K, threshold=0.3, n_threads=8).tocoo()
        pairs.update(zip(qid[lo + C.row].tolist(), tid[C.col].tolist()))
    print(json.dumps({"route": col, "pairs": len(pairs), "s": round(time.time() - t0, 1)}), flush=True)
top = pl.read_parquet("outputs/experiments/CL-003/test_probs/CL-005-top12v2-v1.parquet").select("q", "t")
new = pl.DataFrame(list(pairs), schema=["q", "t"], orient="row").join(top, on=["q", "t"], how="anti")
print(json.dumps({"new_pairs": new.height, "s": round(time.time() - t0, 1)}), flush=True)
S1 = {r[0]: (r[1] or "", r[2] or "") for r in s1}; TG = {r[0]: (r[1] or "", r[2] or "") for r in tg}
idf = json.load(open("outputs/experiments/CL-002/test_s1_idf.json")); IDF = (idf["name"], idf["name_default"], idf["addr"], idf["addr_default"])
items = new.rows()
def work(chunk):
    return np.asarray([F.pair_vector(F.Record(*S1[q]), F.Record(*TG[t]), t.startswith("S2-"), *IDF) for q, t in chunk], dtype=np.float32)
chunks = [items[i:i + 20000] for i in range(0, len(items), 20000)]
with mp.get_context("fork").Pool(8) as pool:
    X = np.vstack(pool.map(work, chunks))
p = lgb.Booster(model_file="outputs/experiments/CL-003/stage2-200k-top12-textonly.txt").predict(X, num_threads=8)
new.with_columns(pl.Series("p", p)).write_parquet(out)
print(json.dumps({"country": country, "new_pairs": len(items), "ge_0.7": int((p >= 0.7).sum()), "ge_0.85": int((p >= 0.85).sum()), "ge_0.9": int((p >= 0.9).sum()), "s1": len(s1), "s": round(time.time() - t0, 1)}))
