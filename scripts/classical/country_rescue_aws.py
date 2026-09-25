"""CL-012 AWS worker: fresh test-fitted retrieval route for one country, scored by text-only stage-2.

--route name|address|combo ; --analyzer char3|char35|word ; K per query; max_df caps ubiquitous n-grams.
Excludes pairs already in the country's existing top-12 set. Unsupervised test statistics only."""
import argparse, json, os, sys, time, multiprocessing as mp
from pathlib import Path
import numpy as np, polars as pl, duckdb, lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
sys.path.insert(0, str(Path(__file__).resolve().parent)); import stage2_features_v2 as F
O = "delim='\t',header=true,quote='',escape='',all_varchar=true"
ap = argparse.ArgumentParser()
for a in ("--country", "--route", "--analyzer", "--test-dir", "--top12", "--idf", "--model", "--out"): ap.add_argument(a, required=True)
ap.add_argument("--k", type=int, default=50); ap.add_argument("--max-df", type=float, default=0.02); ap.add_argument("--min-sim", type=float, default=0.25)
a = ap.parse_args(); t0 = time.time(); nthreads = os.cpu_count()
con = duckdb.connect(); D = a.test_dir
tg = con.execute(f"""SELECT entity_id,business_name,business_address FROM read_csv('{D}/test_source2.tsv',{O}) WHERE country='{a.country}'
  UNION ALL SELECT entity_id,business_name,business_address FROM read_csv('{D}/test_source3.tsv',{O}) WHERE country='{a.country}'""").fetchall()
s1 = con.execute(f"SELECT entity_id,business_name,business_address FROM read_csv('{D}/test_source1.tsv',{O}) WHERE country='{a.country}'").fetchall()
tid = np.array([r[0] for r in tg]); qid = np.array([r[0] for r in s1])
def text(r):
    n, ad = F.norm(r[1] or ""), F.norm(r[2] or "")
    return n if a.route == "name" else ad if a.route == "address" else (n + " | " + ad)
with mp.get_context("fork").Pool(nthreads) as pool:
    tt = pool.map(text, tg, chunksize=20000); qt = pool.map(text, s1, chunksize=5000)
print(json.dumps({"norm_s": round(time.time() - t0, 1), "targets": len(tt), "s1": len(qt)}), flush=True)
kw = dict(analyzer="char", ngram_range=(3, 3)) if a.analyzer == "char3" else dict(analyzer="char_wb", ngram_range=(3, 5)) if a.analyzer == "char35" else dict(analyzer="word", token_pattern=r"(?u)\b\w+\b")
vec = TfidfVectorizer(min_df=2, max_df=a.max_df, sublinear_tf=True, dtype=np.float32, **kw)
M = vec.fit_transform(tt).T.tocsr(); Q = vec.transform(qt).tocsr()
print(json.dumps({"vec_s": round(time.time() - t0, 1), "vocab": len(vec.vocabulary_)}), flush=True)
rows, cols, sims = [], [], []
for lo in range(0, Q.shape[0], 10000):
    C = sp_matmul_topn(Q[lo:lo + 10000], M, top_n=a.k, threshold=a.min_sim, n_threads=nthreads).tocoo()
    rows.append(lo + C.row); cols.append(C.col); sims.append(C.data)
    print(json.dumps({"done": lo + 10000, "s": round(time.time() - t0, 1)}), flush=True)
cand = pl.DataFrame({"q": qid[np.concatenate(rows)], "t": tid[np.concatenate(cols)], "sim": np.concatenate(sims)})
top = pl.read_parquet(a.top12).select("q", "t")
new = cand.join(top, on=["q", "t"], how="anti")
S1 = {r[0]: (r[1] or "", r[2] or "") for r in s1}; TG = {r[0]: (r[1] or "", r[2] or "") for r in tg}
idf = json.load(open(a.idf)); IDF = (idf["name"], idf["name_default"], idf["addr"], idf["addr_default"])
items = new.select("q", "t").rows()
def work(chunk):
    return np.asarray([F.pair_vector(F.Record(*S1[q]), F.Record(*TG[t]), t.startswith("S2-"), *IDF) for q, t in chunk], dtype=np.float32).reshape(-1, len(F.NAMES) - 9)
chunks = [items[i:i + 5000] for i in range(0, len(items), 5000)]
with mp.get_context("fork").Pool(nthreads) as pool:
    X = np.vstack(pool.map(work, chunks)) if chunks else np.zeros((0, len(F.NAMES) - 9), np.float32)
p = lgb.Booster(model_file=a.model).predict(X, num_threads=nthreads) if len(X) else np.array([])
new = new.with_columns(pl.Series("p", p), pl.lit(f"{a.route}-{a.analyzer}-k{a.k}").alias("routecfg"))
Path(a.out).parent.mkdir(parents=True, exist_ok=True); new.write_parquet(a.out)
res = {"country": a.country, "route": a.route, "analyzer": a.analyzer, "k": a.k, "cand": cand.height, "new": new.height,
       "ge_0.7": int((p >= 0.7).sum()), "ge_0.85": int((p >= 0.85).sum()), "ge_0.9": int((p >= 0.9).sum()), "s1": len(s1), "s": round(time.time() - t0, 1)}
Path(a.out).with_suffix(".json").write_text(json.dumps(res)); print(json.dumps(res), flush=True)
