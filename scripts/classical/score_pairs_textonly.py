"""Score arbitrary (q, t) pairs with the text-only stage-2 model (v2 pair features; no first-stage inputs).

--split train|test selects texts and the matching Source 1 IDF. Input parquet needs columns q, t;
extra columns are carried through. Output adds `p_text`. Workers build records from raw text tuples.
"""
import argparse, json, multiprocessing as mp, os, sys, time
from pathlib import Path
import duckdb, numpy as np, polars as pl, lightgbm as lgb
sys.path.insert(0, str(Path(__file__).resolve().parent)); import stage2_features_v2 as F
ROOT = Path(__file__).resolve().parents[2]
O = "delim='\t',header=true,quote='',escape='',all_varchar=true"
G = {}


def work(chunk):
    S1, TG, IDF = G["s1"], G["tg"], G["idf"]
    return np.asarray([F.pair_vector(F.Record(*S1[q]), F.Record(*TG[t]), t.startswith("S2-"), *IDF) for q, t in chunk], dtype=np.float32).reshape(-1, len(F.NAMES) - 9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--inp", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=7)
    ap.add_argument("--no-x", action="store_true", help="do not save the feature matrix")
    a = ap.parse_args(); t0 = time.time()
    df = pl.read_parquet(a.inp)
    D = ROOT / f"student_resource/dataset/{a.split}"
    con = duckdb.connect()
    con.execute("CREATE TABLE w(id VARCHAR)"); con.executemany("INSERT INTO w VALUES (?)", [(x,) for x in set(df["q"].to_list()) | set(df["t"].to_list())])
    recs = {}
    for src in ("source1", "source2", "source3"):
        for i, n, ad in con.execute(f"SELECT entity_id,business_name,business_address FROM read_csv('{D}/{a.split}_{src}.tsv',{O}) JOIN w ON id=entity_id").fetchall():
            recs[i] = (n or "", ad or "")
    idfp = ROOT / ("outputs/experiments/CL-001/train_s1_idf.json" if a.split == "train" else "outputs/experiments/CL-002/test_s1_idf.json")
    idf = json.loads(idfp.read_text())
    G.update({"s1": recs, "tg": recs, "idf": (idf["name"], idf["name_default"], idf["addr"], idf["addr_default"])})
    items = df.select("q", "t").rows()
    chunks = [items[i:i + 5000] for i in range(0, len(items), 5000)]
    with mp.get_context("fork").Pool(a.procs, maxtasksperchild=20) as pool:
        X = np.vstack(pool.map(work, chunks)) if chunks else np.zeros((0, len(F.NAMES) - 9), np.float32)
    p = lgb.Booster(model_file=str(ROOT / "outputs/experiments/CL-003/stage2-200k-top12-textonly.txt")).predict(X, num_threads=8)
    df.with_columns(pl.Series("p_text", p)).write_parquet(a.out)
    if not a.no_x:
        np.save(a.out.replace(".parquet", "-X.npy"), X)
    print(json.dumps({"pairs": len(items), "s": round(time.time() - t0, 1), "ge_0.5": int((p >= 0.5).sum())}))


if __name__ == "__main__":
    main()
