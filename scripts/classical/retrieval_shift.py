"""CL-032: retrieval stress tests 3/4 (Track E). India fold-3 queries against an India target universe, char3 TF-IDF
(name + address) with the vocabulary/IDF fit on:
  FOREIGN  US train targets only (the France situation: statistics from other countries)
  ADAPTED  the India target corpus itself, unlabeled (allowed test-side unsupervised statistics)
  BOTH     US + India targets
plus an accent-folded / NFKC view of ADAPTED. Reports link recall, complete-entity recall and oracle F0.5 at K.
Universe: all true targets of the sampled queries plus a random fill of India S2/S3 (fixed seed). Fold4 CLOSED.
"""
import json, random, unicodedata
from pathlib import Path
import numpy as np, polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "student_resource/dataset/train"
R = lambda f: pl.read_csv(D / f, separator="\t", quote_char=None, infer_schema_length=0).fill_null("")
N_Q, N_FILL, KS = 3000, 1_000_000, (1, 3, 5, 8, 10, 20, 50, 100)


def fold_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", unicodedata.normalize("NFKC", s)) if not unicodedata.combining(c))


def main():
    s1 = R("train_source1.tsv"); tg = pl.concat([R("train_source2.tsv").select("entity_id", "business_name", "business_address", "country"),
                                               R("train_source3.tsv").select("entity_id", "business_name", "business_address", "country")])
    tg = tg.with_columns((pl.col("business_name") + " " + pl.col("business_address")).alias("x"))
    top = pl.read_parquet(ROOT / "outputs/experiments/CL-003/s3/results/top12-fold3.parquet")
    ind = set(s1.filter(pl.col("country") == "India")["entity_id"].to_list())
    qpool = sorted(set(top["source1_entity_id"].to_list()) & ind)
    qs = random.Random(0).sample(qpool, N_Q)
    gt = R("train_ground_truth.tsv"); truth = {q: set(r.split(",")) - {""} for q, r in gt.filter(pl.col("source1_entity_id").is_in(qs)).iter_rows()}
    need = set().union(*truth.values())
    ind_t = tg.filter(pl.col("country") == "India"); us_t = tg.filter(pl.col("country") == "US")
    rest = ind_t.filter(~pl.col("entity_id").is_in(list(need)))["entity_id"].to_list()
    uni = sorted(need | set(random.Random(1).sample(rest, N_FILL)))
    U = ind_t.filter(pl.col("entity_id").is_in(uni)).sort("entity_id"); ids = U["entity_id"].to_list(); tix = U["x"].to_list()
    Q = s1.filter(pl.col("entity_id").is_in(qs)).with_columns((pl.col("business_name") + " " + pl.col("business_address")).alias("x")).sort("entity_id")
    qids, qx = Q["entity_id"].to_list(), Q["x"].to_list()
    us_fit = us_t.sample(N_FILL, seed=2)["x"].to_list(); ind_fit = ind_t.sample(N_FILL, seed=3)["x"].to_list()
    fits = {"FOREIGN_US": (us_fit, None), "ADAPTED_INDIA": (ind_fit, None), "BOTH": (us_fit[:500000] + ind_fit[:500000], None), "ADAPTED_FOLDED": (ind_fit, fold_accents)}
    rows = []
    for name, (fit_docs, pre) in fits.items():
        v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), lowercase=True, sublinear_tf=True, min_df=2, max_features=200000, dtype=np.float32,
                            preprocessor=(lambda s, f=pre: f(s).lower()) if pre else None)
        v.fit(fit_docs); T = v.transform(tix).T.tocsr(); Qm = v.transform(qx)
        hits = {k: [] for k in KS}
        for i in range(0, len(qids), 50):
            S = (Qm[i:i + 50] @ T).toarray()
            top_idx = np.argpartition(-S, 100, axis=1)[:, :100]
            for r, q in enumerate(qids[i:i + 50]):
                o = top_idx[r][np.argsort(-S[r, top_idx[r]])]
                ranked = [ids[j] for j in o]
                for k in KS: hits[k].append((q, set(ranked[:k]) & truth[q]))
        for k in KS:
            nt = sum(len(truth[q]) for q, _ in hits[k]); nh = sum(len(h) for _, h in hits[k])
            pos = [(q, h) for q, h in hits[k] if truth[q]]
            orc = [1.0 if not truth[q] else (0.0 if not h else 1.25 * (len(h) / len(truth[q])) / (0.25 + len(h) / len(truth[q]))) for q, h in hits[k]]
            row = {"fit": name, "K": k, "link_recall": round(nh / nt, 5), "complete_entity_recall": round(float(np.mean([len(h) == len(truth[q]) for q, h in pos])), 5),
                   "oracle_f05": round(float(np.mean(orc)), 5)}
            rows.append(row); print(json.dumps(row), flush=True)
    out = ROOT / "outputs/experiments/CL-032"; out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(out / "retrieval_shift.csv")
    print(json.dumps({"queries": len(qids), "universe": len(ids), "true_links": sum(len(v) for v in truth.values())}))


if __name__ == "__main__":
    main()
