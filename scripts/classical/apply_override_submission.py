"""CL-057: apply a pair-decision override (q, t, new_decision) to a base compact submission, keeping its candidate set.
Base decisions come from the base run's candidate table (--cands, from assemble_compact --dump-cands, all routes incl. dup).
Ownership: an added target already owned by another S1 is not added (base owners keep theirs). Every matched pair must be
a candidate pair (asserted). Deterministic output order = test_source1 order, sorted ids."""
import argparse, hashlib, json
from collections import defaultdict
from pathlib import Path
import polars as pl
ROOT = Path(__file__).resolve().parents[2]; TEST = ROOT / "student_resource/dataset/test"


def sha(p):
    h = hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cands", required=True); ap.add_argument("--base-dir", required=True)
    ap.add_argument("--override", required=True); ap.add_argument("--name", required=True); a = ap.parse_args()
    c = pl.read_parquet(a.cands).select("q", "t", "decision")
    o = pl.read_parquet(a.override).select("q", "t", pl.col("new_decision").cast(pl.Int64))
    dec = {(q, t): d for q, t, d in c.iter_rows()}; base = dict(dec)
    owner = {t: q for (q, t), d in dec.items() if d}
    stats = defaultdict(int)
    for q, t, d in o.sort(["q", "t"]).iter_rows():
        if (q, t) not in dec or dec[(q, t)] == d: continue
        if d == 1 and t in owner and owner[t] != q: stats["add_blocked_ownership"] += 1; continue
        dec[(q, t)] = d; stats["add" if d else "remove"] += 1
        if d: owner[t] = q
        elif owner.get(t) == q: owner.pop(t)
    s1 = pl.read_csv(TEST / "test_source1.tsv", separator="\t", quote_char=None, infer_schema_length=0)
    order, country = s1["entity_id"].to_list(), dict(zip(s1["entity_id"], s1["country"]))
    pred = defaultdict(set)
    for (q, t), d in dec.items():
        if d: pred[q].add(t)
    cand = defaultdict(set)
    with open(Path(a.base_dir) / "candidate_pairs.tsv") as f:
        f.readline()
        for line in f:
            q, _, raw = line.rstrip("\n").partition("\t"); cand[q] = set(x for x in raw.split(",") if x)
    dest = ROOT / "outputs/submissions" / a.name; dest.mkdir(parents=True, exist_ok=False)
    per = defaultdict(lambda: {"s1": 0, "links": 0, "empty": 0})
    with open(dest / "matching_results.tsv", "w") as fm:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        for q in order:
            m = pred.get(q, set()); assert m <= cand[q], q
            fm.write(q + "\t" + ",".join(sorted(m)) + "\n"); k = country[q]; per[k]["s1"] += 1; per[k]["links"] += len(m); per[k]["empty"] += int(not m)
    (dest / "candidate_pairs.tsv").write_bytes((Path(a.base_dir) / "candidate_pairs.tsv").read_bytes())
    prov = {"matching_sha256": sha(dest / "matching_results.tsv"), "candidate_sha256": sha(dest / "candidate_pairs.tsv"), "args": vars(a), "applied": dict(stats), "per_country": dict(per), "fold4": "CLOSED"}
    (dest / "provenance.json").write_text(json.dumps(prov, indent=1)); print(json.dumps(prov, indent=1))


if __name__ == "__main__":
    main()
