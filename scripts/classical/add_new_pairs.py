"""CL-069: add NEW (q, t) matches that are outside the base candidate set (e.g. CE-scored sparse below-band rescue).
Each added pair is appended to the S1's candidate list and to its matches, unless the target is already matched to any S1
in the base (ownership guard) or claimed earlier in this file's order. Output order = test_source1 order, sorted ids.
Usage: add_new_pairs.py BASE_DIR ADDS_PARQUET NAME"""
import hashlib, json, sys
from collections import defaultdict
from pathlib import Path
import polars as pl

ROOT = Path(__file__).resolve().parents[2]; TEST = ROOT / "student_resource/dataset/test"


def sha(p):
    h = hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()


def read(p):
    d = {}
    with open(p) as f:
        f.readline()
        for line in f:
            q, _, raw = line.rstrip("\n").partition("\t"); d[q] = set(x for x in raw.split(",") if x)
    return d


def main():
    base, adds, name = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    match, cand = read(base / "matching_results.tsv"), read(base / "candidate_pairs.tsv")
    owned = {t for v in match.values() for t in v}
    a = pl.read_parquet(adds).select("q", "t")
    n_add = n_block = n_newcand = 0
    for q, t in a.iter_rows():
        if t in owned: n_block += 1; continue
        if q not in match: raise SystemExit(f"unknown S1 {q}")
        owned.add(t); match[q].add(t)
        if t not in cand[q]: cand[q].add(t); n_newcand += 1
        n_add += 1
    s1 = pl.read_csv(TEST / "test_source1.tsv", separator="\t", quote_char=None, infer_schema_length=0)
    order, country = s1["entity_id"].to_list(), dict(zip(s1["entity_id"], s1["country"]))
    dest = ROOT / "outputs/submissions" / name; dest.mkdir(parents=True, exist_ok=False)
    per = defaultdict(lambda: {"s1": 0, "links": 0, "empty": 0, "cand": 0})
    with open(dest / "matching_results.tsv", "w") as fm, open(dest / "candidate_pairs.tsv", "w") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n"); fc.write("source1_entity_id\tcandidate_entity_ids\n") if False else None
        hdr = open(base / "candidate_pairs.tsv").readline(); fc.write(hdr)
        for q in order:
            m, c = match.get(q, set()), cand.get(q, set()); assert m <= c, q
            fm.write(q + "\t" + ",".join(sorted(m)) + "\n"); fc.write(q + "\t" + ",".join(sorted(c)) + "\n")
            k = country[q]; per[k]["s1"] += 1; per[k]["links"] += len(m); per[k]["empty"] += int(not m); per[k]["cand"] += len(c)
    prov = {"matching_sha256": sha(dest / "matching_results.tsv"), "candidate_sha256": sha(dest / "candidate_pairs.tsv"), "base": str(base), "adds": adds,
            "added": n_add, "blocked_ownership": n_block, "new_candidate_pairs": n_newcand, "per_country": dict(per), "fold4": "CLOSED"}
    (dest / "provenance.json").write_text(json.dumps(prov, indent=1)); print(json.dumps(prov, indent=1))


if __name__ == "__main__":
    main()
