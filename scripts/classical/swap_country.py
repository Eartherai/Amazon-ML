"""Replace one country's rows of a base submission with another submission's rows (candidate lists unioned so matches stay
inside candidates). Usage: swap_country.py BASE_DIR DONOR_MATCHING_TSV DONOR_CAND_TSV COUNTRY NAME"""
import hashlib, json, sys
from pathlib import Path
import polars as pl
ROOT = Path(__file__).resolve().parents[2]


def read(p):
    d = {}
    with open(p) as f:
        f.readline()
        for line in f:
            q, _, raw = line.rstrip("\n").partition("\t"); d[q] = set(x for x in raw.split(",") if x)
    return d


base, dm, dc, country, name = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
bm, bc = read(base / "matching_results.tsv"), read(base / "candidate_pairs.tsv"); om, oc = read(dm), read(dc)
s1 = pl.read_csv(ROOT / "student_resource/dataset/test/test_source1.tsv", separator="\t", quote_char=None, infer_schema_length=0)
order, cty = s1["entity_id"].to_list(), dict(zip(s1["entity_id"], s1["country"]))
changed = 0
for q in order:
    if cty[q] == country:
        if bm.get(q, set()) != om.get(q, set()): changed += 1
        bm[q] = set(om.get(q, set())); bc[q] = bc.get(q, set()) | oc.get(q, set()) | bm[q]
dest = ROOT / "outputs/submissions" / name; dest.mkdir(parents=True, exist_ok=False)
hdr = open(base / "candidate_pairs.tsv").readline()
with open(dest / "matching_results.tsv", "w") as fm, open(dest / "candidate_pairs.tsv", "w") as fc:
    fm.write("source1_entity_id\tmatched_entity_ids\n"); fc.write(hdr)
    for q in order:
        assert bm.get(q, set()) <= bc.get(q, set()), q
        fm.write(q + "\t" + ",".join(sorted(bm.get(q, set()))) + "\n"); fc.write(q + "\t" + ",".join(sorted(bc.get(q, set()))) + "\n")
links = sum(len(bm.get(q, ())) for q in order if cty[q] == country)
print(json.dumps({"changed_s1": changed, "country_links": links, "sha": hashlib.sha256((dest / "matching_results.tsv").read_bytes()).hexdigest()}))
