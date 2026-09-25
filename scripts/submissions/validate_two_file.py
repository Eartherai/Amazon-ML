"""Two-file official validation for final submissions (runs on an EC2 worker; stdlib only).

For every <submissions>/<NAME>/ holding matching_results.tsv and candidate_additions.tsv
(S1<TAB>target rows, may be empty): build candidate_pairs.tsv = frozen SUB-002 candidate
shards (192) plus that submission's additions (content-identical duplicate expansion and
rescue-route pairs), in test Source 1 order with sorted unique lists; then run the unchanged
organizer validator (default and --check-ids) and strict_check.py with the candidate file.
"""
import argparse, gzip, hashlib, json, subprocess, sys
from collections import defaultdict
from pathlib import Path


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 23), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=Path, required=True)
    ap.add_argument("--test-dir", type=Path, required=True)
    ap.add_argument("--submissions", type=Path, required=True)
    ap.add_argument("--validator", type=Path, required=True)
    ap.add_argument("--strict", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    files = sorted(a.shards.glob("*-candidates.tsv.gz"))
    if len(files) != 192:
        raise ValueError(f"expected 192 candidate shards, got {len(files)}")
    base = {}
    for p in files:
        with gzip.open(p, "rt", encoding="utf-8") as f:
            assert f.readline() == "source1_entity_id\tcandidate_entity_ids\n"
            for line in f:
                q, _, raw = line.rstrip("\n").partition("\t")
                if q in base:
                    raise ValueError("duplicate S1 across candidate shards")
                base[q] = raw
    with open(a.test_dir / "test_source1.tsv", encoding="utf-8") as f:
        f.readline()
        order = [line.split("\t", 1)[0] for line in f if line.strip()]
    if set(order) != set(base):
        raise ValueError("candidate S1 universe differs from test")
    a.output.mkdir(parents=True, exist_ok=True)
    summary = {}
    for sub in sorted(p for p in a.submissions.iterdir() if p.is_dir()):
        out = a.output / sub.name
        out.mkdir(parents=True, exist_ok=True)
        add = defaultdict(set)
        addp = sub / "candidate_additions.tsv"
        if addp.exists():
            for line in addp.read_text().splitlines():
                if line.strip():
                    q, t = line.split("\t")
                    add[q].add(t)
        cand = out / "candidate_pairs.tsv"
        with open(cand, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            for q in order:
                ids = set(base[q].split(",")) if base[q] else set()
                ids |= add.get(q, set())
                f.write(q + "\t" + ",".join(sorted(ids)) + "\n")
        matching = sub / "matching_results.tsv"
        res = {"matching_sha256": sha(matching), "candidate_sha256": sha(cand), "additions": sum(map(len, add.values()))}
        for mode, extra in (("official_default", []), ("official_check_ids", ["--check-ids"])):
            r = subprocess.run([sys.executable, str(a.validator), "--matching", str(matching), "--candidate", str(cand),
                                "--test-dir", str(a.test_dir), *extra], capture_output=True, text=True)
            (out / f"{mode}.log").write_text(r.stdout + r.stderr)
            res[mode] = {"exit_code": r.returncode, "pass": "PASS" in r.stdout,
                         "warnings": [l for l in (r.stdout + r.stderr).splitlines() if l.startswith("WARNING")]}
        r = subprocess.run([sys.executable, str(a.strict), "--matching", str(matching), "--candidate", str(cand),
                            "--test-dir", str(a.test_dir)], capture_output=True, text=True)
        (out / "strict_check.json").write_text(r.stdout + r.stderr)
        res["strict"] = {"exit_code": r.returncode}
        with open(cand, "rb") as src, gzip.open(out / "candidate_pairs.tsv.gz", "wb", compresslevel=6) as dst:
            for b in iter(lambda: src.read(1 << 23), b""):
                dst.write(b)
        cand.unlink()
        res["all_pass"] = all(res[m]["exit_code"] == 0 and res[m]["pass"] for m in ("official_default", "official_check_ids")) and res["strict"]["exit_code"] == 0
        summary[sub.name] = res
        (out / "result.json").write_text(json.dumps(res, indent=2))
        print(json.dumps({sub.name: res}), flush=True)
    (a.output / "SUMMARY.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
