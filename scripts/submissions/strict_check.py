"""Independent strict verification of a matching_results.tsv (and optional candidate_pairs.tsv).

Checks: UTF-8 across the whole file, exact header, tab-delimited two columns, one row per
test Source 1 (exact universe, no duplicates), matched IDs are S2-/S3- IDs existing in the
test files, no duplicates/empties/'nan'/'None'/'null' tokens, ascending sort within lists,
and (if candidates given) every matched ID appears in the same S1's candidate list.
Exit code 0 only if everything passes.
"""
import argparse, csv, hashlib, json, sys
from pathlib import Path

BAD = {"nan", "none", "null", "na", ""}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 23), b""):
            h.update(b)
    return h.hexdigest()


def ids_of(path, col="entity_id"):
    with open(path, encoding="utf-8", newline="") as f:
        return {row[col] for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)}


def read_lists(path, header, errors, name):
    out = {}
    with open(path, "rb") as fb:
        raw = fb.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        errors.append(f"{name}: not UTF-8 ({e})")
        return out
    lines = text.split("\n")
    if lines[-1] == "":
        lines = lines[:-1]
    if not lines or lines[0] != header:
        errors.append(f"{name}: bad header {lines[0][:80] if lines else ''!r}")
        return out
    for n, line in enumerate(lines[1:], start=2):
        if "\r" in line:
            errors.append(f"{name}: CR at line {n}")
        parts = line.split("\t")
        if len(parts) != 2:
            errors.append(f"{name}: line {n} has {len(parts)} columns")
            continue
        q, raw_ids = parts
        if q in out:
            errors.append(f"{name}: duplicate S1 row {q}")
        ids = raw_ids.split(",") if raw_ids else []
        for t in ids:
            if t.strip().lower() in BAD or t != t.strip():
                errors.append(f"{name}: bad token {t!r} for {q}")
        if len(ids) != len(set(ids)):
            errors.append(f"{name}: duplicate IDs in list for {q}")
        if ids != sorted(ids):
            errors.append(f"{name}: unsorted list for {q}")
        out[q] = ids
        if len(errors) > 50:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matching", required=True)
    ap.add_argument("--candidate")
    ap.add_argument("--test-dir", required=True)
    a = ap.parse_args()
    errors = []
    td = Path(a.test_dir)
    s1 = ids_of(td / "test_source1.tsv")
    targets = ids_of(td / "test_source2.tsv") | ids_of(td / "test_source3.tsv")
    m = read_lists(a.matching, "source1_entity_id\tmatched_entity_ids", errors, "matching")
    if set(m) != s1:
        errors.append(f"matching: S1 universe differs (missing {len(s1 - set(m))}, extra {len(set(m) - s1)})")
    links = 0
    for q, ids in m.items():
        links += len(ids)
        for t in ids:
            if not t.startswith(("S2-", "S3-")) or t not in targets:
                errors.append(f"matching: invalid target {t} for {q}")
                break
    cand_ok = None
    if a.candidate:
        c = read_lists(a.candidate, "source1_entity_id\tcandidate_entity_ids", errors, "candidate")
        if set(c) != s1:
            errors.append("candidate: S1 universe differs")
        missing = [(q, t) for q, ids in m.items() for t in ids if t not in set(c.get(q, []))]
        if missing:
            errors.append(f"matched IDs absent from candidates: {len(missing)} e.g. {missing[:3]}")
        cand_ok = not missing
    report = {"matching": a.matching, "matching_sha256": sha(a.matching), "rows": len(m), "expected_rows": len(s1),
              "links": links, "empty": sum(1 for v in m.values() if not v), "candidate_subset_ok": cand_ok,
              "errors": errors[:50], "pass": not errors}
    if a.candidate:
        report["candidate_sha256"] = sha(a.candidate)
    print(json.dumps(report, indent=1))
    sys.exit(0 if not errors else 1)


if __name__ == "__main__":
    main()
