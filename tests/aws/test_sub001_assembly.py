import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "sub001_assembly", Path(__file__).resolve().parents[2] / "scripts/submissions/assemble_sub001_countries.py"
)
assembly = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assembly)


def test_country_assembly_requires_verified_complete_shards(tmp_path, monkeypatch):
    counts = {"France": 1, "India": 1, "US": 1}
    monkeypatch.setattr(assembly, "COUNTS", counts)
    mac = tmp_path / "mac"
    us = tmp_path / "us"
    for root in (mac, us):
        (root / "shards").mkdir(parents=True)
    ids = {"France": "S1-France", "India": "S1-India", "US": "S1-US"}
    for country, query in ids.items():
        root = us if country == "US" else mac
        occupied = int.from_bytes(hashlib.sha256(query.encode()).digest()[:8], "big") % 64
        for number in range(64):
            base = f"{country}-s{number:03d}"
            receipts = []
            for kind, header in assembly.HEADER.items():
                path = root / "shards" / f"{base}-{kind}.tsv.gz"
                with gzip.open(path, "wt", encoding="utf-8") as handle:
                    handle.write(header)
                    if number == occupied:
                        handle.write(query + "\t" + ("S2-target" if kind == "candidates" else "") + "\n")
                receipts.append({"key": "prefix/" + path.name, "bytes": path.stat().st_size,
                                 "sha256": assembly.digest(path)})
            if country == "US":
                (root / "shards" / f"{base}-receipts.json").write_text(json.dumps(receipts))
    progress = [{"country": country, "queries": 1, "shards": 64} for country in counts]
    common = {"config": {"threshold": 0.83}, "model_sha256": "model", "query_total": 3,
              "fold4": "CLOSED"}
    (mac / "progress.json").write_text(json.dumps({**common, "country_progress": progress[:2]}))
    (us / "COMPLETE.json").write_text(json.dumps({**common, "countries": ["US"],
                                                   "country_progress": progress[2:]}))
    output = tmp_path / "assembled"
    monkeypatch.setattr(sys, "argv", ["assemble", "--mac", str(mac), "--us", str(us),
                                      "--output", str(output)])
    assembly.main()
    result = json.loads((output / "COMPLETE.json").read_text())
    assert result["processed_query_count"] == 3
    assert len(list((output / "shards").glob("*.tsv.gz"))) == 384

    wrong = us / "shards" / "US-s000-receipts.json"
    receipt = json.loads(wrong.read_text())
    receipt[0]["sha256"] = "0" * 64
    wrong.write_text(json.dumps(receipt))
    monkeypatch.setattr(sys, "argv", ["assemble", "--mac", str(mac), "--us", str(us),
                                      "--output", str(tmp_path / "corrupted")])
    with pytest.raises(ValueError, match="differs from verified upload"):
        assembly.main()


def test_country_assembly_combines_mac_india_prefix_with_cloud_remainder(tmp_path, monkeypatch):
    monkeypatch.setattr(assembly, "COUNTS", {"France": 1, "India": 2, "US": 1})
    monkeypatch.setattr(assembly, "INDIA_CLOUD_COUNT", 1)
    monkeypatch.setattr(assembly, "INDIA_FIRST_COUNT", 1)
    roots = {"mac": tmp_path / "mac", "india": tmp_path / "india", "us": tmp_path / "us"}
    for root in roots.values():
        (root / "shards").mkdir(parents=True)

    def shard_id(value):
        return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") % 64

    low = next(f"S1-India-low-{i}" for i in range(1000) if shard_id(f"S1-India-low-{i}") < 8)
    high = next(f"S1-India-high-{i}" for i in range(1000) if shard_id(f"S1-India-high-{i}") >= 8)
    ids = {"France": ["S1-France"], "India": [low, high], "US": ["S1-US"]}
    for country, queries in ids.items():
        for number in range(64):
            root = (roots["us"] if country == "US" else
                    roots["india"] if country == "India" and number >= 8 else roots["mac"])
            receipts = []
            for kind, header in assembly.HEADER.items():
                path = root / "shards" / f"{country}-s{number:03d}-{kind}.tsv.gz"
                with gzip.open(path, "wt", encoding="utf-8") as handle:
                    handle.write(header)
                    for query in sorted(q for q in queries if shard_id(q) == number):
                        handle.write(query + "\t\n")
                receipts.append({"key": "prefix/" + path.name, "bytes": path.stat().st_size,
                                 "sha256": assembly.digest(path)})
            if root != roots["mac"]:
                (root / "shards" / f"{country}-s{number:03d}-receipts.json").write_text(json.dumps(receipts))
    common = {"config": {"threshold": 0.83}, "model_sha256": "model", "query_total": 4,
              "fold4": "CLOSED"}
    (roots["mac"] / "progress.json").write_text(json.dumps({**common, "country_progress": [
        {"country": "France", "queries": 1, "shards": 64}],
        "active": {"country": "India", "queries": 1, "shards": 8}}))
    (roots["india"] / "COMPLETE.json").write_text(json.dumps({**common, "countries": ["India"],
        "processed_query_count": 1, "country_progress": [{"country": "India", "queries": 1, "shards": 56}]}))
    (roots["us"] / "COMPLETE.json").write_text(json.dumps({**common, "countries": ["US"],
        "country_progress": [{"country": "US", "queries": 1, "shards": 64}]}))
    output = tmp_path / "assembled"
    monkeypatch.setattr(sys, "argv", ["assemble", "--mac", str(roots["mac"]),
        "--india", str(roots["india"]), "--us", str(roots["us"]), "--output", str(output)])
    assembly.main()
    result = json.loads((output / "COMPLETE.json").read_text())
    assert result["processed_query_count"] == 4
    assert result["country_progress"][1]["sources"] == {"Mac": "0-7", "EC2": "8-63"}


def test_country_assembly_combines_verified_india_split(tmp_path, monkeypatch):
    counts = {"France": 1, "India": 3, "US": 1}
    for key, value in (("COUNTS", counts), ("INDIA_FIRST_COUNT", 1),
                       ("INDIA_LOWER_COUNT", 1), ("INDIA_UPPER_COUNT", 1)):
        monkeypatch.setattr(assembly, key, value)
    roots = {name: tmp_path / name for name in ("mac", "lower", "upper", "us")}
    for root in roots.values():
        (root / "shards").mkdir(parents=True)

    def shard_id(value):
        return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") % 64

    def query_in_range(prefix, first, last):
        return next(f"S1-{prefix}-{i}" for i in range(10000)
                    if first <= shard_id(f"S1-{prefix}-{i}") < last)

    india = [query_in_range("India-first", 0, 8),
             query_in_range("India-lower", 8, 36),
             query_in_range("India-upper", 36, 64)]
    ids = {"France": ["S1-France"], "India": india, "US": ["S1-US"]}
    for country, queries in ids.items():
        for number in range(64):
            root = (roots["us"] if country == "US" else
                    roots["upper"] if country == "India" and number >= 36 else
                    roots["lower"] if country == "India" and number >= 8 else roots["mac"])
            receipts = []
            for kind, header in assembly.HEADER.items():
                path = root / "shards" / f"{country}-s{number:03d}-{kind}.tsv.gz"
                with gzip.open(path, "wt", encoding="utf-8") as handle:
                    handle.write(header)
                    for query in sorted(q for q in queries if shard_id(q) == number):
                        handle.write(query + "\t\n")
                receipts.append({"key": "prefix/" + path.name, "bytes": path.stat().st_size,
                                 "sha256": assembly.digest(path)})
            if root != roots["mac"]:
                (root / "shards" / f"{country}-s{number:03d}-receipts.json").write_text(json.dumps(receipts))
    common = {"config": {"threshold": 0.83}, "model_sha256": "model", "query_total": 5,
              "fold4": "CLOSED"}
    (roots["mac"] / "progress.json").write_text(json.dumps({**common, "country_progress": [
        {"country": "France", "queries": 1, "shards": 64}],
        "active": {"country": "India", "queries": 1, "shards": 8}}))
    (roots["us"] / "COMPLETE.json").write_text(json.dumps({**common, "countries": ["US"],
        "country_progress": [{"country": "US", "queries": 1, "shards": 64}]}))
    (roots["upper"] / "COMPLETE.json").write_text(json.dumps({**common, "countries": ["India"],
        "processed_query_count": 1, "country_progress": [{"country": "India", "queries": 1, "shards": 28}]}))
    for name, first, last, filename in (("lower", 8, 36, "SEGMENT_READY.json"),
                                        ("upper", 36, 64, "DOWNLOAD_READY.json")):
        ready = {"run_id": "P5-SUB001-INDIA-001" if name == "lower" else "P5-SUB001-INDIA-HIGH-001",
                 "country": "India", "first_shard": first, "last_shard": last,
                 "query_count": 1, "pairs": last-first, "model_sha256": "model",
                 "partial_worker_terminated": name == "lower"}
        (roots[name] / filename).write_text(json.dumps(ready))
    args = ["assemble", "--mac", str(roots["mac"]), "--us", str(roots["us"]),
            "--india", str(roots["lower"]), "--india-upper", str(roots["upper"]),
            "--output", str(tmp_path / "assembled")]
    monkeypatch.setattr(sys, "argv", args)
    assembly.main()
    result = json.loads((tmp_path / "assembled/COMPLETE.json").read_text())
    assert result["processed_query_count"] == 5
    assert result["country_progress"][1]["sources"] == {
        "Mac": "0-7", "EC2-lower": "8-35", "EC2-upper": "36-63"}
    ready_path = roots["lower"] / "SEGMENT_READY.json"
    ready = json.loads(ready_path.read_text())
    ready["partial_worker_terminated"] = False
    ready_path.write_text(json.dumps(ready))
    monkeypatch.setattr(sys, "argv", [*args[:-1], str(tmp_path / "rejected")])
    with pytest.raises(ValueError, match="not stopped"):
        assembly.main()
