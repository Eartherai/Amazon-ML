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
