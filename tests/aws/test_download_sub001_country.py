import base64
import gzip
import hashlib
import importlib.util
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "download_sub001_country", Path(__file__).resolve().parents[2] / "scripts/submissions/download_sub001_country.py"
)
downloader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(downloader)


def test_verified_get_requires_service_sha256_and_length(tmp_path, monkeypatch):
    content = b"source1_entity_id\tmatched_entity_ids\nS1-a\t\n"
    checksum = base64.b64encode(hashlib.sha256(content).digest()).decode()

    def response(*args):
        Path(args[-1]).write_bytes(content)
        return {"ChecksumSHA256": checksum, "ContentLength": len(content), "VersionId": "v1"}

    monkeypatch.setattr(downloader, "aws", response)
    receipt = downloader.verified_get("prefix/file", tmp_path / "file")
    assert receipt["sha256"] == hashlib.sha256(content).hexdigest()

    def bad_response(*args):
        row = response(*args)
        row["ChecksumSHA256"] = base64.b64encode(bytes(32)).decode()
        return row

    monkeypatch.setattr(downloader, "aws", bad_response)
    with pytest.raises(ValueError, match="checksum mismatch"):
        downloader.verified_get("prefix/file", tmp_path / "file")


def test_smoke_parity_checks_exact_rows(tmp_path):
    smoke = tmp_path / "smoke"
    downloaded = tmp_path / "cloud"
    smoke.mkdir()
    downloaded.mkdir()
    for kind in ("candidates", "matching"):
        name = f"US-s000-{kind}.tsv.gz"
        header = "source1_entity_id\t" + ("candidate_entity_ids" if kind == "candidates" else "matched_entity_ids") + "\n"
        for root in (smoke, downloaded):
            with gzip.open(root / name, "wt", encoding="utf-8") as output:
                output.write(header)
                for index in range(100):
                    output.write(f"S1-{index:05d}\tS2-1\n")
    assert downloader.smoke_parity(downloaded, smoke, "US", 0, 1) == {"candidates": 100, "matching": 100}
    with gzip.open(downloaded / "US-s000-matching.tsv.gz", "wt", encoding="utf-8") as output:
        output.write("source1_entity_id\tmatched_entity_ids\n")
        for index in range(100):
            output.write(f"S1-{index:05d}\t" + ("S2-2" if index == 99 else "S2-1") + "\n")
    with pytest.raises(ValueError, match="content mismatch"):
        downloader.smoke_parity(downloaded, smoke, "US", 0, 1)
