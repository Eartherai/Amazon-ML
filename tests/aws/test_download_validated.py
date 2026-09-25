"""Verify that downloaded gzip payloads are checked against raw TSV hashes."""
import gzip
import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/submissions"))
import download_validated


def test_expand_checks_raw_hash_and_preserves_complete_content(tmp_path, monkeypatch):
    monkeypatch.setattr(download_validated, "RESERVE_BYTES", 0)
    content = b"source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1\n"
    compressed = tmp_path / "matching.tsv.gz"
    with gzip.open(compressed, "wb") as target:
        target.write(content)
    raw = tmp_path / "matching.tsv"
    download_validated.expand(compressed, raw, hashlib.sha256(content).hexdigest())
    assert raw.read_bytes() == content
    with pytest.raises(ValueError, match="checksum mismatch"):
        download_validated.expand(compressed, tmp_path / "bad.tsv", "0" * 64)
