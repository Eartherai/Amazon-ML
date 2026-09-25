import importlib.util
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "india_stop", Path(__file__).resolve().parents[2] / "scripts/aws/stop_india_lower_at_boundary.py"
)
stop = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stop)


def test_boundary_inventory_covers_only_lower_half() -> None:
    keys = stop.expected_lower_keys("prefix")
    assert len(keys) == 56
    assert "prefix/shards/India-s008-candidates.tsv.gz" in keys
    assert "prefix/shards/India-s035-matching.tsv.gz" in keys
    assert not any("India-s036-" in key for key in keys)


def test_instance_rejects_wrong_worker_tag(monkeypatch) -> None:
    monkeypatch.setattr(stop, "aws", lambda *args: {"Reservations": [{"Instances": [{
        "Tags": [{"Key": "RunId", "Value": "wrong"},
                 {"Key": "Project", "Value": "aml2026-phase5"}],
        "State": {"Name": "running"}}]}]})
    with pytest.raises(ValueError, match="Wrong EC2 worker identity"):
        stop.instance(stop.LOWER_RUN, "i-test")
