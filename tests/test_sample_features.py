"""Check that independent retrieval routes are joined before scoring."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

from scripts.aws.build_sample_features import route_scores


def _route(path: Path, rows: list[tuple[str, str, float, int]]) -> str:
    pl.DataFrame(rows, schema=["source1_entity_id", "target_id", "route_score", "route_rank"],
                 orient="row").write_parquet(path)
    return str(path)


def test_name_address_union_tracks_two_routes(tmp_path: Path) -> None:
    name = _route(tmp_path / "name.parquet", [("S1-1", "S2-1", .8, 1), ("S1-1", "S3-2", .6, 2)])
    address = _route(tmp_path / "address.parquet", [("S1-1", "S2-1", .5, 1)])
    slots = {}
    route_scores([name], 0, slots)
    route_scores([address], 1, slots)
    assert set(slots) == {("S1-1", "S2-1"), ("S1-1", "S3-2")}
    assert slots[("S1-1", "S2-1")][4] == 2
    assert slots[("S1-1", "S3-2")][3] == 0


def test_duplicate_pair_in_one_route_fails(tmp_path: Path) -> None:
    name = _route(tmp_path / "name.parquet", [("S1-1", "S2-1", .8, 1), ("S1-1", "S2-1", .7, 2)])
    with pytest.raises(ValueError, match="Duplicate"):
        route_scores([name], 0, {})


def test_full_route_skips_locked_query_before_feature_materialization(tmp_path: Path) -> None:
    route = _route(tmp_path / "full.parquet", [
        ("S1-unlocked", "S2-a", .8, 1), ("S1-locked", "S2-b", .9, 1)])
    slots = {}
    route_scores([route], 0, slots, {"S1-unlocked"})
    assert set(slots) == {("S1-unlocked", "S2-a")}


def test_full_tiny_feature_store(tmp_path: Path) -> None:
    inputs, routes, output = (tmp_path / name for name in ("inputs", "routes", "output"))
    inputs.mkdir()
    routes.mkdir()
    pl.DataFrame({"entity_id": ["S1-1", "S1-2", "S1-3"], "country": ["India"] * 3,
                  "n": ["alpha", "beta", "gamma"], "a": ["1 lane", "2 lane", "3 lane"],
                  "fold": [1, 2, 3], "n_matches": [1, 0, 0]}).write_parquet(inputs / "labeled_queries.parquet")
    pl.DataFrame({"entity_id": ["S2-1", "S2-2", "S2-3"], "country": ["India"] * 3,
                  "n": ["alpha", "beta", "gamma"],
                  "a": ["1 lane", "2 lane", "3 lane"]}).write_parquet(inputs / "targets.parquet")
    pl.DataFrame({"target_id": ["S2-1", "S2-2", "S2-3"], "owner_fold": [1, 2, 3]}).write_parquet(inputs / "ownership.parquet")
    pl.DataFrame({"source1_entity_id": ["S1-1"], "target_id": ["S2-1"]}).write_parquet(inputs / "truth.parquet")
    pl.DataFrame(schema={"n": pl.String, "transliterated": pl.String}).write_parquet(inputs / "name_map.parquet")
    file_rows = []
    for path in inputs.iterdir():
        file_rows.append({"name": path.name, "bytes": path.stat().st_size,
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (inputs / "manifest.json").write_text(json.dumps({"files": file_rows}))
    for field in ("name", "address"):
        _route(routes / f"{field}-c000-s000-b00000.parquet",
               [(f"S1-{n}", f"S2-{n}", .9, 1) for n in (1, 2, 3)])
    env = {**os.environ, "PYTHONPATH": "code/business_entity_resolution:."}
    run = subprocess.run([sys.executable, "scripts/aws/build_sample_features.py", "--routes", str(routes),
                          "--inputs", str(inputs), "--output", str(output), "--shards", "1",
                          "--chunk-rows", "2", "--expected-queries", "3", "--expected-targets", "3"],
                         check=False, env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    frame = pl.read_parquet(str(output / "features-*.parquet"))
    metrics = json.loads((output / "metrics.json").read_text())
    assert len(frame) == 3 and frame["label"].sum() == 1
    assert metrics["candidate_pairs"] == 3
    assert metrics["link_recall"] == 1.0
    assert metrics["complete_positive_entity_recall"] == 1.0
