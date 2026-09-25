"""Protect the full old-sample route parity gate before EXP-032 launch."""

import json
import tarfile

import polars as pl
import pytest

from scripts.aws.verify_sample_retrieval_parity import compare_route, sha256, validate_archive, verify_boundary_negatives


def test_archive_hash_and_old_sample_route_parity(tmp_path) -> None:
    summary = tmp_path / "summary"
    summary.mkdir()
    member = tmp_path / "name-c000-s000-b00000.parquet"
    pl.DataFrame({"source1_entity_id": ["S1-0", "S1-OTHER"],
                  "target_id": ["S2-0", "S2-OTHER"],
                  "route_score": [0.9, 0.2], "route_rank": [1, 1]}).write_parquet(member)
    archive = tmp_path / "name-c000-s000.tar"
    with tarfile.open(archive, "w") as destination:
        destination.add(member, arcname=member.name)
    receipt = summary / "name-c000-s000-receipt.json"
    receipt.write_text(json.dumps({"bytes": archive.stat().st_size, "sha256": sha256(archive)}))
    selected = tmp_path / "selected.parquet"
    validate_archive(archive, summary, selected, {"S1-0"})
    assert pl.read_parquet(selected)["source1_entity_id"].to_list() == ["S1-0"]
    receipt.write_text(json.dumps({"bytes": archive.stat().st_size, "sha256": "0" * 64}))
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_archive(archive, summary, tmp_path / "ignored.parquet", {"S1-0"})

    frame = pl.DataFrame({"source1_entity_id": [f"S1-{index}" for index in range(20_000)],
                          "target_id": [f"S2-{index}" for index in range(20_000)],
                          "route_rank": [1] * 20_000, "route_score": [0.9] * 20_000})
    reference, actual = tmp_path / "reference.parquet", tmp_path / "actual.parquet"
    frame.write_parquet(reference)
    frame.write_parquet(actual)
    assert compare_route("name", reference, str(actual), expected_pairs=20_000)["missing_id_or_rank"] == 0
    frame = frame.with_columns(pl.when(pl.col("source1_entity_id") == "S1-0")
                               .then(pl.lit("S2-WRONG")).otherwise(pl.col("target_id")).alias("target_id"))
    frame.write_parquet(actual)
    with pytest.raises(ValueError, match="Route parity failed"):
        compare_route("name", reference, str(actual), expected_pairs=20_000)


def test_float32_near_tie_rank_swap_is_reported_not_called_exact(tmp_path) -> None:
    reference = tmp_path / "reference.parquet"
    actual = tmp_path / "actual.parquet"
    base = {"source1_entity_id": ["S1-0", "S1-0"], "target_id": ["S2-A", "S2-B"],
            "route_score": [0.5, 0.5], "route_rank": [1, 2]}
    pl.DataFrame(base).write_parquet(reference)
    pl.DataFrame({**base, "route_score": [0.50000006, 0.5], "route_rank": [2, 1]}).write_parquet(actual)
    report = compare_route("name", reference, str(actual), expected_pairs=2, expected_queries=1)
    assert report["numeric_tie_equivalence"] is True
    assert report["exact_id_rank_parity"] is False
    assert report["missing_candidate_ids"] == report["extra_candidate_ids"] == 0
    assert report["rank_changed_pairs"] == 2


def test_top100_boundary_swap_requires_known_negative(tmp_path) -> None:
    reference = tmp_path / "reference.parquet"
    actual = tmp_path / "actual.parquet"
    truth = tmp_path / "truth.tsv"
    ids = [f"S2-{index}" for index in range(99)]
    pl.DataFrame({"source1_entity_id": ["S1-0"] * 100, "target_id": ids + ["S2-OLD"],
                  "route_score": [1 - index / 200 for index in range(99)] + [0.5],
                  "route_rank": list(range(1, 101))}).write_parquet(reference)
    pl.DataFrame({"source1_entity_id": ["S1-0"] * 100, "target_id": ids + ["S2-NEW"],
                  "route_score": [1 - index / 200 for index in range(99)] + [0.50000006],
                  "route_rank": list(range(1, 101))}).write_parquet(actual)
    report = compare_route("address", reference, str(actual), expected_pairs=100, expected_queries=1)
    assert report["numeric_tie_equivalence"] is True
    truth.write_text("source1_entity_id\tmatched_entity_ids\nS1-0\tS2-1\n")
    assert len(verify_boundary_negatives([report], truth)) == 2
    truth.write_text("source1_entity_id\tmatched_entity_ids\nS1-0\tS2-OLD\n")
    with pytest.raises(ValueError, match="known positive"):
        verify_boundary_negatives([report], truth)
