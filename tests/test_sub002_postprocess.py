"""Small exact cases for SUB-002 source completion and ownership postprocessing."""

import gzip
from pathlib import Path

import pytest

from scripts.submissions.postprocess_sub002 import postprocess


def test_addition_conflict_and_singleton(tmp_path: Path) -> None:
    matching = tmp_path / "matching.tsv"
    matching.write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-A\tS2-X\n"
        "S1-B\tS2-Y,S3-Z\n"
        "S1-C\t\n"
        "S1-D\tS2-X\n"
    )
    scores = tmp_path / "scores"
    scores.mkdir()
    for index in range(192):
        with gzip.open(scores / f"part-{index:03d}-scores.tsv.gz", "wt") as target:
            target.write("source1_entity_id\ttarget_id\tscore\n")
            if index == 0:
                target.write("S1-A\tS2-X\t0.95\n")
                target.write("S1-A\tS3-Z\t0.8\n")
                target.write("S1-B\tS2-Y\t0.91\n")
                target.write("S1-B\tS3-Z\t0.88\n")
                target.write("S1-D\tS2-X\t0.87\n")
                target.write("S1-D\tS3-Q\t0.54\n")
    output = tmp_path / "final.tsv"
    report = postprocess(matching, scores, output, 4)
    assert output.read_text() == (
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-A\tS2-X\n"
        "S1-B\tS2-Y,S3-Z\n"
        "S1-C\t\n"
        "S1-D\t\n"
    )
    assert report["added_links_before_ownership"] == 1
    assert report["owner_links_removed"] == 2
    assert report["final_links"] == 3
    with pytest.raises(FileExistsError):
        postprocess(matching, scores, output, 4)
