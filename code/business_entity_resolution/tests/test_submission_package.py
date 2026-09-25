"""Critical final-output alignment and membership checks."""

from pathlib import Path

import pytest

from src.submission.merge_outputs import validate


def _fixture(root: Path) -> tuple[Path, Path, Path]:
    test_dir = root / "test"
    test_dir.mkdir()
    (test_dir / "test_source1.tsv").write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-0001\tFirst\tStreet 1\tFrance\n"
        "S1-0002\tSecond\tStreet 2\tIndia\n"
    )
    matching = root / "matching_results.tsv"
    candidates = root / "candidate_pairs.tsv"
    matching.write_text("source1_entity_id\tmatched_entity_ids\nS1-0001\tS2-0010\nS1-0002\t\n")
    candidates.write_text("source1_entity_id\tcandidate_entity_ids\nS1-0001\tS2-0010,S3-0005\nS1-0002\tS3-0006\n")
    return matching, candidates, test_dir


def test_final_outputs_include_singleton_and_scored_match(tmp_path: Path) -> None:
    matching, candidates, test_dir = _fixture(tmp_path)
    validate(matching, candidates, test_dir)


def test_final_outputs_reject_unscored_prediction(tmp_path: Path) -> None:
    matching, candidates, test_dir = _fixture(tmp_path)
    matching.write_text("source1_entity_id\tmatched_entity_ids\nS1-0001\tS2-OTHER\nS1-0002\t\n")
    with pytest.raises(ValueError, match="Unscored prediction"):
        validate(matching, candidates, test_dir)


def test_final_outputs_reject_missing_source1(tmp_path: Path) -> None:
    matching, candidates, test_dir = _fixture(tmp_path)
    candidates.write_text("source1_entity_id\tcandidate_entity_ids\nS1-0001\tS2-0010\n")
    with pytest.raises(ValueError):
        validate(matching, candidates, test_dir)
