import gzip
import importlib.util
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "sub001_merge", Path(__file__).resolve().parents[2] / "scripts/submissions/merge_validate.py"
)
merge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(merge)


def write_shard(path, header, rows):
    with gzip.open(path, "wt", encoding="utf-8") as out:
        out.write(header)
        out.writelines(rows)


def test_merge_preserves_empty_singletons_and_global_order(tmp_path):
    shards = tmp_path / "shards"
    shards.mkdir()
    write_shard(shards / "b-matching.tsv.gz", merge.HEADERS["matching"],
                ["S1-00001\t\n", "S1-00003\tS2-00002\n"])
    write_shard(shards / "a-matching.tsv.gz", merge.HEADERS["matching"],
                ["S1-00002\tS3-00007\n"])
    target = tmp_path / "matching.tsv"
    assert merge.merged("matching", shards, target) == (3, 2)
    assert target.read_text() == (
        merge.HEADERS["matching"] + "S1-00001\t\n"
        + "S1-00002\tS3-00007\n" + "S1-00003\tS2-00002\n"
    )


def test_merge_rejects_duplicate_source1_across_shards(tmp_path):
    for name in ("a", "b"):
        write_shard(tmp_path / f"{name}-candidates.tsv.gz",
                    merge.HEADERS["candidates"], ["S1-00001\tS2-00002\n"])
    with pytest.raises(ValueError, match="Duplicate or unsorted"):
        merge.merged("candidates", tmp_path, tmp_path / "merged.tsv")
