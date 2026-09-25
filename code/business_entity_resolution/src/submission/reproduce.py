"""Recreate both SUB-001 output TSVs from official test TSVs and frozen artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def call(module: str, *arguments: object) -> None:
    subprocess.run([sys.executable, "-m", module, *(str(value) for value in arguments)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--official-validator", type=Path)
    args = parser.parse_args()
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    if args.work_dir.exists() or args.output_dir.exists():
        raise FileExistsError("Work and output directories must be new")
    call("src.submission.prepare_inputs", "--test-dir", args.test_dir, "--output", args.work_dir / "inputs")
    call("src.submission.run_sub001", "--inputs", args.work_dir / "inputs", "--output", args.work_dir / "inference",
         "--shards", 64, "--batch-size", 200, "--threads", args.threads)
    complete = json.loads((args.work_dir / "inference" / "COMPLETE.json").read_text())
    if complete["processed_query_count"] != 1_732_544 or sum(part["shards"] for part in complete["country_progress"]) != 192:
        raise RuntimeError("Inference did not cover the complete test set")
    merge_args = ["--shards", args.work_dir / "inference" / "shards", "--output", args.output_dir,
                  "--test-dir", args.test_dir]
    if args.official_validator:
        merge_args.extend(("--official-validator", args.official_validator))
    call("src.submission.merge_outputs", *merge_args)


if __name__ == "__main__":
    main()
