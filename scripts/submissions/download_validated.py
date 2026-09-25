"""Download and verify completed SageMaker SUB-001 validator artifacts."""
import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


CONFIG = Path("configs/aws/P5-SUB001-SM-VALIDATE-001.json")
BUCKET = "aml2026-ber-08be19ac500747"
RESERVE_BYTES = 8 * 1024**3


def aws(*args: str) -> dict:
    command = ["aws", "--profile", os.environ.get("AWS_PROFILE", "amamzon_01_a1_0"),
               "--region", "us-east-1", "--no-cli-pager", *args, "--output", "json"]
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    return json.loads(result.stdout) if result.stdout.strip() else {}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def reserve(path: Path) -> None:
    if shutil.disk_usage(path).free < RESERVE_BYTES:
        raise RuntimeError("Less than 8 GiB free; refusing to expand submission outputs")


def download(prefix: str, name: str, destination: Path) -> None:
    aws("s3api", "get-object", "--bucket", BUCKET, "--key", prefix + name, str(destination))


def expand(source: Path, target: Path, expected_sha256: str) -> None:
    with gzip.open(source, "rb") as zipped, target.open("xb") as raw:
        shutil.copyfileobj(zipped, raw, length=8 * 1024**2)
    if sha256(target) != expected_sha256:
        raise ValueError(f"Uncompressed TSV checksum mismatch: {target.name}")
    reserve(target.parent)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matching-only", action="store_true",
                        help="Download only the portal-required matching TSV plus validation evidence")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    job = aws("sagemaker", "describe-processing-job", "--processing-job-name", config["run_id"])
    if job["ProcessingJobStatus"] != "Completed":
        raise RuntimeError("Validator processing job has not completed successfully")
    args.output.mkdir(parents=True, exist_ok=False)
    reserve(args.output)
    prefix = config["output_prefix"]
    download(prefix, "manifest.json", args.output / "manifest.json")
    manifest = json.loads((args.output / "manifest.json").read_text())
    if manifest["scope"] != "Frozen SUB-001 complete test results; Fold4 CLOSED" or manifest["rows"] != 1732544:
        raise ValueError("Unexpected validator result manifest")
    required = {"matching_results.tsv.gz", "candidate_pairs.tsv.gz", "validation.json",
                "official.log", "official_check_ids.log"}
    if set(manifest["files"]) != required:
        raise ValueError("Unexpected validator result file inventory")
    to_download = required - ({"candidate_pairs.tsv.gz"} if args.matching_only else set())
    for name in sorted(to_download):
        target = args.output / name
        download(prefix, name, target)
        metadata = manifest["files"][name]
        if target.stat().st_size != metadata["bytes"] or sha256(target) != metadata["sha256"]:
            raise ValueError(f"Downloaded result checksum mismatch: {name}")
        reserve(args.output)
    validation = json.loads((args.output / "validation.json").read_text())
    if validation["rows"] != 1732544 or validation["matching_sha256"] != manifest["matching_sha256"] or validation["candidate_sha256"] != manifest["candidate_sha256"]:
        raise ValueError("Validator metadata disagrees")
    if set(validation["validation"]) != {"official", "official_check_ids"} or any(
        item != {"exit_code": 0, "pass": True} for item in validation["validation"].values()
    ):
        raise ValueError("Official validator did not pass")
    for name in ("official.log", "official_check_ids.log"):
        log = (args.output / name).read_text()
        if "PASS" not in log:
            raise ValueError(f"Official validator PASS absent in {name}")
    if any(line.startswith("WARNING:") for line in (args.output / "official_check_ids.log").read_text().splitlines()):
        raise ValueError("Strict official validator emitted warnings")
    expand(args.output / "matching_results.tsv.gz", args.output / "matching_results.tsv", manifest["matching_sha256"])
    if not args.matching_only:
        expand(args.output / "candidate_pairs.tsv.gz", args.output / "candidate_pairs.tsv", manifest["candidate_sha256"])
    receipt = {"processing_job": job["ProcessingJobArn"], "rows": 1732544,
               "matching_sha256": manifest["matching_sha256"],
               "candidate_sha256": manifest["candidate_sha256"],
               "official_pass": True, "strict_id_check_pass": True,
               "matching_only": args.matching_only}
    (args.output / "READY.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
