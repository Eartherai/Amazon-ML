"""Download and SHA256-verify the complete frozen SUB-001 US cloud portion."""
import argparse
import base64
import hashlib
import json
import os
import subprocess
from pathlib import Path


BUCKET = "aml2026-ber-08be19ac500747"
PREFIX = "amazon-ml-2026/phase5/runs/P5-SUB001-US-001/"
MODEL_SHA256 = "d84957742e05f5cd790d7dfc8c14ca05d3b5a2dc941a5094b8874d623b35117b"


def aws(*args: str) -> dict:
    command = ["aws", "--profile", os.environ.get("AWS_PROFILE", "amamzon_01_a1_0"),
               "--region", "us-east-1", "--no-cli-pager", *args, "--output", "json"]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return json.loads(result.stdout) if result.stdout.strip() else {}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def verified_get(key: str, path: Path) -> dict:
    metadata = aws("s3api", "get-object", "--bucket", BUCKET, "--key", key,
                   "--checksum-mode", "ENABLED", str(path))
    checksum = metadata.get("ChecksumSHA256")
    local = digest(path)
    if not checksum or base64.b64encode(bytes.fromhex(local)).decode() != checksum:
        raise ValueError(f"S3 download checksum mismatch: {key}")
    if metadata.get("ContentLength") != path.stat().st_size:
        raise ValueError(f"S3 download byte length mismatch: {key}")
    return {"key": key, "sha256": local, "bytes": path.stat().st_size,
            "version_id": metadata.get("VersionId")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    complete_receipt = verified_get(PREFIX + "results/COMPLETE.json", args.output / "COMPLETE.json")
    report = json.loads((args.output / "COMPLETE.json").read_text())
    if (report["query_total"] != 1732544 or report["processed_query_count"] != 663106
            or report["countries"] != ["US"] or report["model_sha256"] != MODEL_SHA256
            or report["fold4"] != "CLOSED" or len(report["country_progress"]) != 1
            or report["country_progress"][0]["queries"] != 663106
            or report["country_progress"][0]["shards"] != 64):
        raise ValueError("US cloud job has not completed the frozen full country")
    shard_dir = args.output / "shards"
    shard_dir.mkdir()
    for number in range(64):
        base = f"US-s{number:03d}"
        receipts = []
        for kind in ("candidates", "matching"):
            name = f"{base}-{kind}.tsv.gz"
            receipts.append(verified_get(PREFIX + "shards/" + name, shard_dir / name))
        (shard_dir / f"{base}-receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")
        print(json.dumps({"verified_us_shard": number + 1, "total": 64}), flush=True)
    (args.output / "DOWNLOAD_READY.json").write_text(json.dumps({
        "query_count": 663106, "pairs": 64, "model_sha256": MODEL_SHA256,
        "complete_receipt": complete_receipt}, indent=2) + "\n")


if __name__ == "__main__":
    main()
