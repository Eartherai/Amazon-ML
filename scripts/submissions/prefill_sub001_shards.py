"""Pre-upload completed immutable Mac shards for faster SUB-001 validation."""
import argparse
import base64
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.aws.upload_verified import cli, upload


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--country", choices=("France", "India"), default="France")
    parser.add_argument("--first-shard", type=int, default=0)
    parser.add_argument("--last-shard", type=int, default=64)
    args = parser.parse_args()
    if not 0 <= args.first_shard < args.last_shard <= 64:
        raise ValueError("Invalid shard interval")
    progress = json.loads((args.inference / "progress.json").read_text())
    france = next((row for row in progress["country_progress"] if row["country"] == "France"), None)
    if (france is None or france["queries"] != 259452 or france["shards"] != 64
            or progress["query_total"] != 1732544 or progress["fold4"] != "CLOSED"):
        raise ValueError("Mac France country is not complete and frozen")
    if args.country == "India":
        active = progress.get("active", {})
        india_done = next((row for row in progress["country_progress"] if row["country"] == "India"), None)
        if not (active.get("country") == "India" and active.get("shards", 0) >= args.last_shard) and not (
            india_done and india_done.get("shards") == 64 and india_done.get("queries") == 809986
        ):
            raise ValueError("Requested Mac India shards have not all completed")
    files = [args.inference / "shards" / f"{args.country}-s{number:03d}-{kind}.tsv.gz"
             for number in range(args.first_shard, args.last_shard)
             for kind in ("candidates", "matching")]
    if any(not path.is_file() for path in files):
        raise FileNotFoundError("Requested completed shard pair is absent")
    receipts = []
    for path in files:
        key = args.prefix.rstrip("/") + "/" + path.name
        digest = sha256(path)
        checksum = base64.b64encode(bytes.fromhex(digest)).decode()
        try:
            existing = cli("s3api", "head-object", "--bucket", args.bucket, "--key", key,
                           "--checksum-mode", "ENABLED")
        except Exception:
            existing = None
        if existing:
            if existing.get("ChecksumSHA256") != checksum or existing.get("ContentLength") != path.stat().st_size:
                raise ValueError(f"Existing S3 shard disagrees: {key}")
            receipt = {"key": key, "sha256": digest, "bytes": path.stat().st_size,
                       "version_id": existing.get("VersionId")}
        else:
            receipt = upload(path, args.bucket, key)
        receipts.append(receipt)
        args.receipt.write_text(json.dumps(receipts, indent=2) + "\n")
        print(json.dumps({"uploaded_or_verified": len(receipts), "total": len(files)}), flush=True)
    print(json.dumps({"prefill_complete": True, "country": args.country, "files": len(receipts)}))


if __name__ == "__main__":
    main()
