"""Upload a completed SUB-001 Mac inference with per-object SHA256 verification."""
import argparse
import base64
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.aws.upload_verified import cli, upload


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inference", type=Path, required=True)
    p.add_argument("--bucket", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--receipt", type=Path, required=True)
    args = p.parse_args()
    report = json.loads((args.inference / "COMPLETE.json").read_text())
    if report["query_total"] != 1732544 or report["processed_query_count"] != report["query_total"]:
        raise ValueError("SUB-001 full test inference is incomplete")
    shards = args.inference / "shards"
    candidates = sorted(shards.glob("*-candidates.tsv.gz"))
    matching = sorted(shards.glob("*-matching.tsv.gz"))
    if len(candidates) != 192 or len(matching) != 192:
        raise ValueError(f"Expected 192 paired country/shard outputs, found {len(candidates)} and {len(matching)}")
    if {x.name.removesuffix("-candidates.tsv.gz") for x in candidates} != {
        x.name.removesuffix("-matching.tsv.gz") for x in matching
    }:
        raise ValueError("Candidate and matching shard keys differ")
    receipts = []
    for path in sorted(candidates + matching):
        key = args.prefix.rstrip("/") + "/" + path.name
        sha256 = digest(path)
        checksum = base64.b64encode(bytes.fromhex(sha256)).decode()
        try:
            existing = cli("s3api", "head-object", "--bucket", args.bucket, "--key", key, "--checksum-mode", "ENABLED")
        except Exception:
            existing = None
        if existing:
            if existing.get("ChecksumSHA256") != checksum or existing.get("ContentLength") != path.stat().st_size:
                raise ValueError(f"Existing S3 object differs: {key}")
            receipt = {"key": key, "sha256": sha256, "bytes": path.stat().st_size, "version_id": existing.get("VersionId")}
        else:
            receipt = upload(path, args.bucket, key)
        receipts.append(receipt)
        args.receipt.write_text(json.dumps(receipts, indent=2) + "\n")
        print(json.dumps({"uploaded_or_verified": len(receipts), "total": 384, "key": key}), flush=True)
    complete = args.receipt.with_suffix(".complete.json")
    complete.write_text(json.dumps({"query_total": report["query_total"], "shards": len(receipts), "receipts_sha256": digest(args.receipt)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
