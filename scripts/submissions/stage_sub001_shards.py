"""Stage checksum-verified frozen SUB-001 shards for the official validator.

Completed EC2 checkpoints are copied within S3; completed Mac prefix shards are
already staged and checked against local SHA256. This resumes without replacing
any destination object and emits the validator's 384-file receipt only when all
partition owners have finished and terminated.
"""
import argparse
import base64
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


BUCKET = "aml2026-ber-08be19ac500747"
PREFIX = "amazon-ml-2026/phase5/sub001-mac-shards-v001/"
RUN_PREFIX = "amazon-ml-2026/phase5/runs/"
MODEL_SHA = "d84957742e05f5cd790d7dfc8c14ca05d3b5a2dc941a5094b8874d623b35117b"
MAC = Path("outputs/submissions/SUB-001/local-full-v005/inference")
RECEIPT = Path("artifacts/cloud/phase5/sub001-mac-shards-upload.json")
BOUNDARIES = {
    "P5-SUB001-INDIA-001": ("India", 8, 28),
    "P5-SUB001-INDIA-HIGH-001": ("India", 36, 52),
    "P5-SUB001-US-001": ("US", 0, 48),
}
TAILS = {
    "P5-SUB001-INDIA-TAIL-001": ("India", 28, 36, 100901),
    "P5-SUB001-INDIA-TAIL-002": ("India", 52, 58, 75500),
    "P5-SUB001-INDIA-TAIL-003": ("India", 58, 64, 76455),
    "P5-SUB001-US-TAIL-001": ("US", 48, 56, 83284),
    "P5-SUB001-US-TAIL-002": ("US", 56, 64, 82444),
}


def aws(*args: str, missing_ok: bool = False) -> dict | None:
    cmd = ["aws", "--profile", os.environ.get("AWS_PROFILE", "amamzon_01_a1_0"),
           "--region", "us-east-1", "--no-cli-pager", *args, "--output", "json"]
    for attempt in range(4):
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode and any(word in result.stderr for word in ("Rate exceeded", "Throttling", "TooManyRequests")) and attempt < 3:
            time.sleep(2 ** (attempt + 1))
            continue
        break
    if result.returncode:
        if missing_ok and ("404" in result.stderr or "NoSuchKey" in result.stderr or "Not Found" in result.stderr):
            return None
        raise RuntimeError(f"AWS command failed ({result.returncode}): {' '.join(cmd[:7])}: {result.stderr}")
    return json.loads(result.stdout) if result.stdout.strip() else {}


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def inventory(prefix: str) -> dict[str, dict]:
    response = aws("s3api", "list-objects-v2", "--no-paginate", "--bucket", BUCKET, "--prefix", prefix)
    if response.get("IsTruncated"):
        raise RuntimeError(f"Unexpected S3 inventory truncation: {prefix}")
    return {row["Key"]: row for row in response.get("Contents", [])}


def head(key: str) -> dict | None:
    return aws("s3api", "head-object", "--bucket", BUCKET, "--key", key,
               "--checksum-mode", "ENABLED", missing_ok=True)


def owner(country: str, shard: int) -> str | None:
    if country == "France" or (country == "India" and shard < 8):
        return None
    for run, (row_country, first, last) in BOUNDARIES.items():
        if country == row_country and first <= shard < last:
            return run
    for run, (row_country, first, last, _) in TAILS.items():
        if country == row_country and first <= shard < last:
            return run
    raise ValueError(f"No disjoint owner for {country} shard {shard}")


def require_head(key: str, record: dict, expected_hash: str | None = None) -> dict:
    checksum = record.get("ChecksumSHA256")
    if not checksum or record["ContentLength"] <= 0:
        raise ValueError(f"Missing SHA256 or bytes: {key}")
    digest = base64.b64decode(checksum, validate=True).hex()
    if expected_hash and digest != expected_hash:
        raise ValueError(f"S3 SHA256 disagrees with local file: {key}")
    return {"key": key, "sha256": digest, "bytes": record["ContentLength"],
            "version_id": record.get("VersionId")}


def write_receipt(rows: dict[str, dict]) -> None:
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    temporary = RECEIPT.with_suffix(".tmp")
    temporary.write_text(json.dumps([rows[key] for key in sorted(rows)], indent=2) + "\n")
    os.replace(temporary, RECEIPT)


def stage_one(name: str, run: str | None, existing: dict[str, dict],
              available: dict[str, dict]) -> dict | None:
    dest_key = PREFIX + name
    if run is None:
        local = MAC / "shards" / name
        if not local.is_file():
            return None
        dest = head(dest_key) if dest_key in existing else None
        if dest is None:
            raise RuntimeError(f"Pre-staged Mac shard missing in S3: {dest_key}")
        if dest["ContentLength"] != local.stat().st_size:
            raise ValueError(f"Mac shard size changed: {name}")
        return require_head(dest_key, dest, sha(local))
    source_key = f"{RUN_PREFIX}{run}/shards/{name}"
    if source_key not in available:
        return None
    source = head(source_key)
    if source is None:
        return None
    source_receipt = require_head(source_key, source)
    dest = head(dest_key) if dest_key in existing else None
    if dest is None:
        aws("s3api", "copy-object", "--bucket", BUCKET, "--key", dest_key,
            "--copy-source", BUCKET + "/" + source_key,
            "--copy-source-if-match", source["ETag"], "--checksum-algorithm", "SHA256")
        dest = head(dest_key)
    receipt = require_head(dest_key, dest, source_receipt["sha256"])
    if receipt["bytes"] != source_receipt["bytes"]:
        raise ValueError(f"Cloud shard size changed in S3 copy: {name}")
    return receipt


def finished_workers() -> bool:
    ledgers = {}
    for run in [*BOUNDARIES, *TAILS]:
        ledgers[run] = json.loads((Path("artifacts/cloud/phase5") / run / "ledger.json").read_text())
    ids = [ledger["instance_id"] for ledger in ledgers.values()]
    state = aws("ec2", "describe-instances", "--instance-ids", *ids)
    instances = {item["InstanceId"]: item for group in state["Reservations"] for item in group["Instances"]}
    if any(instances[id]["State"]["Name"] != "terminated" for id in ids):
        return False
    for run, (country, first, last) in BOUNDARIES.items():
        path = Path("artifacts/cloud/phase5") / run / f"boundary-stop-{last}.json"
        if not path.exists():
            raise ValueError(f"Worker terminated without verified boundary receipt: {run}")
        stop = json.loads(path.read_text())
        if (stop.get("termination_requested") is not True or stop.get("instance_id") != ledgers[run]["instance_id"]
                or stop.get("first_shard") != first or stop.get("stop_before") != last
                or stop.get("country") != country or stop.get("complete_pairs") != last - first):
            raise ValueError(f"Boundary provenance changed: {run}")
    for run, (country, first, last, count) in TAILS.items():
        key = f"{RUN_PREFIX}{run}/results/COMPLETE.json"
        metadata = head(key)
        if metadata is None:
            return False
        path = Path("artifacts/cloud/phase5") / run / "staged-COMPLETE.json"
        if not path.exists():
            aws("s3api", "get-object", "--bucket", BUCKET, "--key", key,
                "--checksum-mode", "ENABLED", str(path))
        if base64.b64encode(bytes.fromhex(sha(path))).decode() != metadata["ChecksumSHA256"]:
            raise ValueError(f"Worker COMPLETE checksum mismatch: {run}")
        report = json.loads(path.read_text())
        if (report.get("model_sha256") != MODEL_SHA or report.get("query_total") != 1732544
                or report.get("countries") != [country] or report.get("processed_query_count") != count
                or report.get("country_progress", [{}])[0].get("shards") != last-first):
            raise ValueError(f"Worker COMPLETE disagrees with frozen partition: {run}")
    mac = json.loads((MAC / "progress.json").read_text())
    if (mac.get("model_sha256") != MODEL_SHA or mac.get("country_progress", [{}])[0].get("country") != "France"
            or mac["country_progress"][0].get("shards") != 64):
        raise ValueError("Mac France provenance incomplete")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=40)
    parser.add_argument("--max-hours", type=float, default=3.5)
    args = parser.parse_args()
    if RECEIPT.with_suffix(".complete.json").exists():
        raise FileExistsError("Final staging receipt already exists")
    existing_rows = json.loads(RECEIPT.read_text()) if RECEIPT.exists() else []
    rows = {row["key"]: row for row in existing_rows}
    if len(rows) != len(existing_rows) or any(not key.startswith(PREFIX) for key in rows):
        raise ValueError("Invalid prior receipt inventory")
    deadline = time.monotonic() + args.max_hours * 3600
    last_count = -1
    while time.monotonic() < deadline:
        dest_inventory = inventory(PREFIX)
        source_inventories = {run: inventory(f"{RUN_PREFIX}{run}/shards/") for run in [*BOUNDARIES, *TAILS]}
        for country in ("France", "India", "US"):
            for shard in range(64):
                run = owner(country, shard)
                for kind in ("candidates", "matching"):
                    name = f"{country}-s{shard:03d}-{kind}.tsv.gz"
                    key = PREFIX + name
                    if key in rows:
                        continue
                    staged = stage_one(name, run, dest_inventory,
                                       source_inventories.get(run, {}))
                    if staged:
                        rows[key] = staged
                        write_receipt(rows)
        if len(rows) != last_count:
            print(json.dumps({"time": datetime.now(timezone.utc).isoformat(),
                              "staged_files": len(rows), "required_files": 384}), flush=True)
            last_count = len(rows)
        if len(rows) == 384 and finished_workers():
            keys = {PREFIX + f"{country}-s{shard:03d}-{kind}.tsv.gz"
                    for country in ("France", "India", "US") for shard in range(64)
                    for kind in ("candidates", "matching")}
            if set(rows) != keys:
                raise ValueError("Staged shard inventory is not the exact 384-file set")
            write_receipt(rows)
            complete = {"query_total": 1732544, "shards": 384, "receipts_sha256": sha(RECEIPT)}
            RECEIPT.with_suffix(".complete.json").write_text(json.dumps(complete, indent=2) + "\n")
            print(json.dumps({"complete": True, **complete}), flush=True)
            return
        if not args.watch:
            return
        time.sleep(args.poll_seconds)
    raise TimeoutError("SUB-001 staging did not complete before deadline; partial receipt preserved")


if __name__ == "__main__":
    main()
