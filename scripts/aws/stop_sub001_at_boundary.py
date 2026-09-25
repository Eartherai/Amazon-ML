"""Stop one frozen country worker after a checksum-verified shard boundary.

The worker may already be computing the next shard when termination is sent;
that incomplete shard is discarded. Only completed S3 checkpoint pairs count.
"""
import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


BUCKET = "aml2026-ber-08be19ac500747"


def aws(*args: str) -> dict:
    command = ["aws", "--profile", os.environ.get("AWS_PROFILE", "amamzon_01_a1_0"),
               "--region", "us-east-1", "--no-cli-pager", *args, "--output", "json"]
    for attempt in range(6):
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            break
        retryable = any(text in result.stderr for text in
                        ("Rate exceeded", "Throttling", "TooManyRequests", "RequestTimeout", "Service Unavailable"))
        if not retryable or attempt == 5:
            raise RuntimeError(f"AWS boundary monitor call failed: {result.stderr}")
        time.sleep(min(2 ** (attempt + 1), 30))
    return json.loads(result.stdout) if result.stdout.strip() else {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stop-before", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--deadline-hours", type=float, default=4)
    args = parser.parse_args()
    if args.output.exists() or args.poll_seconds < 10 or not 0 < args.deadline_hours <= 12:
        raise ValueError("Duplicate boundary monitor or invalid bounds")
    ledger = json.loads((Path("artifacts/cloud/phase5") / args.run_id / "ledger.json").read_text())
    if (ledger.get("job_kind") != "sub001_country" or ledger.get("account_suffix") != "6318"
            or not ledger.get("first_shard", 0) < args.stop_before < ledger.get("last_shard", 64)):
        raise ValueError("Unexpected frozen worker ownership")
    country = ledger["country"]
    prefix = ledger["s3_outputs"]
    required = {f"{prefix}/shards/{country}-s{shard:03d}-{kind}.tsv.gz"
                for shard in range(ledger.get("first_shard", 0), args.stop_before)
                for kind in ("candidates", "matching")}
    deadline = time.monotonic() + args.deadline_hours * 3600
    while time.monotonic() < deadline:
        page = aws("s3api", "list-objects-v2", "--no-paginate", "--bucket", BUCKET,
                   "--prefix", prefix + "/shards/" + country + "-s")
        if page.get("IsTruncated"):
            raise RuntimeError("Shard listing truncated")
        found = {row["Key"]: row["Size"] for row in page.get("Contents", [])}
        complete = all(found.get(key, 0) > 0 for key in required)
        print(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "run_id": args.run_id,
                          "complete_pairs": sum(found.get(key, 0) > 0 for key in required) // 2,
                          "required_pairs": args.stop_before - ledger.get("first_shard", 0)}), flush=True)
        if complete:
            record = aws("ec2", "describe-instances", "--instance-ids", ledger["instance_id"])["Reservations"][0]["Instances"][0]
            tags = {tag["Key"]: tag["Value"] for tag in record.get("Tags", [])}
            if (record["State"]["Name"] != "running" or tags.get("RunId") != args.run_id
                    or tags.get("Project") != "aml2026-phase5"):
                raise RuntimeError("Worker is not running with expected identity")
            checks = {}
            for kind in ("candidates", "matching"):
                key = f"{prefix}/shards/{country}-s{args.stop_before-1:03d}-{kind}.tsv.gz"
                head = aws("s3api", "head-object", "--bucket", BUCKET, "--key", key,
                           "--checksum-mode", "ENABLED")
                if not head.get("ChecksumSHA256") or head["ContentLength"] <= 0:
                    raise ValueError("Boundary shard lacks SHA256")
                checks[kind] = {"key": key, "sha256_base64": head["ChecksumSHA256"],
                                "bytes": head["ContentLength"]}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            report = {"time": datetime.now(timezone.utc).isoformat(), "run_id": args.run_id,
                      "instance_id": ledger["instance_id"], "country": country,
                      "first_shard": ledger.get("first_shard", 0), "stop_before": args.stop_before,
                      "complete_pairs": args.stop_before - ledger.get("first_shard", 0),
                      "boundary_checksums": checks, "termination_requested": False}
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            response = aws("ec2", "terminate-instances", "--instance-ids", ledger["instance_id"])
            report["termination_requested"] = True
            report["termination_response"] = response
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"boundary_stop_requested": ledger["instance_id"]}), flush=True)
            return
        time.sleep(args.poll_seconds)
    raise TimeoutError("Frozen worker did not reach boundary by deadline")


if __name__ == "__main__":
    main()
