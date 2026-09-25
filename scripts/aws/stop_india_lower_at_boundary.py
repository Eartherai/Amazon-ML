"""Stop the lower SUB-001 India EC2 worker after shards 8-35 checkpoint.

The upper worker owns shards 36-63. This monitor only terminates the lower
worker after every lower pair and the upper worker's first pair exist in S3.
It does not touch the Mac fallback or either worker's immutable outputs.
"""
import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("artifacts/cloud/phase5")
LOWER_RUN = "P5-SUB001-INDIA-001"
UPPER_RUN = "P5-SUB001-INDIA-HIGH-001"
BUCKET = "aml2026-ber-08be19ac500747"


def aws(*args: str) -> dict:
    command = ["aws", "--profile", os.environ.get("AWS_PROFILE", "amamzon_01_a1_0"),
               "--region", "us-east-1", "--no-cli-pager", *args, "--output", "json"]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return json.loads(result.stdout) if result.stdout.strip() else {}


def expected_lower_keys(prefix: str) -> set[str]:
    return {f"{prefix}/shards/India-s{number:03d}-{kind}.tsv.gz"
            for number in range(8, 36) for kind in ("candidates", "matching")}


def keys(prefix: str) -> dict[str, int]:
    page = aws("s3api", "list-objects-v2", "--no-paginate", "--bucket", BUCKET,
               "--prefix", prefix + "/shards/India-s")
    if page.get("IsTruncated"):
        raise RuntimeError("Unexpected truncated country shard inventory")
    return {item["Key"]: item["Size"] for item in page.get("Contents", [])}


def instance(run_id: str, instance_id: str) -> dict:
    record = aws("ec2", "describe-instances", "--instance-ids", instance_id)["Reservations"][0]["Instances"][0]
    tags = {item["Key"]: item["Value"] for item in record.get("Tags", [])}
    if tags.get("RunId") != run_id or tags.get("Project") != "aml2026-phase5":
        raise ValueError(f"Wrong EC2 worker identity: {instance_id}")
    return {"id": instance_id, "state": record["State"]["Name"], "run_id": run_id}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=45)
    parser.add_argument("--deadline-hours", type=float, default=6)
    args = parser.parse_args()
    if args.output.exists() or args.poll_seconds < 15 or not 0 < args.deadline_hours <= 12:
        raise ValueError("Refusing duplicate monitor or invalid bounds")
    lower = json.loads((ROOT / LOWER_RUN / "ledger.json").read_text())
    upper = json.loads((ROOT / UPPER_RUN / "ledger.json").read_text())
    if lower["job_kind"] != "sub001_country" or upper["job_kind"] != "sub001_country":
        raise ValueError("Worker kinds are not the frozen country runner")
    if lower["country"] != upper["country"] != "India" or (lower["first_shard"], lower["last_shard"]) != (8, 64) or (upper["first_shard"], upper["last_shard"]) != (36, 64):
        raise ValueError("India shard ownership changed")
    if lower["input_prefix"] != upper["input_prefix"] or lower["account_suffix"] != upper["account_suffix"] != "6318":
        raise ValueError("Frozen input/account mismatch")
    lower_prefix = lower["s3_outputs"]
    upper_prefix = upper["s3_outputs"]
    required_lower = expected_lower_keys(lower_prefix)
    required_upper = {f"{upper_prefix}/shards/India-s036-{kind}.tsv.gz" for kind in ("candidates", "matching")}
    deadline = time.monotonic() + args.deadline_hours * 3600
    while time.monotonic() < deadline:
        found_lower = keys(lower_prefix)
        found_upper = keys(upper_prefix)
        ready = all(found_lower.get(key, 0) > 0 for key in required_lower) and all(
            found_upper.get(key, 0) > 0 for key in required_upper
        )
        print(json.dumps({"time": datetime.now(timezone.utc).isoformat(),
                          "lower_pairs": sum(found_lower.get(key, 0) > 0 for key in required_lower) // 2,
                          "lower_complete": ready, "upper_first_pair": all(found_upper.get(key, 0) > 0 for key in required_upper)}), flush=True)
        if ready:
            observed = {"lower": instance(LOWER_RUN, lower["instance_id"]),
                        "upper": instance(UPPER_RUN, upper["instance_id"])}
            upper_state = observed["upper"]["state"]
            if upper_state != "running":
                try:
                    aws("s3api", "head-object", "--bucket", BUCKET,
                        "--key", upper_prefix + "/results/COMPLETE.json")
                except subprocess.CalledProcessError as exc:
                    raise RuntimeError("Upper worker stopped without full completion") from exc
            if upper_state not in {"running", "shutting-down", "terminated"} or observed["lower"]["state"] != "running":
                raise RuntimeError(f"Unexpected worker state before boundary stop: {observed}")
            checked = {}
            for key in sorted({*required_upper, *(key for key in required_lower if "India-s035-" in key)}):
                head = aws("s3api", "head-object", "--bucket", BUCKET, "--key", key,
                           "--checksum-mode", "ENABLED")
                if not head.get("ChecksumSHA256") or head["ContentLength"] <= 0:
                    raise ValueError(f"Unverified boundary shard: {key}")
                checked[key] = {"sha256_base64": head["ChecksumSHA256"], "bytes": head["ContentLength"]}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            report = {"timestamp": datetime.now(timezone.utc).isoformat(), "lower": observed["lower"],
                      "upper": observed["upper"], "lower_complete_pairs": 28,
                      "upper_first_pair_verified": True, "boundary_shard_checksums": checked,
                      "termination_requested": False}
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            response = aws("ec2", "terminate-instances", "--instance-ids", lower["instance_id"])
            report["termination_requested"] = True
            report["termination_response"] = response
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"boundary_stop_requested": lower["instance_id"]}), flush=True)
            return
        time.sleep(args.poll_seconds)
    raise TimeoutError("Lower India worker did not reach verified shard35 before deadline; inspect workers")


if __name__ == "__main__":
    main()
