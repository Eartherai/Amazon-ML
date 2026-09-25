"""Launch the frozen SUB-001 validator only after complete verified S3 uploads.

Run from the project root. Without --launch this prints the planned request and
cost guard; it never creates a processing job.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path

from cost_guard import guard
from provision_worker import aws


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/aws/P5-SUB001-SM-VALIDATE-001.json"
ROLE = "aml2026-phase5-sagemaker-validation"
BUCKET = "aml2026-ber-08be19ac500747"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def verify_upload_receipts(receipt: Path, config: dict, *, head_objects: bool) -> dict:
    complete = receipt.with_suffix(".complete.json")
    final = json.loads(complete.read_text())
    if final != {"query_total": 1732544, "shards": 384, "receipts_sha256": digest(receipt)}:
        raise ValueError("Incomplete or changed shard-upload receipt")
    rows = json.loads(receipt.read_text())
    if len(rows) != 384:
        raise ValueError("Expected exactly 384 shard-upload receipts")
    prefix = config["shard_input_prefix"]
    keys = {item["key"] for item in rows}
    if len(keys) != 384 or any(not key.startswith(prefix) for key in keys):
        raise ValueError("Unexpected or duplicate S3 shard keys")
    names = [key.removeprefix(prefix) for key in keys]
    matching = {name.removesuffix("-matching.tsv.gz") for name in names if name.endswith("-matching.tsv.gz")}
    candidates = {name.removesuffix("-candidates.tsv.gz") for name in names if name.endswith("-candidates.tsv.gz")}
    if len(matching) != 192 or candidates != matching:
        raise ValueError("Matching and candidate shard inventories disagree")
    for item in rows:
        if item["bytes"] <= 0 or len(item["sha256"]) != 64:
            raise ValueError("Invalid shard receipt entry")
        if head_objects:
            remote = aws("s3api", "head-object", "--bucket", BUCKET, "--key", item["key"], "--checksum-mode", "ENABLED")
            checksum = base64.b64encode(bytes.fromhex(item["sha256"])).decode()
            if remote.get("ChecksumSHA256") != checksum or remote.get("ContentLength") != item["bytes"]:
                raise ValueError(f"S3 shard checksum changed: {item['key']}")
    return {"shards": len(rows), "pairs": len(matching), "receipts_sha256": final["receipts_sha256"]}


def request(config: dict, role_arn: str) -> dict:
    def processing_input(name: str, prefix: str) -> dict:
        return {"InputName": name, "S3Input": {
            "S3Uri": f"s3://{BUCKET}/{prefix}",
            "LocalPath": f"/opt/ml/processing/{name.lower()}",
            "S3DataType": "S3Prefix", "S3InputMode": "File",
            "S3DataDistributionType": "FullyReplicated",
        }}

    return {
        "ProcessingJobName": config["run_id"],
        "RoleArn": role_arn,
        "AppSpecification": {"ImageUri": config["image_uri"],
                             "ContainerEntrypoint": ["python3", "/opt/ml/processing/code/driver.py"]},
        "ProcessingInputs": [
            processing_input("Code", config["code_input_prefix"]),
            processing_input("Shards", config["shard_input_prefix"]),
            processing_input("Test", config["raw_test_prefix"]),
        ],
        "ProcessingOutputConfig": {"Outputs": [{"OutputName": "Results", "S3Output": {
            "S3Uri": f"s3://{BUCKET}/{config['output_prefix']}",
            "LocalPath": "/opt/ml/processing/result", "S3UploadMode": "EndOfJob"}}]},
        "ProcessingResources": {"ClusterConfig": {
            "InstanceCount": config["instance_count"],
            "InstanceType": config["instance_type"],
            "VolumeSizeInGB": config["volume_gib"],
        }},
        "StoppingCondition": {"MaxRuntimeInSeconds": config["max_runtime_seconds"]},
        "Tags": [{"Key": "Project", "Value": "aml2026-phase5"},
                 {"Key": "RunId", "Value": config["run_id"]},
                 {"Key": "Experiment", "Value": config["experiment"]}],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, default=ROOT / "artifacts/cloud/phase5/sub001-mac-shards-upload.json")
    parser.add_argument("--launch", action="store_true", help="Spend up to the configured $5 ceiling")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    if config["run_id"] != "P5-SUB001-SM-VALIDATE-001" or config["max_runtime_seconds"] != 21600:
        raise ValueError("Unexpected validator config")
    # The complete staging receipt is emitted only after the stager has fetched
    # and checked S3 SHA256/size for every one of the 384 immutable objects.
    # Repeating 384 serial HEAD calls here delays the deadline-critical job.
    upload = verify_upload_receipts(args.receipt, config, head_objects=False)
    role = aws("iam", "get-role", "--role-name", ROLE)["Role"]
    if role["Arn"] != "arn:aws:iam::634393786318:role/" + ROLE:
        raise ValueError("Unexpected IAM role")
    jobs = aws("sagemaker", "list-processing-jobs", "--max-results", "100")["ProcessingJobSummaries"]
    active = [job for job in jobs if job["ProcessingJobStatus"] in {"InProgress", "Stopping"}]
    if active or any(job["ProcessingJobName"] == config["run_id"] for job in jobs):
        raise RuntimeError("An active processing job or prior validator job exists; inspect before launching")
    cap_config = dict(config, runtime_cap_minutes=config["max_runtime_seconds"] / 60)
    budget = guard(cap_config, str(config["price_usd_per_hour"]), str(config["noncompute_allowance_usd"]))
    payload = request(config, role["Arn"])
    print(json.dumps({"upload": upload, "budget": budget, "request": payload}, indent=2), flush=True)
    if args.launch:
        response = aws("sagemaker", "create-processing-job", "--cli-input-json", json.dumps(payload))
        print(json.dumps({"created": response, "budget": budget, "upload": upload}), flush=True)


if __name__ == "__main__":
    main()
