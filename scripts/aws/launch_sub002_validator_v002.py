"""Launch bounded SUB-002 postprocess and official validator over saved scores.

Eight disjoint worker parent prefixes mount directly as Processing inputs.
The same model inference scores are reused; no duplicate inference is done.
"""

from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path

from cost_guard import guard
from provision_worker import aws


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/aws/P5-SUB002-SM-VALIDATE-002.json"
LEDGERS = ROOT / "artifacts/cloud/phase5"
BUCKET = "aml2026-ber-08be19ac500747"
ROLE = "aml2026-phase5-sagemaker-validation"


def remote_json(key: str) -> dict:
    import subprocess
    command = ["aws", "--profile", "amamzon_01_a1_0", "--region", "us-east-1", "--no-cli-pager",
               "s3", "cp", f"s3://{BUCKET}/{key}", "-", "--only-show-errors"]
    return json.loads(subprocess.run(command, capture_output=True, text=True, check=True).stdout)


def verify_workers(config: dict) -> dict:
    states = aws("ec2", "describe-instances", "--filters", "Name=tag:Project,Values=aml2026-phase5")
    state_by_id = {item["InstanceId"]: item["State"]["Name"]
                   for group in states["Reservations"] for item in group["Instances"]}
    seen: set[str] = set()
    total = 0
    score_seen: set[str] = set()
    report = []
    for item in config["workers"]:
        run = item["run_id"]
        ledger = json.loads((LEDGERS / run / "ledger.json").read_text())
        if ledger.get("exit_code") != 0 or (state_by_id.get(ledger["instance_id"]) != "terminated" and ledger.get("instance_state") != "terminated"):
            raise ValueError(f"Worker not successful and terminated: {run}")
        prefix = f"amazon-ml-2026/phase5/runs/{run}/shards/"
        page = aws("s3api", "list-objects-v2", "--bucket", BUCKET, "--prefix", prefix)
        if page.get("IsTruncated"):
            raise ValueError(f"Unexpected paginated worker shard list: {run}")
        names = {entry["Key"].removeprefix(prefix) for entry in page.get("Contents", [])}
        expected = {f"{item['country']}-s{index:03d}-{kind}.tsv.gz"
                    for index in range(item["first_shard"], item["last_shard"])
                    for kind in ("matching", "candidates")}
        if names != expected or seen & names:
            raise ValueError(f"Incomplete or overlapping worker shards: {run}")
        seen.update(names)
        score_prefix = f"amazon-ml-2026/phase5/runs/{run}/scores/"
        score_page = aws("s3api", "list-objects-v2", "--bucket", BUCKET, "--prefix", score_prefix)
        if score_page.get("IsTruncated"):
            raise ValueError(f"Unexpected paginated score list: {run}")
        scores = {entry["Key"].removeprefix(score_prefix) for entry in score_page.get("Contents", [])}
        expected_scores = {f"{item['country']}-s{index:03d}-scores.tsv.gz"
                           for index in range(item["first_shard"], item["last_shard"])}
        if scores != expected_scores or score_seen & scores:
            raise ValueError(f"Incomplete or overlapping score shards: {run}")
        score_seen.update(scores)
        complete = remote_json(f"amazon-ml-2026/phase5/runs/{run}/results/COMPLETE.json")
        if complete.get("processed_query_count") != item["queries"]:
            raise ValueError(f"Wrong query count: {run}")
        total += item["queries"]
        report.append({"run_id": run, "shards": len(names), "queries": item["queries"]})
    if total != 1_732_544 or len(seen) != 384 or len(score_seen) != 192:
        raise ValueError("Incomplete full test coverage")
    return {"queries": total, "shards": len(seen), "score_shards": len(score_seen), "workers": report}


def processing_input(name: str, prefix: str) -> dict:
    return {"InputName": name, "S3Input": {"S3Uri": f"s3://{BUCKET}/{prefix}",
            "LocalPath": f"/opt/ml/processing/{name.lower()}", "S3DataType": "S3Prefix",
            "S3InputMode": "File", "S3DataDistributionType": "FullyReplicated"}}


def request(config: dict, role_arn: str) -> dict:
    inputs = [processing_input("Code", config["code_input_prefix"]),
              processing_input("Test", config["raw_test_prefix"])]
    inputs += [processing_input(f"Shards{index}", f"amazon-ml-2026/phase5/runs/{worker['run_id']}/")
               for index, worker in enumerate(config["workers"])]
    return {"ProcessingJobName": config["run_id"], "RoleArn": role_arn,
            "AppSpecification": {"ImageUri": config["image_uri"],
                                  "ContainerEntrypoint": ["python3", "/opt/ml/processing/code/driver.py"]},
            "ProcessingInputs": inputs,
            "ProcessingOutputConfig": {"Outputs": [{"OutputName": "Results", "S3Output": {
                "S3Uri": f"s3://{BUCKET}/{config['output_prefix']}",
                "LocalPath": "/opt/ml/processing/result", "S3UploadMode": "EndOfJob"}}]},
            "ProcessingResources": {"ClusterConfig": {"InstanceCount": 1,
                "InstanceType": config["instance_type"], "VolumeSizeInGB": config["volume_gib"]}},
            "StoppingCondition": {"MaxRuntimeInSeconds": config["max_runtime_seconds"]},
            "Tags": [{"Key": "Project", "Value": "aml2026-phase5"},
                     {"Key": "RunId", "Value": config["run_id"]},
                     {"Key": "Experiment", "Value": "SUB-002"}]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--launch", action="store_true")
    a = p.parse_args()
    config = json.loads(CONFIG.read_text())
    if config["run_id"] != "P5-SUB002-SM-VALIDATE-002" or len(config["workers"]) != 8:
        raise ValueError("Unexpected SUB-002 validator configuration")
    inventory = verify_workers(config)
    role = aws("iam", "get-role", "--role-name", ROLE)["Role"]
    if role["Arn"] != "arn:aws:iam::634393786318:role/" + ROLE:
        raise ValueError("Unexpected validator role")
    jobs = aws("sagemaker", "list-processing-jobs", "--max-results", "100")["ProcessingJobSummaries"]
    if any(job["ProcessingJobName"] == config["run_id"] for job in jobs):
        raise RuntimeError("This validator already ran")
    active = [job for job in jobs if job["ProcessingJobStatus"] in {"InProgress", "Stopping"}]
    if len(active) >= 2 or any(job["ProcessingJobName"] != "P5-SUB002-SM-VALIDATE-001" for job in active):
        raise RuntimeError("Other processing capacity is occupied")
    cap = dict(config, runtime_cap_minutes=config["max_runtime_seconds"] / 60)
    budget = guard(cap, str(config["price_usd_per_hour"]), str(config["noncompute_allowance_usd"]))
    if Decimal(budget["project_worst_case_usd"]) > Decimal("100"):
        raise RuntimeError("User $100 sprint ceiling would be exceeded")
    payload = request(config, role["Arn"])
    print(json.dumps({"inventory": inventory, "budget": budget, "request": payload}, indent=2), flush=True)
    if a.launch:
        response = aws("sagemaker", "create-processing-job", "--cli-input-json", json.dumps(payload))
        print(json.dumps({"created": response, "inventory": inventory, "budget": budget}), flush=True)


if __name__ == "__main__":
    main()
