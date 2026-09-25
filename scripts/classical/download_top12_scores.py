"""Download the 192 SUB-003 top-12 score shards and verify each against its S3 SHA256 checksum."""
import base64, hashlib, json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "outputs/experiments/CL-005"
BUCKET = "aml2026-ber-08be19ac500747"
RUNS = [f"P5-SUB003-T12-{t}-001" for t in ["FR0", "FR1"] + [f"IN{i}" for i in range(8)] + [f"US{i}" for i in range(5)]]
def aws(*a):
    import time
    for attempt in range(6):
        r = subprocess.run(["aws", "--profile", "amamzon_01_a1_0", "--region", "us-east-1", "--no-cli-pager", *a, "--output", "json"], capture_output=True, text=True)
        if r.returncode == 0:
            return json.loads(r.stdout) if r.stdout.strip() else {}
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"aws failed: {a} {r.stderr[-300:]}")
def listing(run):
    out, token = [], None
    while True:
        args = ["s3api", "list-objects-v2", "--bucket", BUCKET, "--prefix", f"amazon-ml-2026/phase5/runs/{run}/scores/"]
        if token: args += ["--continuation-token", token]
        page = aws(*args); out += [c["Key"] for c in page.get("Contents", [])]
        token = page.get("NextContinuationToken")
        if not token: return out
def fetch(key):
    name = key.rsplit("/", 1)[1]; dest = EXP / "test_scores" / name
    head = aws("s3api", "head-object", "--bucket", BUCKET, "--key", key, "--checksum-mode", "ENABLED")
    if not dest.exists():
        for attempt in range(6):
            if subprocess.run(["aws", "--profile", "amamzon_01_a1_0", "--region", "us-east-1", "s3", "cp", f"s3://{BUCKET}/{key}", str(dest) + ".part", "--only-show-errors"]).returncode == 0:
                Path(str(dest) + ".part").rename(dest)
                break
        else:
            raise RuntimeError(f"download failed {key}")
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    if base64.b64encode(bytes.fromhex(digest)).decode() != head.get("ChecksumSHA256"):
        raise ValueError(f"checksum mismatch {name}")
    return name, digest
(EXP / "test_scores").mkdir(parents=True, exist_ok=True)
keys = [k for r in RUNS for k in listing(r)]
print("objects", len(keys))
with ThreadPoolExecutor(8) as ex:
    rec = dict(ex.map(fetch, keys))
(EXP / "score_receipts.json").write_text(json.dumps(rec, indent=1))
print("verified", len(rec), "complete" if len(rec) == 192 else "INCOMPLETE")
