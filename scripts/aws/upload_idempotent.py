"""Idempotent SHA256-verified upload with retries (skips objects already present with matching checksum)."""
import base64, hashlib, json, subprocess, sys, time
from pathlib import Path
def cli(*args):
    for i in range(6):
        r = subprocess.run(["aws", "--region", "us-east-1", "--no-cli-pager", *args, "--output", "json"], capture_output=True, text=True)
        if r.returncode == 0: return json.loads(r.stdout) if r.stdout.strip() else {}
        if "Not Found" in r.stderr or "404" in r.stderr: return None
        time.sleep(3 * (i + 1))
    raise RuntimeError(r.stderr[-300:])
root, bucket, prefix, receipt = Path(sys.argv[1]), sys.argv[2], sys.argv[3].rstrip("/"), sys.argv[4]
out = []
for p in sorted(x for x in root.rglob("*") if x.is_file()):
    key = prefix + "/" + str(p.relative_to(root)); d = hashlib.sha256(p.read_bytes()).hexdigest(); c = base64.b64encode(bytes.fromhex(d)).decode()
    h = cli("s3api", "head-object", "--bucket", bucket, "--key", key, "--checksum-mode", "ENABLED")
    if not h or h.get("ChecksumSHA256") != c:
        cli("s3api", "put-object", "--bucket", bucket, "--key", key, "--body", str(p), "--checksum-algorithm", "SHA256", "--checksum-sha256", c)
        h = cli("s3api", "head-object", "--bucket", bucket, "--key", key, "--checksum-mode", "ENABLED")
    if h.get("ChecksumSHA256") != c: raise RuntimeError("verify failed " + key)
    out.append({"key": key, "sha256": d, "bytes": p.stat().st_size, "version_id": h.get("VersionId")})
Path(receipt).write_text(json.dumps(out, indent=2)); print(len(out), "verified")
