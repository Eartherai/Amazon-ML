"""Build the official ZIP from a fully validated immutable SUB-001 result."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code" / "business_entity_resolution"
FROZEN = ROOT / "artifacts" / "cloud" / "phase5" / "sub001-input-v002"
PREFIX = "code/business_entity_resolution/"
EXCLUDE = {"__pycache__", ".pytest_cache"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def file_map(validated: Path) -> dict[str, Path]:
    included = {
        "output/matching_results.tsv": validated / "matching_results.tsv",
        "output/candidate_pairs.tsv": validated / "candidate_pairs.tsv",
        "Documentation_template.md": ROOT / "Documentation_template.md",
        PREFIX + "README.md": CODE / "README.md",
        PREFIX + "MODEL_LICENSE.md": CODE / "MODEL_LICENSE.md",
        PREFIX + "requirements.txt": CODE / "requirements.txt",
        PREFIX + "utils/validate_submission.py": ROOT / "student_resource" / "utils" / "validate_submission.py",
        PREFIX + "src/submission/transliterate_probe.swift": ROOT / "scripts" / "transliterate_probe.swift",
        PREFIX + "configs/SUB-001.yaml": ROOT / "configs" / "submissions" / "SUB-001.yaml",
        PREFIX + "configs/BASELINE-P4-001.yaml": ROOT / "configs" / "baselines" / "BASELINE-P4-001.yaml",
    }
    for path in sorted((CODE / "src").rglob("*.py")):
        if not any(part in EXCLUDE for part in path.parts):
            included[PREFIX + path.relative_to(CODE).as_posix()] = path
    for path in sorted((CODE / "tests").rglob("*.py")):
        included[PREFIX + path.relative_to(CODE).as_posix()] = path
    for name in ("name_map.parquet", "name_char3_idf.npz", "address_char3_idf.npz",
                 "model.txt", "submission-config.json", "train-manifest.json", "manifest.json"):
        included[PREFIX + "artifacts/" + name] = FROZEN / name
    missing = [name for name, path in included.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing package files: {missing}")
    return included


def require_validated(directory: Path) -> dict:
    ready = json.loads((directory / "READY.json").read_text())
    validation = json.loads((directory / "validation.json").read_text())
    if ready.get("rows") != 1_732_544 or ready.get("official_pass") is not True or ready.get("strict_id_check_pass") is not True:
        raise ValueError("Submission is not marked ready by both official validators")
    if validation.get("rows") != ready["rows"]:
        raise ValueError("Validation row count mismatch")
    checks = validation.get("validation", {})
    if set(checks) != {"official", "official_check_ids"} or any(
        item != {"exit_code": 0, "pass": True} for item in checks.values()
    ):
        raise ValueError("Official validator result is incomplete")
    for key, name in (("matching_sha256", "matching_results.tsv"), ("candidate_sha256", "candidate_pairs.tsv")):
        if ready[key] != validation[key] or ready[key] != sha256(directory / name):
            raise ValueError(f"Validated checksum mismatch: {name}")
    if "PASS" not in (directory / "official.log").read_text():
        raise ValueError("Default official validator did not pass")
    strict_log = (directory / "official_check_ids.log").read_text()
    if "PASS" not in strict_log or any(line.startswith("WARNING:") for line in strict_log.splitlines()):
        raise ValueError("Strict official validator did not pass cleanly")
    return ready


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validated-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    ready = require_validated(args.validated_dir)
    files = file_map(args.validated_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.output.parent).free < 9 * 1024**3:
        raise RuntimeError("Insufficient disk reserve to create final ZIP")
    frozen_manifest = json.loads((FROZEN / "manifest.json").read_text())
    frozen_hashes = {item["name"]: item["sha256"] for item in frozen_manifest["files"]}
    for name, expected in frozen_hashes.items():
        if name in {"queries.parquet", "target2.parquet", "target3.parquet"}:
            continue  # Regenerated from the supplied official TSVs by prepare_inputs.py.
        if sha256(FROZEN / name) != expected:
            raise ValueError(f"Frozen artifact mismatch: {name}")
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "experiment": "SUB-001",
                "rows": ready["rows"], "matching_sha256": ready["matching_sha256"],
                "candidate_sha256": ready["candidate_sha256"], "model_sha256": frozen_hashes["model.txt"],
                "official_pass": True, "strict_id_check_pass": True,
                "processing_job": ready.get("processing_job"),
                "contents": sorted(files)}
    try:
        with zipfile.ZipFile(args.output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6,
                             allowZip64=True) as archive:
            for arcname, path in sorted(files.items()):
                archive.write(path, arcname=arcname)
            archive.writestr("PACKAGE_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
        with zipfile.ZipFile(args.output) as archive:
            if archive.testzip() is not None:
                raise ValueError("ZIP CRC verification failed")
            if set(archive.namelist()) != {*files, "PACKAGE_MANIFEST.json"}:
                raise ValueError("ZIP file inventory differs from manifest")
        package_hash = sha256(args.output)
        (args.output.parent / (args.output.name + ".sha256")).write_text(f"{package_hash}  {args.output.name}\n")
        print(json.dumps({"zip": str(args.output), "bytes": args.output.stat().st_size,
                          "sha256": package_hash, "matching": str(args.validated_dir / "matching_results.tsv"),
                          "matching_sha256": ready["matching_sha256"]}), flush=True)
    except Exception:
        args.output.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
