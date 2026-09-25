"""Freeze and checksum the standard-library-only SageMaker validator code."""
import hashlib
import json
import shutil
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = Path("artifacts/cloud/phase5/sub001-validation-code-v001")
    root.mkdir(parents=True, exist_ok=False)
    files = {
        "driver.py": Path("scripts/aws/sagemaker_validate_driver.py"),
        "merge_validate.py": Path("scripts/submissions/merge_validate.py"),
        "validate_submission.py": Path("student_resource/utils/validate_submission.py"),
    }
    for name, source in files.items():
        shutil.copy2(source, root / name)
    (root / "manifest.json").write_text(json.dumps({
        "scope": "SUB-001 validator only; official validator bytes preserved",
        "files": [{"name": name, "sha256": sha(root / name), "bytes": (root / name).stat().st_size}
                  for name in files],
    }, indent=2) + "\n")
    print(json.dumps({"files": len(files), "bytes": sum((root / name).stat().st_size for name in files)}))


if __name__ == "__main__":
    main()
