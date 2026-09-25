"""Plot verified EXP-033 overall and India learning curves without extrapolation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SIZES = (20_000, 50_000, 100_000)


def read_curves(path: Path) -> tuple[list[float], list[float]]:
    report = json.loads(path.read_text())
    if report.get("experiment") != "EXP-033" or "15k new entity OOF" not in report.get("scope", ""):
        raise ValueError("Not the verified EXP-033 fixed-holdout report")
    if set(report.get("sizes", {})) != {str(size) for size in SIZES}:
        raise ValueError("Incomplete or unexpected learning-curve sizes")
    overall, india = [], []
    for size in SIZES:
        row = report["sizes"][str(size)]
        if row["overall"]["entities"] != 15_000:
            raise ValueError(f"Incomplete 15k evaluation at size {size}")
        if len(row.get("folds", [])) != 3:
            raise ValueError(f"Missing held-out fold at size {size}")
        overall.append(float(row["overall"]["macro_f0_5"]))
        india.append(float(row["by_country"]["India"]["macro_f0_5"]))
    if any(not 0 <= value <= 1 for value in overall + india):
        raise ValueError("Invalid macro F0.5 score")
    return overall, india


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.metrics.is_file():
        raise FileNotFoundError(args.metrics)
    if args.output.exists():
        raise FileExistsError(args.output)
    overall, india = read_curves(args.metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for ax, values, title in zip(axes, (overall, india), ("Overall macro F0.5", "India macro F0.5"), strict=True):
        ax.plot(SIZES, values, marker="o", linewidth=2.0, color="#1f5d8a")
        for size, value in zip(SIZES, values, strict=True):
            ax.annotate(f"{value:.5f}", (size, value), xytext=(0, 9),
                        textcoords="offset points", ha="center", fontsize=9)
        spread = max(values) - min(values)
        margin = max(0.002, spread * 0.25)
        ax.set_ylim(max(0, min(values) - margin), min(1, max(values) + margin))
        ax.set_xticks(SIZES, ["20k", "50k", "100k"])
        ax.set_xlabel("Training Source-1 entities per outer fold")
        ax.set_ylabel("Macro F0.5")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    fig.suptitle("EXP-033: fixed 15k held-out entities; Fold 4 closed", fontsize=12)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(json.dumps({"output": str(args.output), "sizes": SIZES,
                      "overall": overall, "india": india}))


if __name__ == "__main__":
    main()
