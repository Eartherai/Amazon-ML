"""Render compact Phase5 development-evidence figures from saved measurements."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path("docs/figures")


def learning_curve() -> None:
    report = json.loads(Path("outputs/experiments/EXP-027-v002/metrics.json").read_text())
    x = [2000, 5000, 10000, 13333]
    means, lows, highs = [], [], []
    rng = np.random.default_rng(20260925)
    fig, ax = plt.subplots(figsize=(8.4, 4.7), constrained_layout=True)
    for fold in report["folds"]:
        ax.plot(x, [row["macro_f0_5"] for row in fold["sizes"]], marker="o", linewidth=1,
                alpha=0.34, label=f"Held fold {fold['held_fold']}")
    for j in range(4):
        values = np.concatenate([np.load(f"outputs/experiments/EXP-027-v002/held{fold['held_fold']}-size{fold['sizes'][j]['fit_entities']}-scores.npy")
                                 for fold in report["folds"]])
        means.append(values.mean())
        bootstrap = [rng.choice(values, len(values), replace=True).mean() for _ in range(1000)]
        lows.append(np.quantile(bootstrap, 0.025))
        highs.append(np.quantile(bootstrap, 0.975))
    ax.plot(x, means, color="#183b65", linewidth=2.4, marker="o", label="Entity-weighted mean")
    ax.fill_between(x, lows, highs, color="#4f85bd", alpha=0.2, label="95% entity bootstrap")
    ax.set(xlabel="Training S1 entities available to each outer fit", ylabel="Macro F0.5",
           title="More labeled entities help on fixed known-country heldouts")
    ax.set_xticks(x, ["2k", "5k", "10k", "~13.3k"])
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="lower right")
    fig.savefig(OUT / "phase5_learning_curve.png", dpi=180)
    plt.close(fig)


def error_slices() -> None:
    report = json.loads(Path("outputs/analysis/P5-NUMERIC-ERRORS-004/report.json").read_text())
    rows = report["country_source"]
    labels = [row["country"] + "/" + row["target_source"] for row in rows]
    retrieval = np.array([row["retrieval_misses"] / row["true_links"] for row in rows]) * 100
    matcher = np.array([row["matcher_misses"] / row["true_links"] for row in rows]) * 100
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    ax = axes[0]
    y = np.arange(len(rows))
    ax.barh(y, retrieval, color="#d3892f", label="Not in candidates")
    ax.barh(y, matcher, left=retrieval, color="#486da5", label="Retrieved, then rejected")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set(xlabel="Share of true links missed (%)", title="Missed-link rates by country and source")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.2)
    data = {row["category"]: row for row in report["dimensions"]["any_address_missing"]}
    labels = ["Address present", "Any address missing"]
    values = [data[False], data[True]]
    ax = axes[1]
    retrieval = np.array([row["retrieval_misses"] / row["true_links"] for row in values]) * 100
    matcher = np.array([row["matcher_misses"] / row["true_links"] for row in values]) * 100
    ax.barh(range(2), retrieval, color="#d3892f")
    ax.barh(range(2), matcher, left=retrieval, color="#486da5")
    ax.set_yticks(range(2), labels)
    ax.invert_yaxis()
    ax.set(xlabel="Share of true links missed (%)", title="Missing addresses sharply increase rejection")
    ax.grid(axis="x", alpha=0.2)
    fig.savefig(OUT / "phase5_error_slices.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    learning_curve()
    error_slices()
    print("Wrote Phase5 development-evidence figures")
