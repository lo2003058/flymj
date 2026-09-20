"""Step 8: plots a comparison of the Arm A/B/C test accuracy distributions.

Run: uv run python src/plot_experiment.py
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import polars as pl
from matplotlib import pyplot as plt

RESULTS_PATH = "data/processed/experiment_results.csv"
OUT_PATH = "artifacts/arm_comparison.png"


def main() -> None:
    df = pl.read_csv(RESULTS_PATH)

    arms = ["A", "B", "C"]
    labels = ["A\n(real connectome)", "B\n(degree-matched random)", "C\n(dense, param-matched)"]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    data = [df.filter(pl.col("arm") == arm)["test_acc"].to_numpy() * 100 for arm in arms]

    fig, ax = plt.subplots(figsize=(7, 5))

    bp = ax.boxplot(data, tick_labels=labels, showmeans=True, widths=0.5, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.3)

    rng = np.random.default_rng(0)
    for i, (values, color) in enumerate(zip(data, colors), start=1):
        jitter = rng.uniform(-0.08, 0.08, size=len(values))
        ax.scatter(np.full(len(values), i) + jitter, values, color=color, alpha=0.8, zorder=3, s=25)

    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Discard prediction accuracy by arm (each point = one seed)")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
