"""Step C: plot in-degree distributions, comparing the real mask
against the random mask.

Since the random mask is generated per-node degree-matched, the real and
random in-degree distributions should overlap exactly. If they don't,
that means build_masks.py has a bug.

Run: uv run python src/plot_degrees.py
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib import pyplot as plt

MASKS_PATH = "data/processed/masks.npz"
OUT_PATH = "artifacts/degree_dist.png"


def plot_degree_overlay(ax, real_mask: np.ndarray, rand_masks: np.ndarray, title: str, xlabel: str) -> bool:
    """Plot one subplot: an overlaid in-degree histogram of the real mask
    vs. the random mask (seed 0).

    Returns whether the real and rand-seed-0 degree sequences are exactly
    identical node-for-node.
    """
    real_degree = real_mask.sum(axis=1)
    rand_degree_seed0 = rand_masks[0].sum(axis=1)

    max_degree = int(max(real_degree.max(), rand_degree_seed0.max()))
    if max_degree <= 30:
        # Small degrees: one bin per integer, to see the actual distribution shape (e.g. PN->KC)
        bins = np.arange(0, max_degree + 2) - 0.5
    else:
        # Large degrees (e.g. MBON reads out broadly from thousands of KCs):
        # one bin per integer would be all noise, use 30 fixed bins instead
        bins = np.linspace(0, max_degree, 31)

    ax.hist(real_degree, bins=bins, alpha=0.5, label="real connectome mask", color="tab:blue")
    ax.hist(
        rand_degree_seed0,
        bins=bins,
        histtype="step",
        linewidth=2,
        label="degree-matched random mask (seed 0)",
        color="tab:red",
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("neuron count")
    ax.legend(fontsize=8)

    identical = np.array_equal(np.sort(real_degree), np.sort(rand_degree_seed0))
    ax.text(
        0.98,
        0.95,
        f"node-wise identical: {identical}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="green" if identical else "red",
    )
    return identical


def main() -> None:
    d = np.load(MASKS_PATH)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ok_pn_kc = plot_degree_overlay(
        axes[0],
        d["mask_pn_kc_real"],
        d["mask_pn_kc_rand"],
        "PN -> KC in-degree",
        "# PN inputs per KC",
    )
    ok_kc_mbon = plot_degree_overlay(
        axes[1],
        d["mask_kc_mbon_real"],
        d["mask_kc_mbon_rand"],
        "KC -> MBON in-degree",
        "# KC inputs per MBON",
    )

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Saved {OUT_PATH}")

    print("\n=== checking each seed's degree distribution exactly matches the real mask ===")
    all_ok = True
    for name, real_mask, rand_masks in [
        ("PN->KC", d["mask_pn_kc_real"], d["mask_pn_kc_rand"]),
        ("KC->MBON", d["mask_kc_mbon_real"], d["mask_kc_mbon_rand"]),
    ]:
        real_degree = np.sort(real_mask.sum(axis=1))
        for seed in range(rand_masks.shape[0]):
            rand_degree = np.sort(rand_masks[seed].sum(axis=1))
            match = np.array_equal(real_degree, rand_degree)
            all_ok = all_ok and match
            print(f"{name} seed {seed}: degree distribution matches real mask = {match}")

    if not all_ok or not (ok_pn_kc and ok_kc_mbon):
        raise AssertionError("Some mask's degree distribution doesn't match the real mask — build_masks.py probably has a bug")

    print("\nAll consistent, degree-matching is fine.")


if __name__ == "__main__":
    main()
