"""Step C (task.md): 畫 in-degree distribution，比較真 mask 同 random mask。

因為 random mask 係逐個 node degree-matched 生成，理論上真同隨機嘅
in-degree 分佈應該完全重疊。如果冇重疊，代表 build_masks.py 有 bug。

Plot 入面嘅文字用英文，因為 matplotlib 預設字型冇 CJK glyph，用中文/廣東話
會變晒方格。Print 出嚟嘅 log 就照用廣東話。

跑法： uv run python src/plot_degrees.py
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib import pyplot as plt

MASKS_PATH = "data/processed/masks.npz"
OUT_PATH = "artifacts/degree_dist.png"


def plot_degree_overlay(ax, real_mask: np.ndarray, rand_masks: np.ndarray, title: str, xlabel: str) -> bool:
    """畫一個 subplot：真 mask vs random mask(seed 0) 嘅 in-degree histogram overlay。

    回傳 real 同 rand seed 0 嘅 degree sequence 係咪逐個 node 完全一致。
    """
    real_degree = real_mask.sum(axis=1)
    rand_degree_seed0 = rand_masks[0].sum(axis=1)

    max_degree = int(max(real_degree.max(), rand_degree_seed0.max()))
    if max_degree <= 30:
        # 度數細，每個整數一格，方便睇實際分佈形狀（例如 PN->KC）
        bins = np.arange(0, max_degree + 2) - 0.5
    else:
        # 度數大（例如 MBON 廣泛讀出成千 KC），逐個整數一格會變晒幼刺，改用固定 30 格
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
    print(f"已存 {OUT_PATH}")

    print("\n=== 逐個 seed 核對 degree 分佈係咪同真 mask 完全一致 ===")
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
            print(f"{name} seed {seed}: degree 分佈同真 mask 一致 = {match}")

    if not all_ok or not (ok_pn_kc and ok_kc_mbon):
        raise AssertionError("有 mask 嘅 degree 分佈同真 mask 對唔上，build_masks.py 大概有 bug")

    print("\n全部一致，degree-matched 冇問題。")


if __name__ == "__main__":
    main()
