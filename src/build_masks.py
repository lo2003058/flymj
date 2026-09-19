"""Step B (task.md): 用 connectome edge weights 生成三組 mask。

輸出 data/processed/masks.npz，包含：
  - mask_pn_kc_real   : (n_kc, n_pn)      真 connectome mask
  - mask_kc_mbon_real : (n_mbon, n_kc)    真 connectome mask
  - mask_pn_kc_rand   : (20, n_kc, n_pn)   degree-matched random mask，seed 0-19
  - mask_kc_mbon_rand : (20, n_mbon, n_kc) degree-matched random mask，seed 0-19
  - pn_ids, kc_ids, mbon_ids              body id（int64），依 mask 嘅 row/col 次序排

Mask convention：跟 PyTorch nn.Linear.weight 嘅 (out_features, in_features) 慣例，
即係 row = 下游（post-synaptic）neuron，column = 上游（pre-synaptic）neuron。

跑法： uv run python src/build_masks.py
"""

import numpy as np
import polars as pl

from io_utils import load_feather
from labels import get_kc, get_mbon, get_pn

ANNOTATIONS_PATH = "data/raw/annotations.feather"
EDGES_PATH = "data/raw/edges.feather"
OUT_PATH = "data/processed/masks.npz"

MIN_WEIGHT = 5
N_SEEDS = 20


def load_relevant_edges(pn_ids: np.ndarray, kc_ids: np.ndarray, mbon_ids: np.ndarray) -> pl.DataFrame:
    """讀 edges.feather，只留低 weight >= MIN_WEIGHT 而且屬於
    PN->KC 或 KC->MBON 呢兩個 subgraph 嘅 edge。

    先試 pl.scan_ipc() 做 lazy filter（唔使成 1.1GB 檔案全部入 RAM）；
    如果撞到 annotations.feather 嗰種 dictionary-encoding bug，就 fallback
    去 load_feather()（pyarrow-based，食多啲 RAM 但一定 work）。
    """
    pn_kc_cond = pl.col("body_pre").is_in(pn_ids) & pl.col("body_post").is_in(kc_ids)
    kc_mbon_cond = pl.col("body_pre").is_in(kc_ids) & pl.col("body_post").is_in(mbon_ids)

    try:
        lf = pl.scan_ipc(EDGES_PATH)
        print("\n=== edges.feather schema (lazy scan) ===")
        print(lf.collect_schema())
        edges = (
            lf.filter(pl.col("weight") >= MIN_WEIGHT)
            .filter(pn_kc_cond | kc_mbon_cond)
            .select(["body_pre", "body_post"])
            .collect()
        )
    except pl.exceptions.ComputeError as e:
        print(f"\npl.scan_ipc() 撞到 ComputeError ({e})，fallback 去 pyarrow loader")
        full = load_feather(EDGES_PATH)
        print("\n=== edges.feather schema (pyarrow fallback) ===")
        print(full.schema)
        edges = full.filter(pl.col("weight") >= MIN_WEIGHT).filter(pn_kc_cond | kc_mbon_cond).select(
            ["body_pre", "body_post"]
        )

    return edges


def build_real_mask(
    edges: pl.DataFrame,
    pre_ids: np.ndarray,
    pre_index: dict[int, int],
    post_ids: np.ndarray,
    post_index: dict[int, int],
) -> np.ndarray:
    """將 (body_pre, body_post) edge list 轉做 binary mask，shape (n_post, n_pre)。"""
    n_pre = len(pre_ids)
    n_post = len(post_ids)
    mask = np.zeros((n_post, n_pre), dtype=np.float32)

    pre_arr = edges["body_pre"].to_numpy()
    post_arr = edges["body_post"].to_numpy()
    pre_idx = np.array([pre_index[b] for b in pre_arr], dtype=np.int64)
    post_idx = np.array([post_index[b] for b in post_arr], dtype=np.int64)
    mask[post_idx, pre_idx] = 1.0
    return mask


def build_random_masks(real_mask: np.ndarray, n_seeds: int) -> np.ndarray:
    """Degree-matched random mask：每個 post neuron 保留返真 mask 嗰個 in-degree k，
    喺 pre neuron pool 入面隨機（無放回）抽 k 個。每個 seed 獨立生成一份。
    """
    n_post, n_pre = real_mask.shape
    degrees = real_mask.sum(axis=1).astype(np.int64)

    rand_masks = np.zeros((n_seeds, n_post, n_pre), dtype=np.float32)
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        for i in range(n_post):
            k = int(degrees[i])
            if k == 0:
                continue
            chosen = rng.choice(n_pre, size=k, replace=False)
            rand_masks[seed, i, chosen] = 1.0
    return rand_masks


def main() -> None:
    pl.Config.set_tbl_rows(50)

    ann = load_feather(ANNOTATIONS_PATH)

    pn_ids = np.sort(get_pn(ann)["bodyId"].to_numpy())
    kc_ids = np.sort(get_kc(ann)["bodyId"].to_numpy())
    mbon_ids = np.sort(get_mbon(ann)["bodyId"].to_numpy())

    print("=== label 數量 (由 src/labels.py 嘅定義揀出，同 explore_labels.py 一致) ===")
    print(f"PN   : {len(pn_ids)}")
    print(f"KC   : {len(kc_ids)}")
    print(f"MBON : {len(mbon_ids)}")

    pn_index = {int(b): i for i, b in enumerate(pn_ids)}
    kc_index = {int(b): i for i, b in enumerate(kc_ids)}
    mbon_index = {int(b): i for i, b in enumerate(mbon_ids)}

    edges = load_relevant_edges(pn_ids, kc_ids, mbon_ids)
    print(f"\n=== filter 完 (weight >= {MIN_WEIGHT}) 剩低嘅相關 edge 數 ===")
    print(edges.height)

    pn_kc_edges = edges.filter(
        pl.col("body_pre").is_in(pn_ids) & pl.col("body_post").is_in(kc_ids)
    )
    kc_mbon_edges = edges.filter(
        pl.col("body_pre").is_in(kc_ids) & pl.col("body_post").is_in(mbon_ids)
    )
    print(f"PN -> KC   edge 數: {pn_kc_edges.height}")
    print(f"KC -> MBON edge 數: {kc_mbon_edges.height}")

    mask_pn_kc_real = build_real_mask(pn_kc_edges, pn_ids, pn_index, kc_ids, kc_index)
    mask_kc_mbon_real = build_real_mask(kc_mbon_edges, kc_ids, kc_index, mbon_ids, mbon_index)

    mask_pn_kc_rand = build_random_masks(mask_pn_kc_real, N_SEEDS)
    mask_kc_mbon_rand = build_random_masks(mask_kc_mbon_real, N_SEEDS)

    print("\n=== Mask shape / density ===")
    print(f"mask_pn_kc_real   : shape={mask_pn_kc_real.shape}, 非零元素={int(mask_pn_kc_real.sum())}, "
          f"density={mask_pn_kc_real.mean():.5f}")
    print(f"mask_kc_mbon_real : shape={mask_kc_mbon_real.shape}, 非零元素={int(mask_kc_mbon_real.sum())}, "
          f"density={mask_kc_mbon_real.mean():.5f}")
    print(f"mask_pn_kc_rand   : shape={mask_pn_kc_rand.shape} (seed 0-{N_SEEDS - 1})")
    print(f"mask_kc_mbon_rand : shape={mask_kc_mbon_rand.shape} (seed 0-{N_SEEDS - 1})")

    kc_in_degree = mask_pn_kc_real.sum(axis=1)
    print(f"\nKC 平均 in-degree (PN input): {kc_in_degree.mean():.2f} "
          f"(min={kc_in_degree.min():.0f}, max={kc_in_degree.max():.0f})")

    # sanity check：random mask 嘅 degree 分佈一定要同真 mask 一模一樣
    for seed in range(N_SEEDS):
        real_deg = mask_pn_kc_real.sum(axis=1)
        rand_deg = mask_pn_kc_rand[seed].sum(axis=1)
        assert np.array_equal(real_deg, rand_deg), f"seed {seed} 嘅 PN->KC degree 同真 mask 對唔上"
        real_deg2 = mask_kc_mbon_real.sum(axis=1)
        rand_deg2 = mask_kc_mbon_rand[seed].sum(axis=1)
        assert np.array_equal(real_deg2, rand_deg2), f"seed {seed} 嘅 KC->MBON degree 同真 mask 對唔上"
    print("\ndegree-matched 驗證通過：每個 seed 嘅 random mask 同真 mask 逐個 node in-degree 一致。")

    np.savez(
        OUT_PATH,
        mask_pn_kc_real=mask_pn_kc_real,
        mask_kc_mbon_real=mask_kc_mbon_real,
        mask_pn_kc_rand=mask_pn_kc_rand,
        mask_kc_mbon_rand=mask_kc_mbon_rand,
        pn_ids=pn_ids,
        kc_ids=kc_ids,
        mbon_ids=mbon_ids,
    )
    print(f"\n已存 {OUT_PATH}")


if __name__ == "__main__":
    main()
