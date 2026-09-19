"""三條 arm 嘅 mask 建構：
  A: 真 connectome mask
  B: degree-matched random mask（masks.npz 入面 20 個 pre-built seed 揀一個）
  C: dense（冇 mask），hidden dim 縮到「有效參數量」夾返 Arm A

「有效參數量」= mask 嘅非零 weight 數 + bias 數（masked-out 嘅 weight 冇
gradient、永遠係 0，唔算落個網絡真正用緊嘅參數度）。
"""

import numpy as np
import torch


def real_masks(masks_npz) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(masks_npz["mask_pn_kc_real"], dtype=torch.float32),
        torch.tensor(masks_npz["mask_kc_mbon_real"], dtype=torch.float32),
    )


def random_masks(masks_npz, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(masks_npz["mask_pn_kc_rand"][seed], dtype=torch.float32),
        torch.tensor(masks_npz["mask_kc_mbon_rand"][seed], dtype=torch.float32),
    )


def dense_hidden_dim(masks_npz) -> int:
    """解 H：等 dense 中間層 (n_pn -> H -> n_mbon) 嘅總參數量
    （weight 全部非零 + bias）夾返 Arm A 嗰兩層 masked layer 嘅有效參數量。
    """
    mask_pn_kc = masks_npz["mask_pn_kc_real"]
    mask_kc_mbon = masks_npz["mask_kc_mbon_real"]
    n_kc, n_pn = mask_pn_kc.shape
    n_mbon = mask_kc_mbon.shape[0]

    nnz_pn_kc = int(mask_pn_kc.sum())
    nnz_kc_mbon = int(mask_kc_mbon.sum())
    target = nnz_pn_kc + nnz_kc_mbon + n_kc + n_mbon  # weight(非零) + bias

    # dense: n_pn*H + H(bias) + H*n_mbon + n_mbon(bias) == target
    h = round((target - n_mbon) / (n_pn + n_mbon + 1))
    return max(h, 1)


def dense_masks(masks_npz) -> tuple[torch.Tensor, torch.Tensor, int]:
    n_pn = masks_npz["mask_pn_kc_real"].shape[1]
    n_mbon = masks_npz["mask_kc_mbon_real"].shape[0]
    h = dense_hidden_dim(masks_npz)
    return torch.ones(h, n_pn), torch.ones(n_mbon, h), h


def build_arm_masks(arm: str, seed: int, masks_npz) -> tuple[torch.Tensor, torch.Tensor]:
    if arm == "A":
        return real_masks(masks_npz)
    if arm == "B":
        return random_masks(masks_npz, seed)
    if arm == "C":
        mask_pn_kc, mask_kc_mbon, _ = dense_masks(masks_npz)
        return mask_pn_kc, mask_kc_mbon
    raise ValueError(f"未知嘅 arm: {arm!r}")
