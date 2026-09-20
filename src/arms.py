"""Mask construction for the three arms:
  A: real connectome mask
  B: degree-matched random mask (pick one of the 20 pre-built seeds in masks.npz)
  C: dense (no mask), hidden dim shrunk so "effective parameter count" matches Arm A

"Effective parameter count" = number of nonzero mask weights + biases
(masked-out weights get no gradient and are always 0, so they don't count
toward what the network actually uses).
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
    """Solve for H: the dense hidden layer's (n_pn -> H -> n_mbon) total
    parameter count (all weights nonzero + biases) should match Arm A's
    two masked layers' effective parameter count.
    """
    mask_pn_kc = masks_npz["mask_pn_kc_real"]
    mask_kc_mbon = masks_npz["mask_kc_mbon_real"]
    n_kc, n_pn = mask_pn_kc.shape
    n_mbon = mask_kc_mbon.shape[0]

    nnz_pn_kc = int(mask_pn_kc.sum())
    nnz_kc_mbon = int(mask_kc_mbon.sum())
    target = nnz_pn_kc + nnz_kc_mbon + n_kc + n_mbon  # weights (nonzero) + biases

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
    raise ValueError(f"Unknown arm: {arm!r}")
