"""Step 7 pilot: train Arm A (seed=0) and Arm B (seed=0) once each, for a
few epochs, to check:
  1. Whether the whole training pipeline (DataLoader, mask, MPS) runs on
     the full 1.5 million decisions
  2. Whether accuracy looks reasonable (not stuck at random-guess
     1/34 ≈ 2.9%, and not wildly off)
  3. How long one run takes, to estimate the time cost of the full
     experiment (20 B seeds x 5 A seeds)

This step isn't the final conclusion — it just decides whether to proceed.

Run: uv run python src/pilot_compare.py
"""

import time

import numpy as np
import torch

from arms import dense_hidden_dim
from train import train_one_arm

FEATURES_PATH = "data/processed/features.npz"
MASKS_PATH = "data/processed/masks.npz"

MAX_EPOCHS = 20
PATIENCE = 3
BATCH_SIZE = 1024
LR = 1e-3
SEED = 0


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    feats = np.load(FEATURES_PATH)
    masks = np.load(MASKS_PATH)

    x_full = feats["X"]
    y_full = feats["y"]
    split_full = feats["split"]

    data = {
        "X_train": x_full[split_full == "train"],
        "y_train": y_full[split_full == "train"],
        "X_val": x_full[split_full == "val"],
        "y_val": y_full[split_full == "val"],
        "X_test": x_full[split_full == "test"],
        "y_test": y_full[split_full == "test"],
    }
    for k, v in data.items():
        print(f"{k}: {v.shape}")

    h = dense_hidden_dim(masks)
    nnz_pn_kc = int(masks["mask_pn_kc_real"].sum())
    nnz_kc_mbon = int(masks["mask_kc_mbon_real"].sum())
    print(f"\nArm A effective parameters (nonzero weights + biases): "
          f"{nnz_pn_kc}+{nnz_kc_mbon}+{masks['mask_pn_kc_real'].shape[0]}+{masks['mask_kc_mbon_real'].shape[0]}"
          f" = {nnz_pn_kc + nnz_kc_mbon + masks['mask_pn_kc_real'].shape[0] + masks['mask_kc_mbon_real'].shape[0]:,}")
    print(f"Arm C dense hidden dim (matching this count): H={h}")

    results = {}
    for arm in ["A", "B"]:
        print(f"\n=== Training Arm {arm} (seed={SEED}) ===")
        t0 = time.time()
        result = train_one_arm(arm, SEED, masks, data, device, MAX_EPOCHS, BATCH_SIZE, LR, patience=PATIENCE)
        elapsed = time.time() - t0
        results[arm] = result
        print(f"Arm {arm}: n_params={result['n_params']:,}  best_val_acc={result['best_val_acc']:.4f} "
              f"(epoch {result['best_epoch'] + 1})  test_acc={result['test_acc']:.4f}  took {elapsed:.1f}s")

    print("\n=== Pilot summary ===")
    print(f"Random-guess baseline: 1/34 = {1 / 34:.4f}")
    for arm, r in results.items():
        print(f"Arm {arm}: n_params={r['n_params']:,}  best_val_acc={r['best_val_acc']:.4f}  test_acc={r['test_acc']:.4f}")


if __name__ == "__main__":
    main()
