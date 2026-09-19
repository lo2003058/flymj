"""Step 7 pilot：train Arm A(seed=0) 同 Arm B(seed=0) 一次，各自淨係少少
epoch，睇吓：
  1. 成條訓練 pipeline（DataLoader、mask、MPS）喺全量 150 萬個決策度行唔行得掂
  2. 準確度睇落合唔合理（唔係跌落 1/34≈2.9% 亂猜、都唔係離晒譜）
  3. 一個 run 要幾耐，等落實 20 個 B seed x 5 個 A seed 全面實驗嘅時間成本

呢步唔係最終結論，淨係決定使唔使跑落去。

跑法： uv run python src/pilot_compare.py
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
    print(f"\nArm A 有效參數(weight 非零 + bias): "
          f"{nnz_pn_kc}+{nnz_kc_mbon}+{masks['mask_pn_kc_real'].shape[0]}+{masks['mask_kc_mbon_real'].shape[0]}"
          f" = {nnz_pn_kc + nnz_kc_mbon + masks['mask_pn_kc_real'].shape[0] + masks['mask_kc_mbon_real'].shape[0]:,}")
    print(f"Arm C dense hidden dim（夾返呢個數）: H={h}")

    results = {}
    for arm in ["A", "B"]:
        print(f"\n=== 訓練 Arm {arm} (seed={SEED}) ===")
        t0 = time.time()
        result = train_one_arm(arm, SEED, masks, data, device, MAX_EPOCHS, BATCH_SIZE, LR, patience=PATIENCE)
        elapsed = time.time() - t0
        results[arm] = result
        print(f"Arm {arm}: n_params={result['n_params']:,}  best_val_acc={result['best_val_acc']:.4f} "
              f"(epoch {result['best_epoch'] + 1})  test_acc={result['test_acc']:.4f}  用時 {elapsed:.1f}s")

    print("\n=== Pilot 總結 ===")
    print(f"隨機亂猜嘅 baseline: 1/34 = {1 / 34:.4f}")
    for arm, r in results.items():
        print(f"Arm {arm}: n_params={r['n_params']:,}  best_val_acc={r['best_val_acc']:.4f}  test_acc={r['test_acc']:.4f}")


if __name__ == "__main__":
    main()
