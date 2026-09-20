"""Phase 1 Part 1：訓練一次 Arm A（真 connectome mask），存低 checkpoint
俾之後嘅本機 UI 用嚟做 inference。

用返 features_scaled.npz（2009-2018 十年，~1550 萬個決策）嚟訓練，唔再用
原本嗰 3000 檔/155 萬決策——train_scaling_experiment.py 已經證實用大幾倍
嘅 dataset 可以將 test accuracy 由 65.7% 谷到 69.7%（+4pp，Welch p≈0，
Cohen's d=36.8，見 data/doc/writeup.md 嘅 scaling 實驗一節），冇理由部署
用緊細嗰份。跑法/config（seed=0、epoch/batch/lr）維持同
run_experiment.py 一致，方便對照。

輸出 data/processed/model_arm_a.pt，包含：
  - model_state: model 嘅 state_dict
  - n_channels, n_tiles: input shape
  - pn_ids, kc_ids, mbon_ids: 三層嘅 body id（同訓練用嗰個 mask 對應）

跑法： uv run python src/train_and_save_model.py
"""

import numpy as np
import torch

from train import train_one_arm

FEATURES_PATH = "data/processed/features_scaled.npz"
MASKS_PATH = "data/processed/masks.npz"
OUT_PATH = "data/processed/model_arm_a.pt"

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

    result = train_one_arm("A", SEED, masks, data, device, MAX_EPOCHS, BATCH_SIZE, LR, patience=PATIENCE)
    print(f"\nbest_val_acc={result['best_val_acc']:.4f}  test_acc={result['test_acc']:.4f}")

    torch.save(
        {
            "model_state": result["model"].state_dict(),
            "n_channels": int(data["X_train"].shape[1]),
            "n_tiles": int(data["X_train"].shape[2]),
            "pn_ids": masks["pn_ids"],
            "kc_ids": masks["kc_ids"],
            "mbon_ids": masks["mbon_ids"],
            "test_acc": result["test_acc"],
        },
        OUT_PATH,
    )
    print(f"已存 {OUT_PATH}")


if __name__ == "__main__":
    main()
