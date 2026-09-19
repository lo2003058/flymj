"""Step 7：跑全套 3 arm x N seed 嘅 supervised training 實驗。

  Arm A: seed 0-4  （5 個 run，個 mask 淨係一條真嘅，5 個 run 淨係
         model init/data order 唔同）
  Arm B: seed 0-19 （20 個 run，每個 seed 用返 masks.npz 入面對應嗰條
         degree-matched random mask，同一個 seed 亦都攞嚟做 model
         init/data order，等 A/B 之間淨係 mask identity 呢個變數唔同）
  Arm C: seed 0-4  （5 個 run，dense、hidden dim 夾返 Arm A 有效參數量）

每個 run 訓練完即刻將全部（包括之前已經跑完嘅）結果重新存過 csv/parquet，
就算中途斷咗都唔會冚晒之前啲已經跑完嘅 run（30 行數據，全部 rewrite 都好平）。
再跑呢個 script 會跳過已經喺 experiment_results.csv 度嘅 (arm, seed)。

跑法： uv run python src/run_experiment.py
"""

import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

from train import train_one_arm

FEATURES_PATH = "data/processed/features.npz"
MASKS_PATH = "data/processed/masks.npz"
RESULTS_PATH = Path("data/processed/experiment_results.csv")
HISTORY_PATH = Path("data/processed/experiment_history.parquet")

MAX_EPOCHS = 20
PATIENCE = 3
BATCH_SIZE = 1024
LR = 1e-3

N_SEEDS_A = 5
N_SEEDS_B = 20
N_SEEDS_C = 5


def load_data() -> dict[str, np.ndarray]:
    feats = np.load(FEATURES_PATH)
    x_full = feats["X"]
    y_full = feats["y"]
    split_full = feats["split"]
    return {
        "X_train": x_full[split_full == "train"],
        "y_train": y_full[split_full == "train"],
        "X_val": x_full[split_full == "val"],
        "y_val": y_full[split_full == "val"],
        "X_test": x_full[split_full == "test"],
        "y_test": y_full[split_full == "test"],
    }


def run_plan() -> list[tuple[str, int]]:
    plan = [("A", s) for s in range(N_SEEDS_A)]
    plan += [("B", s) for s in range(N_SEEDS_B)]
    plan += [("C", s) for s in range(N_SEEDS_C)]
    return plan


def load_existing() -> tuple[list[dict], list[dict]]:
    results = pl.read_csv(RESULTS_PATH).to_dicts() if RESULTS_PATH.exists() else []
    history = pl.read_parquet(HISTORY_PATH).to_dicts() if HISTORY_PATH.exists() else []
    return results, history


def save_all(results: list[dict], history: list[dict]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(results).write_csv(RESULTS_PATH)
    pl.DataFrame(history).write_parquet(HISTORY_PATH)


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    data = load_data()
    masks = np.load(MASKS_PATH)

    plan = run_plan()
    results, history = load_existing()
    done = {(r["arm"], r["seed"]) for r in results}
    print(f"總共 {len(plan)} 個 run，已經跑完 {len(done)} 個")

    for arm, seed in plan:
        if (arm, seed) in done:
            print(f"跳過已跑完: arm={arm} seed={seed}")
            continue

        print(f"\n=== 訓練 arm={arm} seed={seed} ===")
        t0 = time.time()
        result = train_one_arm(arm, seed, masks, data, device, MAX_EPOCHS, BATCH_SIZE, LR, patience=PATIENCE)
        elapsed = time.time() - t0

        results.append(
            {
                "arm": arm,
                "seed": seed,
                "n_params": result["n_params"],
                "best_epoch": result["best_epoch"],
                "best_val_acc": result["best_val_acc"],
                "test_acc": result["test_acc"],
                "elapsed_seconds": elapsed,
            }
        )
        for h in result["history"]:
            history.append({"arm": arm, "seed": seed, **h})

        print(f"arm={arm} seed={seed}: test_acc={result['test_acc']:.4f}  用時 {elapsed:.1f}s")
        save_all(results, history)

    print("\n=== 全部 run 完成，總結（按 arm 平均）===")
    df = pl.DataFrame(results)
    print(df.group_by("arm").agg(
        pl.len().alias("n_runs"),
        pl.col("test_acc").mean().alias("test_acc_mean"),
        pl.col("test_acc").std().alias("test_acc_std"),
        pl.col("test_acc").min().alias("test_acc_min"),
        pl.col("test_acc").max().alias("test_acc_max"),
    ).sort("arm"))
    print(f"\n已存 {RESULTS_PATH}")
    print(f"已存 {HISTORY_PATH}")


if __name__ == "__main__":
    main()
