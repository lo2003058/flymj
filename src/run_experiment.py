"""Step 7: runs the full 3-arm x N-seed supervised training experiment.

  Arm A: seeds 0-4  (5 runs; there's only one real mask, so these 5 runs
         only vary model init/data order)
  Arm B: seeds 0-19 (20 runs; each seed uses the matching degree-matched
         random mask in masks.npz, and that same seed also drives model
         init/data order, so mask identity is the only variable that
         differs between A and B)
  Arm C: seeds 0-4  (5 runs; dense, hidden dim matched to Arm A's
         effective parameter count)

After every run, all results (including previously completed ones) are
immediately rewritten to csv/parquet, so an interruption never loses
already-completed runs (30 rows of data, so a full rewrite is cheap).
Re-running this script skips any (arm, seed) already in
experiment_results.csv.

Run: uv run python src/run_experiment.py
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
    print(f"{len(plan)} runs total, {len(done)} already completed")

    for arm, seed in plan:
        if (arm, seed) in done:
            print(f"Skipping already-completed: arm={arm} seed={seed}")
            continue

        print(f"\n=== Training arm={arm} seed={seed} ===")
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

        print(f"arm={arm} seed={seed}: test_acc={result['test_acc']:.4f}  took {elapsed:.1f}s")
        save_all(results, history)

    print("\n=== All runs complete, summary (mean by arm) ===")
    df = pl.DataFrame(results)
    print(df.group_by("arm").agg(
        pl.len().alias("n_runs"),
        pl.col("test_acc").mean().alias("test_acc_mean"),
        pl.col("test_acc").std().alias("test_acc_std"),
        pl.col("test_acc").min().alias("test_acc_min"),
        pl.col("test_acc").max().alias("test_acc_max"),
    ).sort("arm"))
    print(f"\nSaved {RESULTS_PATH}")
    print(f"Saved {HISTORY_PATH}")


if __name__ == "__main__":
    main()
