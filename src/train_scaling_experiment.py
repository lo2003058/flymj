"""Scaling experiment Part 4: trains only Arm A (real connectome) on the
several-times-larger dataset (features_scaled.npz, 10 years / ~15.5M
decisions), 5 seeds, and compares test accuracy against the 5 Arm A seeds
already in experiment_results.csv (2009, 3000 files / 1.55M decisions) —
answering "does more training data make this model better?", a question
independent of A vs B (wiring pattern).

Run: uv run python src/train_scaling_experiment.py
"""

import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from scipy import stats

from train import train_one_arm

FEATURES_PATH = "data/processed/features_scaled.npz"
MASKS_PATH = "data/processed/masks.npz"
BASELINE_RESULTS_PATH = Path("data/processed/experiment_results.csv")
RESULTS_PATH = Path("data/processed/scaling_results.csv")
HISTORY_PATH = Path("data/processed/scaling_history.parquet")

MAX_EPOCHS = 20
PATIENCE = 3
BATCH_SIZE = 1024
LR = 1e-3
N_SEEDS = 5
N_PERMUTATIONS = 100_000


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


def load_existing() -> tuple[list[dict], list[dict]]:
    results = pl.read_csv(RESULTS_PATH).to_dicts() if RESULTS_PATH.exists() else []
    history = pl.read_parquet(HISTORY_PATH).to_dicts() if HISTORY_PATH.exists() else []
    return results, history


def save_all(results: list[dict], history: list[dict]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(results).write_csv(RESULTS_PATH)
    pl.DataFrame(history).write_parquet(HISTORY_PATH)


def permutation_test(a: np.ndarray, b: np.ndarray, seed: int = 0) -> float:
    """Two-tailed permutation test: randomly reshuffles a and b's samples
    into new groups, checking how extreme the original mean difference is.
    Returns the p value."""
    rng = np.random.default_rng(seed)
    observed = abs(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    n_a = len(a)
    count = 0
    for _ in range(N_PERMUTATIONS):
        rng.shuffle(pooled)
        diff = abs(pooled[:n_a].mean() - pooled[n_a:].mean())
        if diff >= observed:
            count += 1
    return count / N_PERMUTATIONS


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    n_a, n_b = len(a), len(b)
    pooled_std = np.sqrt(((n_a - 1) * a.var(ddof=1) + (n_b - 1) * b.var(ddof=1)) / (n_a + n_b - 2))
    return (a.mean() - b.mean()) / pooled_std


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    data = load_data()
    masks = np.load(MASKS_PATH)
    print(f"train={len(data['y_train']):,}  val={len(data['y_val']):,}  test={len(data['y_test']):,}")

    results, history = load_existing()
    done = {r["seed"] for r in results}
    print(f"{N_SEEDS} runs total, {len(done)} already completed")

    for seed in range(N_SEEDS):
        if seed in done:
            print(f"Skipping already-completed: seed={seed}")
            continue

        print(f"\n=== Training arm=A (scaled dataset) seed={seed} ===")
        t0 = time.time()
        result = train_one_arm(
            "A", seed, masks, data, device, MAX_EPOCHS, BATCH_SIZE, LR, patience=PATIENCE, verbose_tag="A-scaled"
        )
        elapsed = time.time() - t0

        results.append(
            {
                "seed": seed,
                "n_params": result["n_params"],
                "best_epoch": result["best_epoch"],
                "best_val_acc": result["best_val_acc"],
                "test_acc": result["test_acc"],
                "elapsed_seconds": elapsed,
            }
        )
        for h in result["history"]:
            history.append({"seed": seed, **h})

        print(f"seed={seed}: test_acc={result['test_acc']:.4f}  took {elapsed:.1f}s")
        save_all(results, history)

    df = pl.DataFrame(results)
    scaled_acc = df["test_acc"].to_numpy()
    print("\n=== Arm A, scaled dataset (10 years, 5 seeds) ===")
    print(f"mean={scaled_acc.mean():.4f}  std={scaled_acc.std(ddof=1):.4f}  "
          f"min={scaled_acc.min():.4f}  max={scaled_acc.max():.4f}")

    if not BASELINE_RESULTS_PATH.exists():
        print(f"\n({BASELINE_RESULTS_PATH} not found, can't compare against the original Arm A)")
        return

    baseline_acc = (
        pl.read_csv(BASELINE_RESULTS_PATH).filter(pl.col("arm") == "A")["test_acc"].to_numpy()
    )
    print("\n=== Comparison: Arm A, original dataset (2009, 3000 files, 5 seeds) ===")
    print(f"mean={baseline_acc.mean():.4f}  std={baseline_acc.std(ddof=1):.4f}  "
          f"min={baseline_acc.min():.4f}  max={baseline_acc.max():.4f}")

    diff_pp = (scaled_acc.mean() - baseline_acc.mean()) * 100
    t_stat, p_ttest = stats.ttest_ind(scaled_acc, baseline_acc, equal_var=False)
    p_perm = permutation_test(scaled_acc, baseline_acc)
    d = cohens_d(scaled_acc, baseline_acc)

    print(f"\nDifference (scaled - original) = {diff_pp:+.2f} percentage points")
    print(f"Welch t = {t_stat:.2f}   p (t-test) = {p_ttest:.4f}   p (permutation, {N_PERMUTATIONS:,} draws) = {p_perm:.4f}")
    print(f"Cohen's d = {d:.2f}")

    print(f"\nSaved {RESULTS_PATH}")
    print(f"Saved {HISTORY_PATH}")


if __name__ == "__main__":
    main()
