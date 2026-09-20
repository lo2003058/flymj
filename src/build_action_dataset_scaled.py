"""Scaling experiment Part 5: the same idea as
build_discard_dataset_scaled.py, but applied to the Phase 2 action dataset
(calls/riichi/tsumo/defense, not just discards) — using 10 years of logs
(2009+2010-2018) and the already-100%-validated replay + oracle (see
validate_replay.py), builds a several-times-larger action decision
dataset so the action_model_arm_a.pt actually used by game_app.py/app.py
can enjoy the same "more data" benefit (train_scaling_experiment.py
already showed a +4pp gain for the discard-only model; this applies the
same already-downloaded 10 years of logs to the action model).

Like build_discard_dataset_scaled.py: each year uses the same
MAX_FILES_PER_YEAR (3000), all 10 years combined, with the PASS-decision
subsample done once after combining all years (matching the original
single-year approach), so no year's PASS ratio is skewed.

Run (assuming download_paifu_years.py already ran and 2010-2018 are downloaded):
  uv run python src/build_action_dataset_scaled.py
"""

from pathlib import Path

import numpy as np
import polars as pl
from jansou.io.mjai import parse_mjai

from build_action_dataset import PASS_KEEP_RATE, PASS_SUBSAMPLE_SEED, process_round
from splits import assign_splits

YEARS = [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]
MAX_FILES_PER_YEAR = 3000
OUT_PATH = Path("data/processed/action_dataset_scaled.parquet")


def paifu_dir(year: int) -> Path:
    return Path(f"data/raw/paifu/{year}")


def year_cache_path(year: int) -> Path:
    """A separate parquet per year: building all 19.6M raw decision rows
    into one giant Python list[dict] before converting to polars blows out
    memory (this hit an OOM kill once already). Converting/writing to disk
    year by year avoids that, and as a bonus lets the script resume after
    an interruption (already-built years are skipped)."""
    return Path(f"data/processed/_action_scaled_year_{year}.parquet")


def build_year(year: int) -> None:
    d = paifu_dir(year)
    all_files = sorted(d.glob("*.mjson"))
    if not all_files:
        raise SystemExit(f"No downloaded logs found in {d}, run src/download_paifu_years.py first")
    files = all_files[:MAX_FILES_PER_YEAR]
    print(f"\n=== {year}: {d} has {len(all_files)} files total, using the first {len(files)} this run ===")

    file_splits = assign_splits(len(files))
    year_rows: list[dict] = []
    year_skipped = 0

    for file_index, path in enumerate(files):
        match_id = path.stem
        split = file_splits[file_index]
        paifu = parse_mjai(path)
        for kyoku_index, round_log in enumerate(paifu.rounds):
            rows = process_round(round_log, paifu.player_count, paifu.rules, match_id, kyoku_index, split)
            if rows is None:
                year_skipped += 1
                continue
            year_rows.extend(rows)

        if (file_index + 1) % 500 == 0:
            print(f"  [{year}] Processed {file_index + 1}/{len(files)} files, {len(year_rows)} decisions so far")

    print(f"{year}: {len(year_rows)} decisions, {year_skipped} rounds skipped, from {len(files)} files")
    pl.DataFrame(year_rows).write_parquet(year_cache_path(year))
    print(f"Saved {year_cache_path(year)}")


def main() -> None:
    for year in YEARS:
        if year_cache_path(year).exists():
            print(f"[{year}] {year_cache_path(year)} already exists, skipping rebuild")
            continue
        build_year(year)

    print(f"\nAll {len(YEARS)} years ready, combining and subsampling...")
    df = pl.concat([pl.read_parquet(year_cache_path(year)) for year in YEARS])
    print(f"{df.height} decisions total, from {len(YEARS)} years")

    print("\n=== action_type distribution (before subsampling) ===")
    print(df.group_by("action_type").agg(pl.len().alias("count")).sort("count", descending=True))

    # Pick indices with numpy and do a single filter, instead of the
    # original single-year approach's separate filter-twice-then-concat-
    # then-sort: repeatedly copying 19.6M rows of nested-list columns
    # blows out memory (this step hit an OOM kill twice already). Also
    # skips the final sort — row order doesn't matter for train/val/test
    # (the DataLoader shuffles anyway).
    is_pass = (df["action_type"] == "PASS").to_numpy()
    n_pass = int(is_pass.sum())
    n_keep_pass = int(n_pass * PASS_KEEP_RATE)
    rng = np.random.default_rng(PASS_SUBSAMPLE_SEED)
    keep_pass_indices = rng.choice(np.flatnonzero(is_pass), size=n_keep_pass, replace=False)
    keep_mask = ~is_pass
    keep_mask[keep_pass_indices] = True
    df = df.filter(pl.Series(keep_mask))

    print(f"\nPASS subsampled from {n_pass} to {n_keep_pass} (PASS_KEEP_RATE={PASS_KEEP_RATE})")
    print(f"Total decisions after subsampling: {df.height}")

    print("\n=== action_type distribution (after subsampling) ===")
    print(df.group_by("action_type").agg(pl.len().alias("count")).sort("count", descending=True))

    print("\n=== decision count by split ===")
    print(df.group_by("split").agg(pl.len().alias("count")))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT_PATH)
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
