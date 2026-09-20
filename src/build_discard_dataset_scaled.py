"""Scaling experiment Part 2: builds a several-times-larger discard-
decision dataset from 10 years of logs (2009+2010-2018), used to answer
"does more training data make the real connectome (Arm A) model better?"
— a separate question from the A/B/C wiring-topology comparison already
completed in writeup.md, so this deliberately uses a separate filename
(*_scaled) rather than overwriting the original 3000-file/1.55M-decision
dataset that experiment_results.csv is based on.

Each year uses the same MAX_FILES_PER_YEAR (3000, same as the original
single-year run), with all 10 years combined so every year has even
representation across train/val/test (avoiding a hidden meta-drift leak
where old eras are only in train and new eras only in test).

Run:
  1. uv run python src/download_paifu_years.py   (downloads 2010-2018; 2009 should already exist)
  2. uv run python src/build_discard_dataset_scaled.py
"""

from pathlib import Path

import polars as pl
from jansou.io.mjai import parse_mjai

from paifu_replay import iter_discard_decisions
from splits import assign_splits

YEARS = [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]
MAX_FILES_PER_YEAR = 3000
OUT_PATH = Path("data/processed/discard_dataset_scaled.parquet")


def paifu_dir(year: int) -> Path:
    return Path(f"data/raw/paifu/{year}")


def build_rows_for_year(year: int, files: list[Path], file_splits) -> list[dict]:
    rows: list[dict] = []
    for file_index, path in enumerate(files):
        match_id = path.stem
        split = file_splits[file_index]
        paifu = parse_mjai(path)

        for kyoku_index, round_log in enumerate(paifu.rounds):
            for dc in iter_discard_decisions(round_log, paifu.player_count):
                rows.append(
                    {
                        "match_id": match_id,
                        "year": year,
                        "split": split,
                        "kyoku_index": kyoku_index,
                        "event_index": dc.event_index,
                        "seat": dc.seat,
                        "round_wind": round_log.round_wind.value,
                        "dealer": round_log.dealer,
                        "honba": round_log.honba,
                        "kyotaku": round_log.riichi_sticks,
                        "scores_at_round_start": list(round_log.scores),
                        "hand_counts": dc.hand_counts,
                        "hand_red_counts": dc.hand_red_counts,
                        "n_melds": dc.n_melds,
                        "n_dora_indicators": dc.n_dora_indicators,
                        "discard_tile": dc.discard_tile,
                        "discard_is_red": dc.discard_is_red,
                        "is_riichi": dc.is_riichi,
                        "is_tsumogiri": dc.is_tsumogiri,
                    }
                )

        if (file_index + 1) % 500 == 0:
            print(f"  [{year}] Processed {file_index + 1}/{len(files)} files, {len(rows)} decisions so far")

    return rows


def main() -> None:
    all_rows: list[dict] = []

    for year in YEARS:
        d = paifu_dir(year)
        all_files = sorted(d.glob("*.mjson"))
        if not all_files:
            raise SystemExit(f"No downloaded logs found in {d}, run src/download_paifu_years.py first")
        files = all_files[:MAX_FILES_PER_YEAR]
        print(f"\n=== {year}: {d} has {len(all_files)} files total, using the first {len(files)} this run ===")

        file_splits = assign_splits(len(files))
        rows = build_rows_for_year(year, files, file_splits)
        print(f"{year}: {len(rows)} decisions, from {len(files)} files")
        all_rows.extend(rows)

    print(f"\n{len(all_rows)} discard decisions total, from {len(YEARS)} years")

    df = pl.DataFrame(all_rows)
    print("\n=== dataset schema ===")
    print(df.schema)

    print("\n=== decision count by year ===")
    print(df.group_by("year").agg(pl.len().alias("count")).sort("year"))

    print("\n=== decision count by split ===")
    print(df.group_by("split").agg(pl.len().alias("count")))

    print("\n=== hand size sanity check (hand_size == 14 - 3*n_melds) ===")
    df_check = df.with_columns(
        hand_size=pl.col("hand_counts").list.sum(),
        expected=14 - 3 * pl.col("n_melds"),
    )
    bad = df_check.filter(pl.col("hand_size") != pl.col("expected"))
    if bad.height:
        raise AssertionError(f"{bad.height} rows have hand_size != 14 - 3*n_melds, the replay logic has a bug")
    print("All consistent: hand_size == 14 - 3*n_melds")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT_PATH)
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
