"""Step 6: builds the full discard-decision dataset from the already-
downloaded 2009 MJAI logs (see explore_paifu.py).

Uses only 2009 (already downloaded) — no further years — since all three
arms use the same data, so no era's meta can favor any particular arm, and
downloading more years just adds ToS risk and download time with no
experimental benefit. MAX_FILES is chosen to bring the total decision
count close to the original task's 1-2 million target (see explore_paifu.py: one
year has 6897 files, averaging ~515 decisions each).

Writes data/processed/discard_dataset.parquet, one row per discard decision:
  - match_id, year, kyoku_index, event_index, seat   (these four together
    let you look up the original file's full event stream again via
    jansou.io.mjai.parse_mjai())
  - split                                             train/val/test, split
    by match (not by row), see splits.py
  - round_wind, dealer, honba, kyotaku, scores_at_round_start
  - hand_counts, hand_red_counts, n_melds, n_dora_indicators
  - discard_tile, discard_is_red, is_riichi, is_tsumogiri   (these four are the labels)

Run: uv run python src/build_discard_dataset.py
"""

from pathlib import Path

import polars as pl
from jansou.io.mjai import parse_mjai

from paifu_replay import iter_discard_decisions
from splits import assign_splits

YEAR = 2009
PAIFU_DIR = Path(f"data/raw/paifu/{YEAR}")
OUT_PATH = Path("data/processed/discard_dataset.parquet")

MAX_FILES = 3000


def build_rows(files: list[Path], file_splits: list[str]) -> list[dict]:
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
                        "year": YEAR,
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

        if (file_index + 1) % 200 == 0:
            print(f"Processed {file_index + 1}/{len(files)} files, {len(rows)} decisions so far")

    return rows


def main() -> None:
    all_files = sorted(PAIFU_DIR.glob("*.mjson"))
    print(f"{PAIFU_DIR} has {len(all_files)} files total")
    if not all_files:
        raise SystemExit("No downloaded logs found, run src/explore_paifu.py to download data first")

    files = all_files[:MAX_FILES]
    print(f"Processing the first {len(files)} files this run (MAX_FILES={MAX_FILES})")

    file_splits = assign_splits(len(files))
    print(f"Split distribution (by match): "
          f"train={int((file_splits == 'train').sum())}  "
          f"val={int((file_splits == 'val').sum())}  "
          f"test={int((file_splits == 'test').sum())}")

    rows = build_rows(files, file_splits)
    print(f"\nTotal {len(rows)} discard decisions, from {len(files)} files")
    print(f"Mean per file (one hanchan): {len(rows) / len(files):.1f} decisions")

    df = pl.DataFrame(rows)
    print("\n=== dataset schema ===")
    print(df.schema)

    print("\n=== decision count by split ===")
    print(df.group_by("split").agg(pl.len().alias("count")))

    print("\n=== discard tile distribution (top 10) ===")
    print(df.group_by("discard_tile").agg(pl.len().alias("count")).sort("count", descending=True).head(10))

    print("\n=== riichi / tsumogiri rate ===")
    print(f"riichi discard rate: {df['is_riichi'].mean():.4f}")
    print(f"tsumogiri rate: {df['is_tsumogiri'].mean():.4f}")

    print("\n=== hand size sanity check ===")
    # Not simply 13/14: with each called meld, the correct relationship
    # for the remaining concealed hand is hand_size == 14 - 3*n_melds
    # (a meld call removes 2 of your own tiles, and the immediately
    # forced discard after removes one more, so -3 total, not the naive
    # -2 you might expect). This relationship was verified by manually
    # tracing the raw JSONL event-by-event, not just assumed upfront.
    df_check = df.with_columns(
        hand_size=pl.col("hand_counts").list.sum(),
        expected=14 - 3 * pl.col("n_melds"),
    )
    print(df_check.group_by(["n_melds", "hand_size"]).agg(pl.len().alias("count")).sort(["n_melds", "hand_size"]))
    bad = df_check.filter(pl.col("hand_size") != pl.col("expected"))
    if bad.height:
        raise AssertionError(f"{bad.height} rows have hand_size != 14 - 3*n_melds, the replay logic has a bug")
    print("All consistent: hand_size == 14 - 3*n_melds")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT_PATH)
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
