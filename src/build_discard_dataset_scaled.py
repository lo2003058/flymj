"""Scaling 實驗 Part 2：由 2009+2010-2018 共 10 年嘅牌譜砌一個大幾倍嘅掉牌
決策 dataset，用嚟答「多啲 training data 會唔會令真 connectome（Arm A）model
打得叻啲」——同 writeup.md 已經完成嘅 A/B/C 拓撲對照實驗係兩件唔同嘅事，所以
特登用獨立檔名（*_scaled），唔會覆蓋原本嗰 3000 檔/155 萬決策嘅 dataset，
同埋建基於佢嘅 experiment_results.csv 結果。

每年攞同一個 MAX_FILES_PER_YEAR（3000，同原本單一年嗰陣一樣），10 年一齊
夾埋落，等每一年喺 train/val/test 都有平均代表性（唔會出現「淨係用舊年代
train、新年代 test」呢種 meta drift 嘅隱藏 leakage）。

跑法：
  1. uv run python src/download_paifu_years.py   （落 2010-2018；2009 應該已經有）
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
            print(f"  [{year}] 已處理 {file_index + 1}/{len(files)} 個檔，累積 {len(rows)} 個決策")

    return rows


def main() -> None:
    all_rows: list[dict] = []

    for year in YEARS:
        d = paifu_dir(year)
        all_files = sorted(d.glob("*.mjson"))
        if not all_files:
            raise SystemExit(f"{d} 搵唔到已落嘅牌譜，先跑 src/download_paifu_years.py")
        files = all_files[:MAX_FILES_PER_YEAR]
        print(f"\n=== {year} 年：{d} 總共 {len(all_files)} 個檔，呢次用頭 {len(files)} 個 ===")

        file_splits = assign_splits(len(files))
        rows = build_rows_for_year(year, files, file_splits)
        print(f"{year} 年：{len(rows)} 個決策，嚟自 {len(files)} 個檔")
        all_rows.extend(rows)

    print(f"\n總共 {len(all_rows)} 個掉牌決策，嚟自 {len(YEARS)} 個年份")

    df = pl.DataFrame(all_rows)
    print("\n=== dataset schema ===")
    print(df.schema)

    print("\n=== decision 數按年份 ===")
    print(df.group_by("year").agg(pl.len().alias("count")).sort("year"))

    print("\n=== decision 數按 split ===")
    print(df.group_by("split").agg(pl.len().alias("count")))

    print("\n=== 手牌大細 sanity check（hand_size == 14 - 3*n_melds）===")
    df_check = df.with_columns(
        hand_size=pl.col("hand_counts").list.sum(),
        expected=14 - 3 * pl.col("n_melds"),
    )
    bad = df_check.filter(pl.col("hand_size") != pl.col("expected"))
    if bad.height:
        raise AssertionError(f"{bad.height} 行 hand_size != 14 - 3*n_melds，replay 邏輯有 bug")
    print("全部一致：hand_size == 14 - 3*n_melds")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT_PATH)
    print(f"\n已存 {OUT_PATH}")


if __name__ == "__main__":
    main()
