"""Scaling 實驗 Part 5：同 build_discard_dataset_scaled.py 一樣嘅諗法，但套用
落 Phase 2 嘅 action dataset（叫牌/立直/自摸/防守，唔淨係掉牌）度——由
2009+2010-2018 共 10 年嘅牌譜，用已經驗證過 100% 準確嘅 replay + oracle
（見 validate_replay.py），砌一個大幾倍嘅 action decision dataset，等
game_app.py/app.py 實際用緊嘅 action_model_arm_a.pt 都可以享受到「多啲
data」嘅著數（train_scaling_experiment.py 已經證實過對純掉牌 model 嘅
提升係 +4pp，呢度用同一批已經落好嘅 10 年牌譜，對 action model 做返一次）。

跟 build_discard_dataset_scaled.py 一樣：每年攞同一個 MAX_FILES_PER_YEAR
（3000），10 年一齊夾埋落，PASS 決策嘅 subsample 喺全部年份夾埋之後先做
一次（同原本單一年版本一致嘅做法），避免各年 PASS 比例唔一。

跑法（假設 download_paifu_years.py 已經跑過，2010-2018 已經落好）：
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
    """每年獨立一個 parquet：19.6M 行嘅 raw decision 一次過砌成一個 Python
    list[dict] 再轉 polars 會爆記憶體（試過俾 OOM kill 咗），逐年轉/寫落
    disk 就冇問題，仲順便令個 script 可以斷咗續返（已經有嘅年份會跳過）。"""
    return Path(f"data/processed/_action_scaled_year_{year}.parquet")


def build_year(year: int) -> None:
    d = paifu_dir(year)
    all_files = sorted(d.glob("*.mjson"))
    if not all_files:
        raise SystemExit(f"{d} 搵唔到已落嘅牌譜，先跑 src/download_paifu_years.py")
    files = all_files[:MAX_FILES_PER_YEAR]
    print(f"\n=== {year} 年：{d} 總共 {len(all_files)} 個檔，呢次用頭 {len(files)} 個 ===")

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
            print(f"  [{year}] 已處理 {file_index + 1}/{len(files)} 個檔，累積 {len(year_rows)} 個決策")

    print(f"{year} 年：{len(year_rows)} 個決策，跳過 {year_skipped} 局，嚟自 {len(files)} 個檔")
    pl.DataFrame(year_rows).write_parquet(year_cache_path(year))
    print(f"已存 {year_cache_path(year)}")


def main() -> None:
    for year in YEARS:
        if year_cache_path(year).exists():
            print(f"[{year}] {year_cache_path(year)} 已經存在，跳過重新 build")
            continue
        build_year(year)

    print(f"\n全部 {len(YEARS)} 個年份就緒，開始夾埋做 subsample...")
    df = pl.concat([pl.read_parquet(year_cache_path(year)) for year in YEARS])
    print(f"總共 {df.height} 個決策，嚟自 {len(YEARS)} 個年份")

    print("\n=== action_type 分佈（subsample 之前）===")
    print(df.group_by("action_type").agg(pl.len().alias("count")).sort("count", descending=True))

    # 用 numpy 揀 index、一次過 filter，唔好似原本單一年版本咁分開 filter
    # 兩份再 concat 再 sort：19.6M 行嘅 nested-list 欄一路複製多幾份會爆
    # 記憶體（呢步已經試過俾 OOM kill 咗兩次）。冇再做最後嘅 sort，行嘅
    # 次序對 train/val/test 唔重要（DataLoader 自己會再 shuffle）。
    is_pass = (df["action_type"] == "PASS").to_numpy()
    n_pass = int(is_pass.sum())
    n_keep_pass = int(n_pass * PASS_KEEP_RATE)
    rng = np.random.default_rng(PASS_SUBSAMPLE_SEED)
    keep_pass_indices = rng.choice(np.flatnonzero(is_pass), size=n_keep_pass, replace=False)
    keep_mask = ~is_pass
    keep_mask[keep_pass_indices] = True
    df = df.filter(pl.Series(keep_mask))

    print(f"\nPASS 由 {n_pass} 個 subsample 到 {n_keep_pass} 個（PASS_KEEP_RATE={PASS_KEEP_RATE}）")
    print(f"Subsample 之後總共 {df.height} 個決策")

    print("\n=== action_type 分佈（subsample 之後）===")
    print(df.group_by("action_type").agg(pl.len().alias("count")).sort("count", descending=True))

    print("\n=== 決策數按 split ===")
    print(df.group_by("split").agg(pl.len().alias("count")))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT_PATH)
    print(f"\n已存 {OUT_PATH}")


if __name__ == "__main__":
    main()
