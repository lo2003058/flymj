"""Step 6：由本機已經落好嘅 2009 年 MJAI 牌譜（見 explore_paifu.py）砌全量
掉牌決策 dataset。

淨係用 2009 年（已經落咗），唔再落多年份——三條 arm 用緊同一批 data，
邊個年代嘅 meta 都唔會偏袒任何一條 arm，落多一年淨係加 ToS 風險同下載
時間，冇實驗上嘅著數。MAX_FILES 揀到令總決策數貼近 task.md 講嘅
100-200 萬（見 explore_paifu.py：一年 6897 個檔，平均每檔 ~515 個決策）。

輸出 data/processed/discard_dataset.parquet，一行一個掉牌決策：
  - match_id, year, kyoku_index, event_index, seat   （呢四樣夾埋可以返轉頭
    用 jansou.io.mjai.parse_mjai() 揾返原始檔案嘅完整 event stream）
  - split                                             train/val/test，按
    match 分（唔按行），見 splits.py
  - round_wind, dealer, honba, kyotaku, scores_at_round_start
  - hand_counts, hand_red_counts, n_melds, n_dora_indicators
  - discard_tile, discard_is_red, is_riichi, is_tsumogiri   （呢四樣係 label）

跑法： uv run python src/build_discard_dataset.py
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
            print(f"已處理 {file_index + 1}/{len(files)} 個檔，累積 {len(rows)} 個決策")

    return rows


def main() -> None:
    all_files = sorted(PAIFU_DIR.glob("*.mjson"))
    print(f"{PAIFU_DIR} 總共有 {len(all_files)} 個檔")
    if not all_files:
        raise SystemExit("搵唔到已落嘅牌譜，先跑 src/explore_paifu.py 落 data")

    files = all_files[:MAX_FILES]
    print(f"呢次處理頭 {len(files)} 個檔（MAX_FILES={MAX_FILES}）")

    file_splits = assign_splits(len(files))
    print(f"Split 分佈（按 match）: "
          f"train={int((file_splits == 'train').sum())}  "
          f"val={int((file_splits == 'val').sum())}  "
          f"test={int((file_splits == 'test').sum())}")

    rows = build_rows(files, file_splits)
    print(f"\n總共 {len(rows)} 個掉牌決策，嚟自 {len(files)} 個檔")
    print(f"平均每個檔（一場 hanchan）: {len(rows) / len(files):.1f} 個決策")

    df = pl.DataFrame(rows)
    print("\n=== dataset schema ===")
    print(df.schema)

    print("\n=== decision 數按 split ===")
    print(df.group_by("split").agg(pl.len().alias("count")))

    print("\n=== 掉牌 tile 分佈（頭 10）===")
    print(df.group_by("discard_tile").agg(pl.len().alias("count")).sort("count", descending=True).head(10))

    print("\n=== riichi / tsumogiri 比例 ===")
    print(f"riichi discard 比例: {df['is_riichi'].mean():.4f}")
    print(f"tsumogiri 比例: {df['is_tsumogiri'].mean():.4f}")

    print("\n=== 手牌大細 sanity check ===")
    # 唔係淨係 13/14：每 call 咗一舊 meld，concealed hand 淨落嚟嘅正確關係係
    # hand_size == 14 - 3*n_melds（call 咗嘅 meld 出 2 隻自己嘅牌 + 之後嗰下
    # 逼住即刻掉牌，兩樣加埋淨低 -3，唔係天真咁諗嘅 -2；用 raw JSONL 逐個
    # event 手動 trace 驗證過先落實呢條關係，唔係一開始就假設嘅）。
    df_check = df.with_columns(
        hand_size=pl.col("hand_counts").list.sum(),
        expected=14 - 3 * pl.col("n_melds"),
    )
    print(df_check.group_by(["n_melds", "hand_size"]).agg(pl.len().alias("count")).sort(["n_melds", "hand_size"]))
    bad = df_check.filter(pl.col("hand_size") != pl.col("expected"))
    if bad.height:
        raise AssertionError(f"{bad.height} 行 hand_size != 14 - 3*n_melds，replay 邏輯有 bug")
    print("全部一致：hand_size == 14 - 3*n_melds")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT_PATH)
    print(f"\n已存 {OUT_PATH}")


if __name__ == "__main__":
    main()
