"""Step 3 探索：落一年天鳳鳳凰卓 MJAI 牌譜，用 jansou parse，肉眼核對啱唔啱。

嚟源：NikkeTryHard/tenhou-to-mjai 喺 GitHub Releases 派發嘅原始 MJAI yearly
zip（人類可讀嘅牌名字串，例如 "7s"、"5pr"=紅五，唔係 opaque 整數編碼）。
揀 2009 年純粹因為佢檔案最細（32MB），落得快，用嚟驗證 pipeline；
同「呢年 data 質素/數量夠唔夠」冇關係 —— 單一年已經有 ~8000 局，
每局約 500 個掉牌決策，遠超成個 project 最終想要嘅 100-200 萬個。

跑法： uv run python src/explore_paifu.py
"""

import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from jansou.core.notation import dump_mjai
from jansou.io.mjai import parse_mjai
from jansou.io.paifu import Discard, Draw

YEAR = 2009
ARCHIVE_URL = f"https://github.com/NikkeTryHard/tenhou-to-mjai/releases/download/v2.0.0/{YEAR}.zip"
ARCHIVE_PATH = Path(f"data/raw/paifu_archives/{YEAR}.zip")
EXTRACT_DIR = Path(f"data/raw/paifu/{YEAR}")

N_SAMPLE_FILES = 5


def download_archive() -> None:
    if ARCHIVE_PATH.exists():
        print(f"{ARCHIVE_PATH} 已經存在，跳過下載")
        return
    print(f"落緊 {ARCHIVE_URL} ...")
    urlretrieve(ARCHIVE_URL, ARCHIVE_PATH)
    print(f"落完，size={ARCHIVE_PATH.stat().st_size / 1e6:.1f}MB")


def extract_archive() -> list[Path]:
    if EXTRACT_DIR.exists() and any(EXTRACT_DIR.iterdir()):
        print(f"{EXTRACT_DIR} 已經有嘢，跳過解壓")
    else:
        print(f"解緊壓去 {EXTRACT_DIR} ...")
        EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ARCHIVE_PATH) as zf:
            zf.extractall(EXTRACT_DIR)
    files = sorted(EXTRACT_DIR.glob("*.mjson"))
    print(f"總共 {len(files)} 個 .mjson 檔")
    return files


def main() -> None:
    download_archive()
    files = extract_archive()

    sample_files = files[:N_SAMPLE_FILES]
    print(f"\n=== 用 jansou parse 頭 {len(sample_files)} 個檔，核對 schema ===")

    total_rounds = 0
    total_draws = 0
    total_discards = 0
    total_riichi_discards = 0
    total_tsumogiri = 0

    for i, path in enumerate(sample_files):
        paifu = parse_mjai(path)
        n_discards = sum(1 for r in paifu.rounds for e in r.events if isinstance(e, Discard))
        n_draws = sum(1 for r in paifu.rounds for e in r.events if isinstance(e, Draw))
        n_riichi = sum(1 for r in paifu.rounds for e in r.events if isinstance(e, Discard) and e.riichi)
        n_tsumogiri = sum(1 for r in paifu.rounds for e in r.events if isinstance(e, Discard) and e.tsumogiri)

        print(f"\n--- 檔 {i}: {path.name} ---")
        print(f"player_count={paifu.player_count}  局數={len(paifu.rounds)}  final_scores={paifu.final_scores}")
        print(f"draw 次數={n_draws}  discard 次數={n_discards}  riichi discard={n_riichi}  tsumogiri={n_tsumogiri}")

        total_rounds += len(paifu.rounds)
        total_draws += n_draws
        total_discards += n_discards
        total_riichi_discards += n_riichi
        total_tsumogiri += n_tsumogiri

    print(f"\n=== {len(sample_files)} 個檔加埋 ===")
    print(f"局數: {total_rounds}")
    print(f"discard 決策總數: {total_discards}")
    print(f"平均每局 discard 數: {total_discards / total_rounds:.1f}")
    print(f"平均每個檔(一場 hanchan)嘅 discard 數: {total_discards / len(sample_files):.1f}")

    print(f"\n=== 第一個檔第一局，逐個 event 肉眼睇（頭 12 個）===")
    paifu = parse_mjai(sample_files[0])
    r0 = paifu.rounds[0]
    print(f"round_wind={r0.round_wind}  dealer={r0.dealer}  initial_dora={dump_mjai([r0.initial_dora])}")
    for seat in range(paifu.player_count):
        print(f"  seat {seat} 起手: {dump_mjai(r0.hands[seat])}")
    for e in r0.events[:12]:
        if isinstance(e, Draw):
            print(f"  Draw  seat={e.seat} tile={dump_mjai([e.tile])}")
        elif isinstance(e, Discard):
            flags = []
            if e.riichi:
                flags.append("riichi")
            if e.tsumogiri:
                flags.append("tsumogiri")
            flag_str = f" ({','.join(flags)})" if flags else ""
            print(f"  Discard seat={e.seat} tile={dump_mjai([e.tile])}{flag_str}")
        else:
            print(f"  {e}")


if __name__ == "__main__":
    main()
