"""Step 3 exploration: download one year of Tenhou houou-room MJAI logs,
parse with jansou, and eyeball-check the results.

Source: raw yearly MJAI zips distributed via GitHub Releases by
NikkeTryHard/tenhou-to-mjai (human-readable tile name strings, e.g. "7s",
"5pr" = red five, not an opaque integer encoding). 2009 was picked purely
because it's the smallest file (32MB) and downloads fast, to validate the
pipeline — this has nothing to do with whether that year's data
quality/volume is sufficient. A single year already has ~8000 matches, at
~500 discard decisions each, far exceeding the 1-2 million the project
ultimately wants.

Run: uv run python src/explore_paifu.py
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
        print(f"{ARCHIVE_PATH} already exists, skipping download")
        return
    print(f"Downloading {ARCHIVE_URL} ...")
    urlretrieve(ARCHIVE_URL, ARCHIVE_PATH)
    print(f"Done, size={ARCHIVE_PATH.stat().st_size / 1e6:.1f}MB")


def extract_archive() -> list[Path]:
    if EXTRACT_DIR.exists() and any(EXTRACT_DIR.iterdir()):
        print(f"{EXTRACT_DIR} already has content, skipping extraction")
    else:
        print(f"Extracting to {EXTRACT_DIR} ...")
        EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ARCHIVE_PATH) as zf:
            zf.extractall(EXTRACT_DIR)
    files = sorted(EXTRACT_DIR.glob("*.mjson"))
    print(f"Total {len(files)} .mjson files")
    return files


def main() -> None:
    download_archive()
    files = extract_archive()

    sample_files = files[:N_SAMPLE_FILES]
    print(f"\n=== Parsing the first {len(sample_files)} files with jansou, checking the schema ===")

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

        print(f"\n--- file {i}: {path.name} ---")
        print(f"player_count={paifu.player_count}  rounds={len(paifu.rounds)}  final_scores={paifu.final_scores}")
        print(f"draws={n_draws}  discards={n_discards}  riichi discards={n_riichi}  tsumogiri={n_tsumogiri}")

        total_rounds += len(paifu.rounds)
        total_draws += n_draws
        total_discards += n_discards
        total_riichi_discards += n_riichi
        total_tsumogiri += n_tsumogiri

    print(f"\n=== totals across {len(sample_files)} files ===")
    print(f"rounds: {total_rounds}")
    print(f"total discard decisions: {total_discards}")
    print(f"mean discards per round: {total_discards / total_rounds:.1f}")
    print(f"mean discards per file (one hanchan): {total_discards / len(sample_files):.1f}")

    print(f"\n=== first file, first round, eyeballing each event (first 12) ===")
    paifu = parse_mjai(sample_files[0])
    r0 = paifu.rounds[0]
    print(f"round_wind={r0.round_wind}  dealer={r0.dealer}  initial_dora={dump_mjai([r0.initial_dora])}")
    for seat in range(paifu.player_count):
        print(f"  seat {seat} starting hand: {dump_mjai(r0.hands[seat])}")
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
