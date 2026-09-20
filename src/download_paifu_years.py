"""Scaling experiment Part 1: downloads 9 more years (2010-2018) of Tenhou
houou-room MJAI logs, adding them to the already-downloaded 2009 year to
build a 10-year log library, so build_discard_dataset_scaled.py can build
a dataset several times larger (see that file's docstring for why it's
kept separate from the original 3000-file/1.55M-decision version).

Same source/approach as explore_paifu.py: NikkeTryHard/tenhou-to-mjai
release v2.0.0, one zip per year, downloaded and extracted to
data/raw/paifu/<year>/. Each year is independently idempotent (already
downloaded/extracted years are skipped), so an interrupted run picks up
from whichever year didn't finish.

Run: uv run python src/download_paifu_years.py
"""

import zipfile
from pathlib import Path
from urllib.request import urlretrieve

YEARS = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]
RELEASE_TAG = "v2.0.0"


def archive_url(year: int) -> str:
    return f"https://github.com/NikkeTryHard/tenhou-to-mjai/releases/download/{RELEASE_TAG}/{year}.zip"


def archive_path(year: int) -> Path:
    return Path(f"data/raw/paifu_archives/{year}.zip")


def extract_dir(year: int) -> Path:
    return Path(f"data/raw/paifu/{year}")


def download_one(year: int) -> None:
    path = archive_path(year)
    if path.exists():
        print(f"[{year}] {path} already exists, skipping download")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    url = archive_url(year)
    print(f"[{year}] Downloading {url} ...")
    urlretrieve(url, path)
    print(f"[{year}] Done, size={path.stat().st_size / 1e6:.1f}MB")


def extract_one(year: int) -> int:
    out_dir = extract_dir(year)
    if out_dir.exists() and any(out_dir.iterdir()):
        n_files = sum(1 for _ in out_dir.glob("*.mjson"))
        print(f"[{year}] {out_dir} already has {n_files} files, skipping extraction")
        return n_files
    print(f"[{year}] Extracting to {out_dir} ...")
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path(year)) as zf:
        zf.extractall(out_dir)
    n_files = sum(1 for _ in out_dir.glob("*.mjson"))
    print(f"[{year}] Done, {n_files} .mjson files total")
    return n_files


def main() -> None:
    total_files = 0
    for year in YEARS:
        download_one(year)
        total_files += extract_one(year)
    print(f"\nAll {len(YEARS)} years downloaded/extracted, {total_files} files total")


if __name__ == "__main__":
    main()
