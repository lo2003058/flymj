"""Scaling 實驗 Part 1：落多 9 年（2010-2018）嘅天鳳鳳凰卓 MJAI 牌譜，撈埋
已經有嗰 2009 年，一齊夾成 10 年嘅牌譜庫，等 build_discard_dataset_scaled.py
可以砌一個大幾倍嘅 dataset（見嗰個檔嘅 docstring 解釋點解要獨立過原本嗰 3000
檔/155 萬決策嘅版本）。

跟 explore_paifu.py 一樣嘅嚟源/做法：NikkeTryHard/tenhou-to-mjai release
v2.0.0，逐年一個 zip，落完解壓去 data/raw/paifu/<year>/。每年獨立
idempotent（已經落/解過就跳過），中斷咗再跑一次會由未完成嗰年繼續。

跑法： uv run python src/download_paifu_years.py
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
        print(f"[{year}] {path} 已經存在，跳過下載")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    url = archive_url(year)
    print(f"[{year}] 落緊 {url} ...")
    urlretrieve(url, path)
    print(f"[{year}] 落完，size={path.stat().st_size / 1e6:.1f}MB")


def extract_one(year: int) -> int:
    out_dir = extract_dir(year)
    if out_dir.exists() and any(out_dir.iterdir()):
        n_files = sum(1 for _ in out_dir.glob("*.mjson"))
        print(f"[{year}] {out_dir} 已經有 {n_files} 個檔，跳過解壓")
        return n_files
    print(f"[{year}] 解緊壓去 {out_dir} ...")
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path(year)) as zf:
        zf.extractall(out_dir)
    n_files = sum(1 for _ in out_dir.glob("*.mjson"))
    print(f"[{year}] 解完，共 {n_files} 個 .mjson 檔")
    return n_files


def main() -> None:
    total_files = 0
    for year in YEARS:
        download_one(year)
        total_files += extract_one(year)
    print(f"\n全部 {len(YEARS)} 年落/解完，加埋 {total_files} 個檔")


if __name__ == "__main__":
    main()
