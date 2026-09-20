"""Scaling experiment Part 3: builds a (C x 34) feature tensor for the 10
years / ~15.5M decisions in build_discard_dataset_scaled.py — a separate
file from features.npz with the same schema, but not overwriting the
original (see build_discard_dataset_scaled.py's docstring).

Writes data/processed/features_scaled.npz.

Run: uv run python src/build_features_scaled.py
"""

import numpy as np
from jansou.io.mjai import parse_mjai

from build_discard_dataset_scaled import MAX_FILES_PER_YEAR, YEARS, paifu_dir
from features import N_CHANNELS, N_TILE_TYPES, encode
from paifu_replay import iter_full_discard_states
from splits import assign_splits

OUT_PATH = "data/processed/features_scaled.npz"


def main() -> None:
    x_list: list[np.ndarray] = []
    y_list: list[int] = []
    split_list: list[str] = []
    riichi_list: list[bool] = []
    tsumogiri_list: list[bool] = []

    for year in YEARS:
        all_files = sorted(paifu_dir(year).glob("*.mjson"))
        files = all_files[:MAX_FILES_PER_YEAR]
        print(f"\n=== {year}: processing {len(files)} files ===")
        file_splits = assign_splits(len(files))

        for file_index, path in enumerate(files):
            split = file_splits[file_index]
            paifu = parse_mjai(path)
            for round_log in paifu.rounds:
                for state in iter_full_discard_states(round_log, paifu.player_count):
                    x_list.append(encode(state, paifu.player_count))
                    y_list.append(state.discard_tile)
                    split_list.append(split)
                    riichi_list.append(state.is_riichi)
                    tsumogiri_list.append(state.is_tsumogiri)

            if (file_index + 1) % 500 == 0:
                print(f"  [{year}] Processed {file_index + 1}/{len(files)} files, {len(y_list)} samples so far")

    x = np.stack(x_list).astype(np.uint8)
    y = np.array(y_list, dtype=np.uint8)
    split_arr = np.array(split_list, dtype="<U5")
    is_riichi = np.array(riichi_list, dtype=bool)
    is_tsumogiri = np.array(tsumogiri_list, dtype=bool)

    print(f"\nX shape={x.shape} dtype={x.dtype}")
    print(f"y shape={y.shape}")
    assert x.shape[1:] == (N_CHANNELS, N_TILE_TYPES)

    for s in ("train", "val", "test"):
        print(f"split={s}: {(split_arr == s).sum():,}")

    np.savez_compressed(OUT_PATH, X=x, y=y, split=split_arr, is_riichi=is_riichi, is_tsumogiri=is_tsumogiri)
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
