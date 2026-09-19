"""Step 4/6：由本機已經落好嘅 MJAI 牌譜（同 build_discard_dataset.py 用緊嗰批一樣）
砌 (C x 34) feature tensor，逐行對應 data/processed/discard_dataset.parquet。

輸出 data/processed/features.npz：
  X            (N, 32, 34) uint8   見 features.py 嘅 channel 設計
  y            (N,) uint8          discard_tile 0-33（label）
  split        (N,) <U5           train/val/test，同 discard_dataset.parquet 一致
  is_riichi    (N,) bool
  is_tsumogiri (N,) bool

跑法： uv run python src/build_features.py
"""

import numpy as np
from jansou.io.mjai import parse_mjai

from build_discard_dataset import MAX_FILES, PAIFU_DIR
from features import N_CHANNELS, N_TILE_TYPES, encode
from paifu_replay import iter_full_discard_states
from splits import assign_splits

OUT_PATH = "data/processed/features.npz"


def main() -> None:
    all_files = sorted(PAIFU_DIR.glob("*.mjson"))
    files = all_files[:MAX_FILES]
    print(f"處理 {len(files)} 個檔（MAX_FILES={MAX_FILES}，同 build_discard_dataset.py 一致）")

    file_splits = assign_splits(len(files))

    x_list: list[np.ndarray] = []
    y_list: list[int] = []
    split_list: list[str] = []
    riichi_list: list[bool] = []
    tsumogiri_list: list[bool] = []

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

        if (file_index + 1) % 200 == 0:
            print(f"已處理 {file_index + 1}/{len(files)} 個檔，累積 {len(y_list)} 個樣本")

    x = np.stack(x_list).astype(np.uint8)
    y = np.array(y_list, dtype=np.uint8)
    split_arr = np.array(split_list, dtype="<U5")
    is_riichi = np.array(riichi_list, dtype=bool)
    is_tsumogiri = np.array(tsumogiri_list, dtype=bool)

    print(f"\nX shape={x.shape} dtype={x.dtype}")
    print(f"y shape={y.shape}")
    assert x.shape[1:] == (N_CHANNELS, N_TILE_TYPES)

    print(f"\n=== 每個 channel 嘅平均 activation（sanity check，唔應該有成 0 或成 1 嘅 channel）===")
    mean_per_channel = x.mean(axis=(0, 2))
    for c, m in enumerate(mean_per_channel):
        print(f"  channel {c:2d}: {m:.4f}")

    for s in ("train", "val", "test"):
        print(f"split={s}: {(split_arr == s).sum():,}")

    np.savez_compressed(
        OUT_PATH, X=x, y=y, split=split_arr, is_riichi=is_riichi, is_tsumogiri=is_tsumogiri
    )
    print(f"\n已存 {OUT_PATH}")


if __name__ == "__main__":
    main()
