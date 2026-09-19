"""每個 match（一個 .mjson 檔案 = 一場 hanchan）分去 train/val/test 其中一邊，
唔可以將同一場牌局嘅決策撕開兩份（會漏料——同一局入面嘅決策高度相關）。
"""

import numpy as np

SPLIT_SEED = 42
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
# test 攞返剩低嗰啲


def assign_splits(n_files: int) -> np.ndarray:
    """回傳 length n_files 嘅 array，值係 'train'/'val'/'test'。

    Index 對應嗰個 match 喺 sorted(glob) 入面嘅次序（即係 files[i]）。
    """
    rng = np.random.default_rng(SPLIT_SEED)
    order = rng.permutation(n_files)
    n_train = int(n_files * TRAIN_FRAC)
    n_val = int(n_files * VAL_FRAC)

    split = np.empty(n_files, dtype=object)
    split[order[:n_train]] = "train"
    split[order[n_train : n_train + n_val]] = "val"
    split[order[n_train + n_val :]] = "test"
    return split
