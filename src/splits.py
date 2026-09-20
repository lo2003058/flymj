"""Assigns each match (one .mjson file = one hanchan) to train/val/test —
never splitting a single match's decisions across two sides (that would
leak information, since decisions within one match are highly correlated).
"""

import numpy as np

SPLIT_SEED = 42
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
# test gets whatever's left


def assign_splits(n_files: int) -> np.ndarray:
    """Returns an array of length n_files, valued 'train'/'val'/'test'.

    Index corresponds to that match's position in sorted(glob) (i.e. files[i]).
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
