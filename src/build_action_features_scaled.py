"""Scaling experiment Part 6: builds a (34 x 34) feature tensor plus each
head's labels for the 10-year action dataset in
build_action_dataset_scaled.py — same schema as action_features.npz, but a
separate file that doesn't overwrite the original.

Run: uv run python src/build_action_features_scaled.py
"""

import numpy as np
import polars as pl

from action_features import N_CHANNELS, N_TILE_TYPES, encode_row
from action_model import REACTION_ACTION_TYPES, SELF_ACTION_TYPES

DATASET_PATH = "data/processed/action_dataset_scaled.parquet"
OUT_PATH = "data/processed/action_features_scaled.npz"


def main() -> None:
    df = pl.read_parquet(DATASET_PATH)
    print(f"Read {df.height} rows")

    x_list: list[np.ndarray] = []
    self_type_list: list[int] = []
    discard_tile_list: list[int] = []
    reaction_list: list[int] = []

    for i, row in enumerate(df.iter_rows(named=True)):
        x_list.append(encode_row(row))

        if row["decision_kind"] == "SELF":
            action_type = row["action_type"]
            self_type_list.append(SELF_ACTION_TYPES.index(action_type))
            discard_tile_list.append(row["tile"] if action_type in ("DISCARD", "RIICHI") else -1)
            reaction_list.append(-1)
        else:
            self_type_list.append(-1)
            discard_tile_list.append(-1)
            reaction_list.append(REACTION_ACTION_TYPES.index(row["action_type"]))

        if (i + 1) % 500_000 == 0:
            print(f"Encoded {i + 1}/{df.height}")

    x = np.stack(x_list).astype(np.uint8)
    self_type = np.array(self_type_list, dtype=np.int8)
    discard_tile = np.array(discard_tile_list, dtype=np.int8)
    reaction = np.array(reaction_list, dtype=np.int8)
    decision_kind = df["decision_kind"].to_numpy().astype("<U20")
    split = df["split"].to_numpy().astype("<U5")

    print(f"\nX shape={x.shape} dtype={x.dtype}")
    assert x.shape[1:] == (N_CHANNELS, N_TILE_TYPES)

    print("\n=== self_type distribution (SELF decisions) ===")
    for i, name in enumerate(SELF_ACTION_TYPES):
        print(f"  {name}: {(self_type == i).sum()}")
    print("\n=== reaction distribution (DISCARD_REACTION decisions) ===")
    for i, name in enumerate(REACTION_ACTION_TYPES):
        print(f"  {name}: {(reaction == i).sum()}")
    print(f"\nRows with a valid discard_tile (!=-1): {(discard_tile != -1).sum()}")

    np.savez_compressed(
        OUT_PATH,
        X=x,
        self_type=self_type,
        discard_tile=discard_tile,
        reaction=reaction,
        decision_kind=decision_kind,
        split=split,
    )
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
