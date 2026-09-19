"""Phase 2 Part 4：由 action_dataset.parquet 砌 (34 x 34) feature tensor +
三個 head 各自嘅 label，存做 data/processed/action_features.npz。

輸出：
  X             (N, 34, 34) uint8
  self_type     (N,) int8   SELF 決策揀咗邊種（index 落 action_model.SELF_ACTION_TYPES），
                            唔係 SELF 決策就係 -1
  discard_tile  (N,) int8   DISCARD/RIICHI 揀咗邊隻牌，其他情況 -1
  reaction      (N,) int8   DISCARD_REACTION 決策揀咗邊種（index 落
                            action_model.REACTION_ACTION_TYPES），其他情況 -1
  decision_kind (N,) <U20
  split         (N,) <U5

跑法： uv run python src/build_action_features.py
"""

import numpy as np
import polars as pl

from action_features import N_CHANNELS, N_TILE_TYPES, encode_row
from action_model import REACTION_ACTION_TYPES, SELF_ACTION_TYPES

DATASET_PATH = "data/processed/action_dataset.parquet"
OUT_PATH = "data/processed/action_features.npz"


def main() -> None:
    df = pl.read_parquet(DATASET_PATH)
    print(f"讀到 {df.height} 行")

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

        if (i + 1) % 50_000 == 0:
            print(f"已編碼 {i + 1}/{df.height}")

    x = np.stack(x_list).astype(np.uint8)
    self_type = np.array(self_type_list, dtype=np.int8)
    discard_tile = np.array(discard_tile_list, dtype=np.int8)
    reaction = np.array(reaction_list, dtype=np.int8)
    decision_kind = df["decision_kind"].to_numpy().astype("<U20")
    split = df["split"].to_numpy().astype("<U5")

    print(f"\nX shape={x.shape} dtype={x.dtype}")
    assert x.shape[1:] == (N_CHANNELS, N_TILE_TYPES)

    print("\n=== self_type 分佈（SELF 決策）===")
    for i, name in enumerate(SELF_ACTION_TYPES):
        print(f"  {name}: {(self_type == i).sum()}")
    print("\n=== reaction 分佈（DISCARD_REACTION 決策）===")
    for i, name in enumerate(REACTION_ACTION_TYPES):
        print(f"  {name}: {(reaction == i).sum()}")
    print(f"\ndiscard_tile 有效（!=-1）行數: {(discard_tile != -1).sum()}")

    np.savez_compressed(
        OUT_PATH,
        X=x,
        self_type=self_type,
        discard_tile=discard_tile,
        reaction=reaction,
        decision_kind=decision_kind,
        split=split,
    )
    print(f"\n已存 {OUT_PATH}")


if __name__ == "__main__":
    main()
