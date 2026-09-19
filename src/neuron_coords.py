"""Phase 1 Part 2：攞 PN/KC/MBON 神經元嘅真實 3D soma 座標（annotations.feather
嘅 `somaLocation` 欄），俾 UI 畫「邊粒神經元喺邊度、依家幾活躍」用。

座標次序同 masks.npz 入面嘅 pn_ids/kc_ids/mbon_ids 一致，等 index 可以直接
對應返個 model 嘅 PN/KC/MBON 輸出維度。少數冇 soma 座標嘅神經元（fragment
邊緣、未追蹤完），座標填 NaN，畫圖嗰陣跳過。
"""

import numpy as np
import polars as pl

from io_utils import load_feather

ANNOTATIONS_PATH = "data/raw/annotations.feather"


def load_coords_for(ids: np.ndarray, ann: pl.DataFrame | None = None) -> np.ndarray:
    """回傳 shape (len(ids), 3) 嘅 array，次序同 ids 一致。冇 soma 座標嘅填 NaN。"""
    if ann is None:
        ann = load_feather(ANNOTATIONS_PATH)

    lookup = {
        row["bodyId"]: row["somaLocation"]
        for row in ann.filter(pl.col("bodyId").is_in(ids)).select(["bodyId", "somaLocation"]).iter_rows(named=True)
    }

    coords = np.full((len(ids), 3), np.nan, dtype=np.float64)
    for i, body_id in enumerate(ids):
        loc = lookup.get(int(body_id))
        if loc is not None:
            coords[i] = loc
    return coords


def load_all_coords(masks_npz) -> dict[str, np.ndarray]:
    """一次過攞 PN/KC/MBON 三層嘅座標，淨係讀一次 annotations.feather。"""
    ann = load_feather(ANNOTATIONS_PATH)
    return {
        "pn": load_coords_for(masks_npz["pn_ids"], ann),
        "kc": load_coords_for(masks_npz["kc_ids"], ann),
        "mbon": load_coords_for(masks_npz["mbon_ids"], ann),
    }
