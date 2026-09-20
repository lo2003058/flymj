"""Phase 1 Part 2: fetches the real 3D soma coordinates of PN/KC/MBON
neurons (the `somaLocation` column in annotations.feather), for the UI to
plot "which neuron is where, and how active is it right now."

Coordinate order matches pn_ids/kc_ids/mbon_ids in masks.npz, so indices
map directly onto the model's PN/KC/MBON output dimensions. The handful of
neurons without a soma coordinate (fragments, incomplete reconstructions)
get NaN coordinates and are skipped when plotting.
"""

import numpy as np
import polars as pl

from io_utils import load_feather

ANNOTATIONS_PATH = "data/raw/annotations.feather"


def load_coords_for(ids: np.ndarray, ann: pl.DataFrame | None = None) -> np.ndarray:
    """Returns an array of shape (len(ids), 3), in the same order as ids.
    Neurons without a soma coordinate get NaN."""
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
    """Fetch coordinates for all three layers (PN/KC/MBON) at once,
    reading annotations.feather only once."""
    ann = load_feather(ANNOTATIONS_PATH)
    return {
        "pn": load_coords_for(masks_npz["pn_ids"], ann),
        "kc": load_coords_for(masks_npz["kc_ids"], ann),
        "mbon": load_coords_for(masks_npz["mbon_ids"], ann),
    }
