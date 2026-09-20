"""Step 4: encodes a FullDiscardState into a (C, 34) feature tensor.

Channel design (32 planes, all binary 0/1):

  0-3   own hand count thermometer (>=1 / >=2 / >=3 / >=4)
  4     whether own hand has a red five
  5-8   own melded tiles count thermometer
  9-12  own (relative seat 0) discard pile count thermometer
  13-16 right (+1) discard pile count thermometer
  17-20 across (+2) discard pile count thermometer
  21-24 left (+3) discard pile count thermometer
  25    currently active dora tile
  26    round wind (one-hot into columns 27-30)
  27    seat wind
  28    own riichi status (broadcast)
  29-31 right/across/left riichi status (broadcast)
"""

import numpy as np

from paifu_replay import FullDiscardState

N_TILE_TYPES = 34
N_CHANNELS = 32


def _thermometer(counts: list[int]) -> np.ndarray:
    """count -> 4 binary planes: plane k means "this tile has at least k+1 copies"."""
    arr = np.asarray(counts, dtype=np.uint8)
    return np.stack([(arr >= k).astype(np.uint8) for k in (1, 2, 3, 4)])


def encode(state: FullDiscardState, player_count: int = 4) -> np.ndarray:
    planes = [
        _thermometer(state.hand_counts),  # 4
        (np.asarray(state.hand_red_counts, dtype=np.uint8) > 0).astype(np.uint8)[None, :],  # 1
        _thermometer(state.meld_counts[state.seat]),  # 4
    ]

    for rel in range(player_count):
        seat = (state.seat + rel) % player_count
        planes.append(_thermometer(state.discard_counts[seat]))  # 4 x 4 = 16

    dora_plane = np.zeros(N_TILE_TYPES, dtype=np.uint8)
    for t in state.dora_tiles:
        dora_plane[t] = 1
    planes.append(dora_plane[None, :])  # 1

    wind_plane = np.zeros(N_TILE_TYPES, dtype=np.uint8)
    wind_plane[27 + state.round_wind] = 1
    planes.append(wind_plane[None, :])  # 1

    seat_wind_plane = np.zeros(N_TILE_TYPES, dtype=np.uint8)
    seat_wind_plane[27 + state.seat_wind] = 1
    planes.append(seat_wind_plane[None, :])  # 1

    for rel in range(player_count):
        seat = (state.seat + rel) % player_count
        value = np.uint8(1 if state.riichi[seat] else 0)
        planes.append(np.full((1, N_TILE_TYPES), value, dtype=np.uint8))  # 4

    tensor = np.concatenate(planes, axis=0)
    assert tensor.shape == (N_CHANNELS, N_TILE_TYPES), tensor.shape
    return tensor
