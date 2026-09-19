"""Step 4：將一個 FullDiscardState 編碼做 (C, 34) feature tensor。

Channel 設計（32 planes，全部 binary 0/1）：

  0-3   自己手牌 count thermometer（>=1 / >=2 / >=3 / >=4）
  4     自己手牌入面有冇紅五
  5-8   自己已 meld 嘅牌 count thermometer
  9-12  自己（relative seat 0）牌河 count thermometer
  13-16 下家（+1）牌河 count thermometer
  17-20 對家（+2）牌河 count thermometer
  21-24 上家（+3）牌河 count thermometer
  25    現正生效嘅 dora 牌
  26    場風（one-hot 落 27-30 嗰四欄）
  27    自風
  28    自己 riichi 咗未（broadcast）
  29-31 下家/對家/上家 riichi 咗未（broadcast）
"""

import numpy as np

from paifu_replay import FullDiscardState

N_TILE_TYPES = 34
N_CHANNELS = 32


def _thermometer(counts: list[int]) -> np.ndarray:
    """count -> 4 個 binary planes：plane k 代表「呢隻牌至少有 k+1 隻」。"""
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
