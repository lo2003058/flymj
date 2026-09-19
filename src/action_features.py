"""Phase 2 Part 3：將 action_dataset.parquet 嘅一行（場面 + trigger）編碼做
(34, 34) feature tensor——喺 Phase 1 discard-only 嗰 32-channel（見
features.py）上面加多 2 個 channel：而家反應緊邊隻棄牌、呢個係咪一個
reaction 決策。

同 features.py 分開一個檔，係因為 Phase 1 已經訓練/存低嘅 model_arm_a.pt
跟 32-channel 輸入形狀嚟，唔想動到佢。

Channel 設計（34 planes，全部 binary）：
  0-3   自己手牌 count thermometer
  4     自己手牌有冇紅五
  5-8   自己已 meld 嘅牌 count thermometer
  9-12  自己牌河 count thermometer
  13-16 下家牌河 count thermometer
  17-20 對家牌河 count thermometer
  21-24 上家牌河 count thermometer
  25    現正生效嘅 dora 牌
  26    場風
  27    自風
  28-31 自己/下家/對家/上家 riichi 咗未
  32    而家反應緊邊隻棄牌（DISCARD_REACTION 先有，SELF 決策全 0）
  33    呢個係咪一個 DISCARD_REACTION 決策（broadcast）
"""

import numpy as np

N_TILE_TYPES = 34
N_CHANNELS = 34


def _thermometer(counts: list[int]) -> np.ndarray:
    arr = np.asarray(counts, dtype=np.uint8)
    return np.stack([(arr >= k).astype(np.uint8) for k in (1, 2, 3, 4)])


def encode(
    seat: int,
    hand_counts: list[int],
    hand_red_counts: list[int],
    meld_counts: list[list[int]],
    discard_counts: list[list[int]],
    riichi: list[bool],
    dora_tiles: list[int],
    round_wind: int,
    seat_wind: int,
    trigger_tile: int = -1,
    player_count: int = 4,
) -> np.ndarray:
    """meld_counts/discard_counts/riichi 呢三個係用「絕對 seat」(0-3) 做
    index（同 action_dataset.parquet 嘅 snapshot_state 一致），呢度轉做
    「相對於 seat」（自己/下家/對家/上家）先編碼，同 Phase 1 嗰套一致。
    """
    planes = [
        _thermometer(hand_counts),  # 4
        (np.asarray(hand_red_counts, dtype=np.uint8) > 0).astype(np.uint8)[None, :],  # 1
        _thermometer(meld_counts[seat]),  # 4
    ]

    for rel in range(player_count):
        abs_seat = (seat + rel) % player_count
        planes.append(_thermometer(discard_counts[abs_seat]))  # 4 x 4 = 16

    dora_plane = np.zeros(N_TILE_TYPES, dtype=np.uint8)
    for t in dora_tiles:
        dora_plane[t] = 1
    planes.append(dora_plane[None, :])  # 1

    wind_plane = np.zeros(N_TILE_TYPES, dtype=np.uint8)
    wind_plane[27 + round_wind] = 1
    planes.append(wind_plane[None, :])  # 1

    seat_wind_plane = np.zeros(N_TILE_TYPES, dtype=np.uint8)
    seat_wind_plane[27 + seat_wind] = 1
    planes.append(seat_wind_plane[None, :])  # 1

    for rel in range(player_count):
        abs_seat = (seat + rel) % player_count
        value = np.uint8(1 if riichi[abs_seat] else 0)
        planes.append(np.full((1, N_TILE_TYPES), value, dtype=np.uint8))  # 4

    trigger_plane = np.zeros(N_TILE_TYPES, dtype=np.uint8)
    if trigger_tile >= 0:
        trigger_plane[trigger_tile] = 1
    planes.append(trigger_plane[None, :])  # 1

    is_reaction = np.uint8(1 if trigger_tile >= 0 else 0)
    planes.append(np.full((1, N_TILE_TYPES), is_reaction, dtype=np.uint8))  # 1

    tensor = np.concatenate(planes, axis=0)
    assert tensor.shape == (N_CHANNELS, N_TILE_TYPES), tensor.shape
    return tensor


def encode_row(row: dict, player_count: int = 4) -> np.ndarray:
    """由 action_dataset.parquet 嘅一行（已經 to_dict）直接編碼。"""
    return encode(
        seat=row["seat"],
        hand_counts=row["hand_counts"],
        hand_red_counts=row["hand_red_counts"],
        meld_counts=row["meld_counts"],
        discard_counts=row["discard_counts"],
        riichi=row["riichi"],
        dora_tiles=row["dora_tiles"],
        round_wind=row["round_wind"],
        seat_wind=row["seat_wind"],
        trigger_tile=row["trigger_tile"],
        player_count=player_count,
    )
