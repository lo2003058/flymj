"""Phase 2 Part 3: encodes one row of action_dataset.parquet (game state +
trigger) into a (34, 34) feature tensor — adding 2 channels on top of the
Phase 1 discard-only 32-channel scheme (see features.py): which tile is
currently being reacted to, and whether this is a reaction decision.

Kept in a separate file from features.py because the already-trained
Phase 1 checkpoint was built against the 32-channel input shape, which we
don't want to disturb.

Channel design (34 planes, all binary):
  0-3   own hand count thermometer
  4     whether own hand has a red five
  5-8   own melded tiles count thermometer
  9-12  own discard pile count thermometer
  13-16 right (shimocha) discard pile count thermometer
  17-20 across (toimen) discard pile count thermometer
  21-24 left (kamicha) discard pile count thermometer
  25    currently active dora tile
  26    round wind
  27    seat wind
  28-31 self/right/across/left riichi status
  32    which tile is currently being reacted to (only set for
        DISCARD_REACTION; all zero for SELF decisions)
  33    whether this is a DISCARD_REACTION decision (broadcast)
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
    """meld_counts/discard_counts/riichi are indexed by absolute seat
    (0-3), matching action_dataset.parquet's snapshot_state. Here they get
    rotated to seat-relative order (self/right/across/left) before
    encoding, matching the Phase 1 scheme.
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
    """Encode directly from a row of action_dataset.parquet (already a dict)."""
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
