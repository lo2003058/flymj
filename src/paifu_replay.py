"""Replays a RoundLog's event stream, tracking each seat's concealed hand
step by step, and yields the full state at the moment of each Discard event.

This logic mirrors what `jansou.io.paifu`'s `_SeatState`/`_apply_*`
(private, not exported) do, but only tracks what's needed for "discard
decisions" — it doesn't need their full riichi/ippatsu/haitei win-context
tracking.
"""

from dataclasses import dataclass

from jansou.core.hand import Meld, MeldType
from jansou.core.tiles import Tile, TileKind
from jansou.io.paifu import Call, Discard, DoraReveal, Draw, Kita, RoundLog


@dataclass(frozen=True)
class DiscardDecision:
    event_index: int
    seat: int
    hand_counts: list[int]  # 34 slots, this seat's concealed hand at the moment of the discard
    hand_red_counts: list[int]  # 34 slots, red fives
    n_melds: int
    n_dora_indicators: int  # includes initial_dora, so always >=1
    discard_tile: int  # 0-33
    discard_is_red: bool
    is_riichi: bool
    is_tsumogiri: bool


def _hand_counts(concealed: list[Tile]) -> tuple[list[int], list[int]]:
    counts = [0] * 34
    red_counts = [0] * 34
    for tile in concealed:
        counts[tile.kind.value] += 1
        if tile.red:
            red_counts[tile.kind.value] += 1
    return counts, red_counts


def _remove_meld_tiles(concealed: list[Tile], meld: Meld) -> None:
    """Same logic as jansou.io.paifu._remove_meld_tiles: removes from the
    meld only the tiles that weren't the "called" tile (the called tile
    itself was never in one's own concealed hand)."""
    from_hand = list(meld.tiles)
    if meld.called is not None:
        from_hand.remove(meld.called)
    for tile in from_hand:
        concealed.remove(tile)


def iter_discard_decisions(round_log: RoundLog, player_count: int):
    """Replay a round's events step by step, yielding a DiscardDecision
    every time a Discard is encountered."""
    concealed: list[list[Tile]] = [list(hand) for hand in round_log.hands]
    n_melds = [0] * player_count
    n_dora = 1  # initial_dora itself counts as the first one

    for index, event in enumerate(round_log.events):
        if isinstance(event, Draw):
            concealed[event.seat].append(event.tile)

        elif isinstance(event, Discard):
            hand_counts, hand_red_counts = _hand_counts(concealed[event.seat])
            yield DiscardDecision(
                event_index=index,
                seat=event.seat,
                hand_counts=hand_counts,
                hand_red_counts=hand_red_counts,
                n_melds=n_melds[event.seat],
                n_dora_indicators=n_dora,
                discard_tile=event.tile.kind.value,
                discard_is_red=event.tile.red,
                is_riichi=event.riichi,
                is_tsumogiri=event.tsumogiri,
            )
            concealed[event.seat].remove(event.tile)

        elif isinstance(event, Call):
            meld = event.meld
            if meld.type is MeldType.SHOUMINKAN:
                # Added kan: upgrades an already-counted pon; the added
                # tile was just drawn and is still in concealed.
                concealed[event.seat].remove(meld.added)  # type: ignore[arg-type]
            else:
                _remove_meld_tiles(concealed[event.seat], meld)
                n_melds[event.seat] += 1

        elif isinstance(event, Kita):
            concealed[event.seat].remove(Tile(TileKind.NORTH))

        elif isinstance(event, DoraReveal):
            n_dora += 1


@dataclass(frozen=True)
class FullDiscardState:
    """The full game state of a discard decision: one's own private state
    + public information for all 4 seats.

    Used for Step 4 (feature encoding); kept separate from DiscardDecision
    because Step 3 already validated that iter_discard_decisions produces
    a correct dataset, and we don't want to touch that already-verified
    code just to add public information.
    """

    event_index: int
    seat: int
    hand_counts: list[int]  # 34, own concealed hand
    hand_red_counts: list[int]  # 34
    meld_counts: list[list[int]]  # 4 x 34, each seat's melded tiles
    discard_counts: list[list[int]]  # 4 x 34, each seat's discard pile
    riichi: list[bool]  # 4
    dora_tiles: list[int]  # currently active dora tile types (can repeat)
    round_wind: int
    seat_wind: int  # the seat wind of the seat currently deciding
    discard_tile: int
    discard_is_red: bool
    is_riichi: bool
    is_tsumogiri: bool


def iter_full_discard_states(round_log: RoundLog, player_count: int):
    """Same replay as iter_discard_decisions, but also tracks public
    information for all 4 seats."""
    concealed: list[list[Tile]] = [list(hand) for hand in round_log.hands]
    meld_counts = [[0] * 34 for _ in range(player_count)]
    discard_counts = [[0] * 34 for _ in range(player_count)]
    riichi = [False] * player_count
    dora_tiles = [round_log.initial_dora.kind.successor().value]

    for index, event in enumerate(round_log.events):
        if isinstance(event, Draw):
            concealed[event.seat].append(event.tile)

        elif isinstance(event, Discard):
            hand_counts, hand_red_counts = _hand_counts(concealed[event.seat])
            seat_wind = (event.seat - round_log.dealer) % player_count
            yield FullDiscardState(
                event_index=index,
                seat=event.seat,
                hand_counts=hand_counts,
                hand_red_counts=hand_red_counts,
                meld_counts=[list(c) for c in meld_counts],
                discard_counts=[list(c) for c in discard_counts],
                riichi=list(riichi),
                dora_tiles=list(dora_tiles),
                round_wind=round_log.round_wind.value,
                seat_wind=seat_wind,
                discard_tile=event.tile.kind.value,
                discard_is_red=event.tile.red,
                is_riichi=event.riichi,
                is_tsumogiri=event.tsumogiri,
            )
            concealed[event.seat].remove(event.tile)
            discard_counts[event.seat][event.tile.kind.value] += 1
            if event.riichi:
                riichi[event.seat] = True

        elif isinstance(event, Call):
            meld = event.meld
            if meld.type is MeldType.SHOUMINKAN:
                concealed[event.seat].remove(meld.added)  # type: ignore[arg-type]
                meld_counts[event.seat][meld.added.kind.value] += 1  # type: ignore[union-attr]
            else:
                _remove_meld_tiles(concealed[event.seat], meld)
                for tile in meld.tiles:
                    meld_counts[event.seat][tile.kind.value] += 1

        elif isinstance(event, Kita):
            concealed[event.seat].remove(Tile(TileKind.NORTH))

        elif isinstance(event, DoraReveal):
            dora_tiles.append(event.indicator.kind.successor().value)
