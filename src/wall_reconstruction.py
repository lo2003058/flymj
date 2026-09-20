"""Phase 2 Part 1: reconstructs a jansou.game.wall.Wall from an already-
parsed RoundLog (the jansou.io.mjai structure used by paifu_replay.py), so
jansou's own game engine (deal_steps) can replay a historical match step
by step.

The idea: a Wall is just one fully-specified 136-tile sequence — which
tile sits at which position is entirely determined by position (see the
docstring in jansou/game/wall.py). Known positions (starting hands, draw
order, dora indicators, kan/nuki replacement draws) get placed at their
correct index; unknown positions (usually the ura indicators and
never-drawn tail tiles) get filled with whatever tiles remain.
"""

from jansou.core.hand import MeldType
from jansou.core.tiles import FIVE_KINDS, Tile, TileKind
from jansou.game.wall import DEAD_WALL_SIZE, DEAL_ROUNDS, DORA_SLOTS, URA_SLOTS, Wall
from jansou.io.paifu import Call, DoraReveal, Draw, Kita, RoundLog

_N_TILES = 136
_KAN_MELD_TYPES = {MeldType.ANKAN, MeldType.SHOUMINKAN, MeldType.DAIMINKAN}


def _dealt_slot_indices(player_count: int) -> list[list[int]]:
    """Which wall-sequence indices hold each seat's 13 starting tiles.

    This must exactly follow Wall.deal()'s 4-4-4-1 round-robin dealing
    pattern (one seat takes that round's share, then the next seat takes
    theirs), not contiguous blocks per seat — otherwise the hands
    Wall.deal() later produces won't match the historical record.
    """
    slots: list[list[int]] = [[] for _ in range(player_count)]
    front = DEAD_WALL_SIZE
    for count in DEAL_ROUNDS:
        for seat in range(player_count):
            for _ in range(count):
                slots[seat].append(front)
                front += 1
    return slots


def full_deck() -> list[Tile]:
    """The full 136-tile deck, where one copy (copy 0) of each five is the red five."""
    deck: list[Tile] = []
    for kind_value in range(34):
        kind = TileKind(kind_value)
        is_five = kind in FIVE_KINDS
        for copy in range(4):
            deck.append(Tile(kind, red=(is_five and copy == 0)))
    return deck


def _take(pool: list[Tile], tile: Tile) -> None:
    pool.remove(tile)  # relies on Tile's (kind, red) equality, removes the first match


def _classify_draws(events) -> tuple[list[Tile], list[Tile]]:
    """Walk the event stream once, separating "ordinary draws" from
    "kan/nuki replacement draws," each returned in chronological order."""
    ordinary: list[Tile] = []
    replacement: list[Tile] = []
    pending_replacement = False
    for event in events:
        if isinstance(event, Call) and event.meld.type in _KAN_MELD_TYPES:
            pending_replacement = True
        elif isinstance(event, Kita):
            pending_replacement = True
        elif isinstance(event, Draw):
            if pending_replacement:
                replacement.append(event.tile)
                pending_replacement = False
            else:
                ordinary.append(event.tile)
    return ordinary, replacement


def reconstruct_wall(round_log: RoundLog, player_count: int) -> Wall:
    sequence: list[Tile | None] = [None] * _N_TILES
    pool = full_deck()

    # 1. Dora indicators: slot 4 = initial_dora, subsequent DoraReveals go in order at 6,8,10,12
    sequence[DORA_SLOTS[0]] = round_log.initial_dora
    _take(pool, round_log.initial_dora)
    reveals = [e.indicator for e in round_log.events if isinstance(e, DoraReveal)]
    for i, indicator in enumerate(reveals, start=1):
        sequence[DORA_SLOTS[i]] = indicator
        _take(pool, indicator)

    # 2. Ura indicators: if someone won with a riichi ura record, use it;
    #    otherwise (no win / no riichi) leave as None and fill from
    #    remaining tiles at the end.
    ura_known: list[Tile] = []
    outcome = round_log.outcome
    agari_list = outcome if isinstance(outcome, tuple) else ()
    for agari in agari_list:
        if agari.ura_indicators:
            ura_known = list(agari.ura_indicators)
            break
    for i, tile in enumerate(ura_known):
        slot = URA_SLOTS[i]
        if tile in pool:
            sequence[slot] = tile
            _take(pool, tile)

    # 3. Replacement draws (kan/nuki): slots 0-3, in chronological order
    ordinary_draws, replacement_draws = _classify_draws(round_log.events)
    for i, tile in enumerate(replacement_draws):
        sequence[i] = tile
        _take(pool, tile)

    # 4. Starting hands: placed following Wall.deal()'s 4-4-4-1 round-robin
    #    pattern (which of a seat's 13 tiles goes in which slot doesn't
    #    matter, as long as they're placed in order)
    dealt_slots = _dealt_slot_indices(player_count)
    for hand, seat_slots in zip(round_log.hands, dealt_slots):
        for tile, slot in zip(hand, seat_slots):
            sequence[slot] = tile
            _take(pool, tile)

    # 5. Ordinary draws: follow the 52 starting-hand tiles, in
    #    chronological order (starting right after the highest slot used by starting hands)
    idx = 14 + player_count * 13
    for tile in ordinary_draws:
        sequence[idx] = tile
        _take(pool, tile)
        idx += 1

    # 6. Fill any remaining None slots (unknown ura + never-drawn tail
    #    tiles) with whatever tiles are left, order doesn't matter
    for slot in range(_N_TILES):
        if sequence[slot] is None:
            sequence[slot] = pool.pop()

    assert not pool, f"{len(pool)} tiles left unplaced — reconstruction has a bug"
    return Wall(tuple(sequence))  # type: ignore[arg-type]
