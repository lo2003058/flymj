"""Phase 2 Part 1：由已經 parse 好嘅 RoundLog（見 paifu_replay.py 用緊嗰個
jansou.io.mjai 結構）逆推返個 jansou.game.wall.Wall，等可以用 jansou 自己個
game engine（deal_steps）逐步重演歷史牌局。

原理：Wall 淨係一舊完全指定嘅 136 隻牌次序，邊個位擺邊隻牌全部由位置決定
（見 jansou/game/wall.py 個 docstring）。已知嘅位（開手嘅牌、抽牌次序、
dora 指示牌、槓/拔北嘅補牌）擺返落去啱嘅 index，唔知嘅位（通常係 ura 同
牌尾冇抽到嘅牌）用返剩低嘅牌填。
"""

from jansou.core.hand import MeldType
from jansou.core.tiles import FIVE_KINDS, Tile, TileKind
from jansou.game.wall import DEAD_WALL_SIZE, DEAL_ROUNDS, DORA_SLOTS, URA_SLOTS, Wall
from jansou.io.paifu import Call, DoraReveal, Draw, Kita, RoundLog

_N_TILES = 136
_KAN_MELD_TYPES = {MeldType.ANKAN, MeldType.SHOUMINKAN, MeldType.DAIMINKAN}


def _dealt_slot_indices(player_count: int) -> list[list[int]]:
    """每個 seat 開手嘅 13 隻牌，落喺 wall sequence 邊啲 index。

    要跟實 Wall.deal() 個 4-4-4-1 輪流派法（一個 round 入面一個 seat 攞晒
    嗰 round 嘅份先輪到下一個 seat），唔係一個人攞晒佢嗰 13 個連續 slot，
    否則之後 Wall.deal() 派返出嚟嘅手牌會同歷史記錄對唔上。
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
    """136 隻牌嘅完整牌組，五嘅其中一隻(copy 0)係紅五。"""
    deck: list[Tile] = []
    for kind_value in range(34):
        kind = TileKind(kind_value)
        is_five = kind in FIVE_KINDS
        for copy in range(4):
            deck.append(Tile(kind, red=(is_five and copy == 0)))
    return deck


def _take(pool: list[Tile], tile: Tile) -> None:
    pool.remove(tile)  # 靠 Tile 嘅 (kind, red) equality，移走第一個啱嘅


def _classify_draws(events) -> tuple[list[Tile], list[Tile]]:
    """行一次 event stream，分開「正常抽牌」同「槓/拔北嘅補牌」，各自
    按時間次序回傳。"""
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

    # 1. Dora 指示牌：slot 4 = initial_dora，之後嘅 DoraReveal 順序擺 6,8,10,12
    sequence[DORA_SLOTS[0]] = round_log.initial_dora
    _take(pool, round_log.initial_dora)
    reveals = [e.indicator for e in round_log.events if isinstance(e, DoraReveal)]
    for i, indicator in enumerate(reveals, start=1):
        sequence[DORA_SLOTS[i]] = indicator
        _take(pool, indicator)

    # 2. Ura 指示牌：如果有人糊咗仲要有 riichi ura 記錄，用返個記錄；
    #    冇嘅話（冇人糊 / 冇 riichi）留返 None，尾段用剩低嘅牌填。
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

    # 3. 補牌（槓/拔北）：slot 0-3，按時間次序
    ordinary_draws, replacement_draws = _classify_draws(round_log.events)
    for i, tile in enumerate(replacement_draws):
        sequence[i] = tile
        _take(pool, tile)

    # 4. 開手嘅牌：跟 Wall.deal() 個 4-4-4-1 輪流 pattern 擺落啱嘅 slot
    #    （一個 seat 嗰 13 隻入面邊隻擺邊個 slot 唔緊要，順序擺就得）
    dealt_slots = _dealt_slot_indices(player_count)
    for hand, seat_slots in zip(round_log.hands, dealt_slots):
        for tile, slot in zip(hand, seat_slots):
            sequence[slot] = tile
            _take(pool, tile)

    # 5. 正常抽牌：跟住 52 隻開手牌之後，按時間次序（開手用咗嘅最大 slot 之後）
    idx = 14 + player_count * 13
    for tile in ordinary_draws:
        sequence[idx] = tile
        _take(pool, tile)
        idx += 1

    # 6. 剩低嘅 None（未知嘅 ura + 牌尾冇抽到嘅牌）用返剩低嘅牌填，次序唔緊要
    for slot in range(_N_TILES):
        if sequence[slot] is None:
            sequence[slot] = pool.pop()

    assert not pool, f"仲有 {len(pool)} 隻牌冇擺，reconstruction 有 bug"
    return Wall(tuple(sequence))  # type: ignore[arg-type]
