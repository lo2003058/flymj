"""Phase 2 UI 細化：俾定手牌 + 反應緊嘅棄牌，計返實際可以用邊啲手牌組成
PON/CHII（純用 count 計，唔使起成個 jansou GameState）。

呢個唔係 model 估緊嘅嘢，係實際牌理計出嚟嘅合法組合，等 UI 可以話畀
使用者知「叫嘅話實際係用邊兩隻手牌」，唔淨係得個 action 種類。
"""

_SUIT_SIZE = 9


def pon_available(hand_counts: list[int], trigger_tile: int) -> bool:
    return trigger_tile >= 0 and hand_counts[trigger_tile] >= 2


def chii_combos(hand_counts: list[int], trigger_tile: int) -> list[tuple[int, int]]:
    """回傳吃嗰隻牌可以用嘅每一種（tile1, tile2）組合。字牌冇得吃，回傳 []。"""
    if trigger_tile < 0 or trigger_tile >= 27:
        return []

    suit_base = (trigger_tile // _SUIT_SIZE) * _SUIT_SIZE
    rank = trigger_tile - suit_base  # 0-8

    combos: list[tuple[int, int]] = []
    for low in (rank - 2, rank - 1, rank):
        high = low + 2
        if low < 0 or high > 8:
            continue
        needed = [suit_base + r for r in (low, low + 1, low + 2) if suit_base + r != trigger_tile]
        t1, t2 = needed
        if t1 == t2:
            ok = hand_counts[t1] >= 2
        else:
            ok = hand_counts[t1] >= 1 and hand_counts[t2] >= 1
        if ok:
            combos.append((t1, t2))
    return combos


def open_kan_available(hand_counts: list[int], trigger_tile: int) -> bool:
    return trigger_tile >= 0 and hand_counts[trigger_tile] >= 3
