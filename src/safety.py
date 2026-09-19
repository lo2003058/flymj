"""最簡單、最實用嗰種防守判斷：現物（genbutsu）。

規則好簡單，唔使識牌理都明：如果一個對手自己已經掉過某隻牌，噉隻牌就
放心打俾佢——因為如果嗰隻牌真係佢嘅聽牌，佢自己一早都已經振聽咗（自摸
可以，但唔可以食糊），實戰上等於「呢隻牌對呢個人 100% 唔會食糊」。

呢個判斷淨係靠已經有嘅牌河資料，唔使識佢聽緊咩、唔使算牌，係最基本、
最無爭議嘅安全讀法，其他好似筋（suji）、壁（kabe）呢啲屬於統計性、
有機會出錯嘅進階讀法，暫時唔做。
"""

SEAT_LABELS_REL = ["你自己", "下家", "對家", "上家"]


def genbutsu_seats(discard_counts: list[list[int]], tile: int, player_count: int = 4) -> list[int]:
    """回傳邊啲相對座位（1=下家/2=對家/3=上家）自己掉過呢隻牌，即係對佢哋
    嚟講而家已經係現物。"""
    return [seat for seat in range(1, player_count) if discard_counts[seat][tile] > 0]


def safety_label(discard_counts: list[list[int]], tile: int, riichi: list[bool], player_count: int = 4) -> str:
    """一句形容：呢隻牌對邊個安全，有冇人立直緊都俾埋提示。"""
    safe_seats = genbutsu_seats(discard_counts, tile, player_count)
    riichi_seats = [s for s in range(1, player_count) if riichi[s]]

    if not riichi_seats:
        if safe_seats:
            names = "、".join(SEAT_LABELS_REL[s] for s in safe_seats)
            return f"現物：對 {names} 安全"
        return ""

    safe_vs_riichi = [s for s in riichi_seats if s in safe_seats]
    unsafe_vs_riichi = [s for s in riichi_seats if s not in safe_seats]

    if safe_vs_riichi and not unsafe_vs_riichi:
        names = "、".join(SEAT_LABELS_REL[s] for s in safe_vs_riichi)
        return f"✅ 對立直緊嘅 {names} 係現物"
    if safe_vs_riichi and unsafe_vs_riichi:
        safe_names = "、".join(SEAT_LABELS_REL[s] for s in safe_vs_riichi)
        unsafe_names = "、".join(SEAT_LABELS_REL[s] for s in unsafe_vs_riichi)
        return f"⚠️ 對 {safe_names} 安全，但對立直緊嘅 {unsafe_names} 未知（唔係現物）"
    names = "、".join(SEAT_LABELS_REL[s] for s in unsafe_vs_riichi)
    return f"⚠️ 對立直緊嘅 {names} 未知安全與否（唔係現物，可能放銃）"
