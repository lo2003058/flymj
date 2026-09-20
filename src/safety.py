"""The simplest, most practical defense read: genbutsu (safe tiles).

The rule is simple enough to need no hand-reading skill: if an opponent has
already discarded a given tile themselves, that tile is safe to discard to
them — because if it really were their winning tile, they'd already be in
furiten (they can still tsumo, but can't ron on it). In practice this means
"this tile is 100% guaranteed not to deal in against this player."

This read relies purely on discard-pile data already visible — no hand
reading, no tile counting — making it the most basic, uncontroversial
safety read. More advanced, statistical (and error-prone) reads like suji
or kabe are deliberately left out for now.
"""

SEAT_LABELS_REL = ["You", "Right", "Across", "Left"]


def genbutsu_seats(discard_counts: list[list[int]], tile: int, player_count: int = 4) -> list[int]:
    """Return which relative seats (1=right/2=across/3=left) have
    discarded this tile themselves, i.e. it's currently genbutsu against
    them."""
    return [seat for seat in range(1, player_count) if discard_counts[seat][tile] > 0]


def safety_label(discard_counts: list[list[int]], tile: int, riichi: list[bool], player_count: int = 4) -> str:
    """A one-line description of who this tile is safe against, flagging
    any riichi players."""
    safe_seats = genbutsu_seats(discard_counts, tile, player_count)
    riichi_seats = [s for s in range(1, player_count) if riichi[s]]

    if not riichi_seats:
        if safe_seats:
            names = ", ".join(SEAT_LABELS_REL[s] for s in safe_seats)
            return f"Genbutsu: safe against {names}"
        return ""

    safe_vs_riichi = [s for s in riichi_seats if s in safe_seats]
    unsafe_vs_riichi = [s for s in riichi_seats if s not in safe_seats]

    if safe_vs_riichi and not unsafe_vs_riichi:
        names = ", ".join(SEAT_LABELS_REL[s] for s in safe_vs_riichi)
        return f"✅ Genbutsu against riichi from {names}"
    if safe_vs_riichi and unsafe_vs_riichi:
        safe_names = ", ".join(SEAT_LABELS_REL[s] for s in safe_vs_riichi)
        unsafe_names = ", ".join(SEAT_LABELS_REL[s] for s in unsafe_vs_riichi)
        return f"⚠️ Safe against {safe_names}, but unknown against riichi from {unsafe_names} (not genbutsu)"
    names = ", ".join(SEAT_LABELS_REL[s] for s in unsafe_vs_riichi)
    return f"⚠️ Unknown safety against riichi from {names} (not genbutsu, could deal in)"
