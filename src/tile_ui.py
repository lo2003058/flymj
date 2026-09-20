"""Maps the 34 mahjong tile indices (0-33) <-> Unicode mahjong tile glyphs /
short text labels.

TileKind order: 0-8 man, 9-17 pin, 18-26 sou, 27-33 East/South/West/North/
White/Green/Red. The Unicode "Mahjong Tiles" block orders things
differently (winds/dragons first, then man, then sou, then pin), so each
range has to be mapped separately.
"""

_MAN_BASE = 0x1F007  # 0x1F007..0x1F00F = man 1-9
_PIN_BASE = 0x1F019  # 0x1F019..0x1F021 = pin 1-9
_SOU_BASE = 0x1F010  # 0x1F010..0x1F018 = sou 1-9
_HONOR_CODEPOINTS = {
    27: 0x1F000,  # East
    28: 0x1F001,  # South
    29: 0x1F002,  # West
    30: 0x1F003,  # North
    31: 0x1F006,  # White dragon
    32: 0x1F005,  # Green dragon
    33: 0x1F004,  # Red dragon
}
_HONOR_LABELS = {27: "東", 28: "南", 29: "西", 30: "北", 31: "白", 32: "發", 33: "中"}


def tile_glyph(kind: int) -> str:
    if 0 <= kind <= 8:
        return chr(_MAN_BASE + kind)
    if 9 <= kind <= 17:
        return chr(_PIN_BASE + (kind - 9))
    if 18 <= kind <= 26:
        return chr(_SOU_BASE + (kind - 18))
    return chr(_HONOR_CODEPOINTS[kind])


def tile_label(kind: int) -> str:
    """Human-readable tile name (not the m/p/s shorthand)."""
    if 0 <= kind <= 8:
        return f"{kind + 1}萬"
    if 9 <= kind <= 17:
        return f"{kind - 9 + 1}筒"
    if 18 <= kind <= 26:
        return f"{kind - 18 + 1}索"
    return _HONOR_LABELS[kind]


TILE_GLYPHS = [tile_glyph(k) for k in range(34)]
TILE_LABELS = [tile_label(k) for k in range(34)]

#: Which indices are a "5", for the red-five checkbox. The "m"/"p"/"s"
#: here are just internal keys, never shown to the user — display uses
#: tile_label()/TILE_LABELS instead.
FIVE_KINDS = {4: "m", 13: "p", 22: "s"}
