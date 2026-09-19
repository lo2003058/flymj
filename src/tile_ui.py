"""34 隻麻雀牌 index (0-33) <-> Unicode 麻雀牌符號 / 短文字標籤。

TileKind 次序：0-8 萬(m)，9-17 筒(p)，18-26 索(s)，27-33 東南西北白發中。
Unicode「Mahjong Tiles」block 次序唔一樣（先風牌/三元牌，再萬，再索，再筒），
要逐段對返。
"""

_MAN_BASE = 0x1F007  # 0x1F007..0x1F00F = 1-9 萬
_PIN_BASE = 0x1F019  # 0x1F019..0x1F021 = 1-9 筒
_SOU_BASE = 0x1F010  # 0x1F010..0x1F018 = 1-9 索
_HONOR_CODEPOINTS = {
    27: 0x1F000,  # 東
    28: 0x1F001,  # 南
    29: 0x1F002,  # 西
    30: 0x1F003,  # 北
    31: 0x1F006,  # 白
    32: 0x1F005,  # 發
    33: 0x1F004,  # 中
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
    """人類睇得明嘅牌名（唔用 m/p/s 呢種簡寫）。"""
    if 0 <= kind <= 8:
        return f"{kind + 1}萬"
    if 9 <= kind <= 17:
        return f"{kind - 9 + 1}筒"
    if 18 <= kind <= 26:
        return f"{kind - 18 + 1}索"
    return _HONOR_LABELS[kind]


TILE_GLYPHS = [tile_glyph(k) for k in range(34)]
TILE_LABELS = [tile_label(k) for k in range(34)]

#: 邊幾個 index 係「5」，先可以俾紅五 checkbox 用（呢度嘅 "m"/"p"/"s" 淨係
#: 內部 key，唔會顯示俾使用者睇，顯示嗰陣用 tile_label()/TILE_LABELS）。
FIVE_KINDS = {4: "m", 13: "p", 22: "s"}
