"""Replay 一個 RoundLog 嘅 event stream，逐步 track 每個 seat 嘅 concealed hand，
喺遇到 Discard event 嗰陣 yield 返嗰吓嘅完整狀態。

呢個邏輯照跟 jansou.io.paifu 入面 `_SeatState`/`_apply_*`（private，唔對外)
嘅做法，但淨係要「掉牌決策」需要嘅嘢，唔使追蹤佢哋嗰套完整嘅
riichi/ippatsu/haitei win-context。
"""

from dataclasses import dataclass

from jansou.core.hand import Meld, MeldType
from jansou.core.tiles import Tile, TileKind
from jansou.io.paifu import Call, Discard, DoraReveal, Draw, Kita, RoundLog


@dataclass(frozen=True)
class DiscardDecision:
    event_index: int
    seat: int
    hand_counts: list[int]  # 34 位，呢個 seat 掉牌嗰吓嘅 concealed hand
    hand_red_counts: list[int]  # 34 位，紅五
    n_melds: int
    n_dora_indicators: int  # 包括 initial_dora，即係 >=1
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
    """跟 jansou.io.paifu._remove_meld_tiles 一樣嘅邏輯：拎走個 meld 入面
    唔係「叫嚟」嗰啲牌（叫嚟嗰隻本身唔喺自己 concealed hand 度）。"""
    from_hand = list(meld.tiles)
    if meld.called is not None:
        from_hand.remove(meld.called)
    for tile in from_hand:
        concealed.remove(tile)


def iter_discard_decisions(round_log: RoundLog, player_count: int):
    """逐步 replay 一個 round 嘅 events，每次遇到 Discard 就 yield 一個 DiscardDecision。"""
    concealed: list[list[Tile]] = [list(hand) for hand in round_log.hands]
    n_melds = [0] * player_count
    n_dora = 1  # initial_dora 本身就算第一個

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
                # 加槓：由已經計過嘅 pon 升級，加嗰隻岩岩先摸到，喺 concealed 度
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
    """一個掉牌決策嘅完整場面：自己私人狀態 + 4 個 seat 嘅公開資訊。

    用喺 Step 4（feature encoding），同 DiscardDecision 分開係因為
    Step 3 已經驗證過 iter_discard_decisions 出嚟嘅 dataset 啱，
    唔想為咗加公開資訊而動嗰段已經核實過嘅 code。
    """

    event_index: int
    seat: int
    hand_counts: list[int]  # 34，自己 concealed hand
    hand_red_counts: list[int]  # 34
    meld_counts: list[list[int]]  # 4 x 34，每個 seat 已 meld 嘅牌
    discard_counts: list[list[int]]  # 4 x 34，每個 seat 牌河
    riichi: list[bool]  # 4
    dora_tiles: list[int]  # 現正生效嘅 dora 牌 type（可以有重複）
    round_wind: int
    seat_wind: int  # 揀緊嘢嗰個 seat 嘅自風
    discard_tile: int
    discard_is_red: bool
    is_riichi: bool
    is_tsumogiri: bool


def iter_full_discard_states(round_log: RoundLog, player_count: int):
    """同 iter_discard_decisions 一樣咁 replay，但連 4 個 seat 嘅公開資訊都track埋。"""
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
