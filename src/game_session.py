"""Phase 3 Part 1: wraps the jansou.game.environment.Environment generator.

Seat 0 is the human; their turn pauses to wait for the UI to pass back a
choice. Seats 1-3 use `SmartEfficiencyAgent` (jansou's built-in agent —
calls, riichis, and picks discards using shanten/acceptance, not randomly)
and resolve immediately on their own.

The full match (four east + four south rounds) is driven by
`Environment.play()` itself, which only stops once the whole match ends
(a GameResult) — we don't have to manage dealer rotation or scoring
ourselves in between.
"""

from dataclasses import dataclass, field

from jansou.core.rules import preset
from jansou.game.actions import Action
from jansou.game.agents import SmartEfficiencyAgent
from jansou.game.environment import DecisionRequest, Environment, GameResult
from jansou.game.events import Discard, Event, IndicatorReveal, RiichiAccepted, Ryuukyoku, ScoreChange, Win
from jansou.scoring.score import LimitTier

HUMAN_SEAT = 0

#: Proper display names for jansou's Yaku enum, cross-checked against the
#: full yaku list (https://zh.wikipedia.org/wiki/日本麻將的和牌牌型列表) —
#: naive title-casing of the enum name alone produces awkward results like
#: "Yakuhai Haku" instead of "Yakuhai (White Dragon)".
_YAKU_LABELS: dict[str, str] = {
    "RIICHI": "Riichi",
    "IPPATSU": "Ippatsu",
    "MENZEN_TSUMO": "Menzen Tsumo",
    "PINFU": "Pinfu",
    "IIPEIKOU": "Iipeikou",
    "TANYAO": "Tanyao",
    "YAKUHAI_HAKU": "Yakuhai (White Dragon)",
    "YAKUHAI_HATSU": "Yakuhai (Green Dragon)",
    "YAKUHAI_CHUN": "Yakuhai (Red Dragon)",
    "YAKUHAI_ROUND": "Yakuhai (Round Wind)",
    "YAKUHAI_SEAT": "Yakuhai (Seat Wind)",
    "HAITEI": "Haitei Raoyue",
    "HOUTEI": "Houtei Raoyui",
    "RINSHAN": "Rinshan Kaihou",
    "CHANKAN": "Chankan",
    "DOUBLE_RIICHI": "Double Riichi",
    "CHIITOITSU": "Chiitoitsu",
    "SANSHOKU_DOUJUN": "Sanshoku Doujun",
    "ITTSU": "Ittsu",
    "CHANTA": "Chanta",
    "TOITOI": "Toitoi",
    "SANANKOU": "Sanankou",
    "SANSHOKU_DOUKOU": "Sanshoku Doukou",
    "SANKANTSU": "Sankantsu",
    "SHOUSANGEN": "Shousangen",
    "HONROUTOU": "Honroutou",
    "HONITSU": "Honitsu",
    "JUNCHAN": "Junchan",
    "RYANPEIKOU": "Ryanpeikou",
    "CHINITSU": "Chinitsu",
    "KOKUSHI": "Kokushi Musou",
    "SUUANKOU": "Suuankou",
    "CHUUREN": "Chuuren Poutou",
    "DAISANGEN": "Daisangen",
    "SHOUSUUSHI": "Shousuushi",
    "DAISUUSHI": "Daisuushi",
    "TSUUIISOU": "Tsuuiisou",
    "CHINROUTOU": "Chinroutou",
    "RYUUIISOU": "Ryuuiisou",
    "SUUKANTSU": "Suukantsu",
    "TENHOU": "Tenhou",
    "CHIIHOU": "Chiihou",
}


def _yaku_label(name: str) -> str:
    return _YAKU_LABELS.get(name, name.replace("_", " ").title())


def _win_text(event: Win, tile_label) -> str:
    """A detailed win summary: how it was won, on which tile, every yaku
    that scored, the han/fu (or yakuman status), and the points gained —
    the hand-summary screen used to only show the han/fu total, which
    wasn't enough to understand what actually happened."""
    source = "tsumo" if event.from_seat is None else f"ron off seat {event.from_seat}"
    tile = tile_label(event.winning_tile.kind.value)
    r = event.result

    if r.is_yakuman:
        yaku_str = ", ".join(
            f"{_yaku_label(yv.yaku.name)}" + (f" x{yv.value}" if yv.value > 1 else "")
            for yv in r.yaku
        )
        value_str = f"Yakuman — {yaku_str}"
    else:
        yaku_str = ", ".join(f"{_yaku_label(yv.yaku.name)} ({yv.value})" for yv in r.yaku)
        limit_str = f" — {_yaku_label(r.limit.name)}!" if r.limit is not LimitTier.NONE else ""
        value_str = f"{r.han} han {r.fu.total} fu{limit_str} — {yaku_str}"

    return f"Seat {event.seat} wins by {source} with {tile}! {value_str} — +{r.payment.total:,} pts"


def _event_text(event: Event, tile_label) -> str | None:
    """Render an event as a human-readable line for the UI's event log.
    Returns None for events we don't care to display."""
    if isinstance(event, Discard):
        flag = " (riichi)" if event.riichi else ""
        return f"Seat {event.seat} discards {tile_label(event.tile.kind.value)}{flag}"
    if isinstance(event, RiichiAccepted):
        return f"Seat {event.seat} declares riichi"
    if isinstance(event, IndicatorReveal):
        return f"New dora indicator: {tile_label(event.tile.kind.value)}"
    if isinstance(event, Win):
        return _win_text(event, tile_label)
    if isinstance(event, Ryuukyoku):
        return f"Draw ({event.kind.name})"
    return None


def _score_change_text(event: ScoreChange) -> str:
    parts = [f"Seat {seat}: {'+' if delta >= 0 else ''}{delta} (now {score})" for seat, (delta, score) in
              enumerate(zip(event.deltas, event.scores))]
    return "Score changes — " + ", ".join(parts)


@dataclass
class GameSession:
    env: Environment
    bots: dict[int, SmartEfficiencyAgent]
    tile_label: callable
    game: object = None
    pending_request: DecisionRequest | None = None
    result: GameResult | None = None
    event_log: list[str] = field(default_factory=list)
    pending_summary: str | None = None
    _summary_lines: list[str] = field(default_factory=list, repr=False)

    def _fan_out(self, event: Event) -> None:
        for seat, bot in self.bots.items():
            bot.observe(event.mask_for(seat))
        text = _event_text(event, self.tile_label)
        if text:
            self.event_log.append(text)
        if isinstance(event, (Win, Ryuukyoku)) and text:
            self._summary_lines.append(text)
        elif isinstance(event, ScoreChange):
            self._summary_lines.append(_score_change_text(event))

    def start(self) -> None:
        self.game = self.env.play(observe=self._fan_out)
        self._advance(None)

    def choose(self, action: Action) -> None:
        self._advance(action)

    def acknowledge_summary(self) -> None:
        """User pressed "Continue" — clear the summary before showing the
        next hand's decision UI."""
        self.pending_summary = None

    def _advance(self, action: Action | None) -> None:
        """Advance until it's the human's (seat 0) turn to decide, or the
        whole match ends. If a hand ends along the way (win/draw), pause
        so the user can acknowledge the summary before moving on.
        """
        self._summary_lines = []
        try:
            request = next(self.game) if action is None else self.game.send(action)
        except StopIteration as stop:
            self.result = stop.value
            self.pending_request = None
            self._finalize_summary()
            return

        while request.seat != HUMAN_SEAT:
            bot_action = self.bots[request.seat].act(request.seat, request.kind, list(request.actions))
            try:
                request = self.game.send(bot_action)
            except StopIteration as stop:
                self.result = stop.value
                self.pending_request = None
                self._finalize_summary()
                return

        self.pending_request = request
        self._finalize_summary()

    def _finalize_summary(self) -> None:
        if self._summary_lines:
            self.pending_summary = "\n\n".join(self._summary_lines)


def new_session(tile_label, seed: int | None = None) -> GameSession:
    rules = preset("tenhou")
    env = Environment(rules, seed=seed)
    bots = {seat: SmartEfficiencyAgent(seed=seed) for seat in (1, 2, 3)}
    session = GameSession(env=env, bots=bots, tile_label=tile_label)
    session.start()
    return session
