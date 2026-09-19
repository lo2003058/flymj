"""Phase 3 Part 1：包住 jansou.game.environment.Environment 嘅 generator。

Seat 0 係人類，佢嘅回合停低等 UI 攞返個選擇；seat 1-3 用
`SmartEfficiencyAgent`（jansou 自帶，識叫牌、識用 shanten/acceptance 揀
discard，唔係亂咁打）自動即刻解決。

成鋪(東南各四局)由 Environment.play() 自己一路行落去，直到成場完
（GameResult）先停，中間唔使我哋自己管 dealer 輪替/計分。
"""

from dataclasses import dataclass, field

from jansou.core.rules import preset
from jansou.game.actions import Action
from jansou.game.agents import SmartEfficiencyAgent
from jansou.game.environment import DecisionRequest, Environment, GameResult
from jansou.game.events import Discard, Event, IndicatorReveal, RiichiAccepted, Ryuukyoku, ScoreChange, Win

HUMAN_SEAT = 0


def _event_text(event: Event, tile_label) -> str | None:
    """將一個 event 譯做人睇得明嘅一句（俾 UI 做事件記錄），冇特別想顯示嘅
    event 就回傳 None。"""
    if isinstance(event, Discard):
        flag = "（立直）" if event.riichi else ""
        return f"Seat {event.seat} 掉 {tile_label(event.tile.kind.value)}{flag}"
    if isinstance(event, RiichiAccepted):
        return f"Seat {event.seat} 立直成功"
    if isinstance(event, IndicatorReveal):
        return f"新 dora 指示牌：{tile_label(event.tile.kind.value)}"
    if isinstance(event, Win):
        source = "自摸" if event.from_seat is None else f"食糊 seat {event.from_seat}"
        return f"Seat {event.seat} {source}！{event.result.han} 飜 {event.result.fu.total} 符"
    if isinstance(event, Ryuukyoku):
        return f"流局（{event.kind.name}）"
    return None


def _score_change_text(event: ScoreChange) -> str:
    parts = [f"Seat {seat}：{'+' if delta >= 0 else ''}{delta}（而家 {score}）" for seat, (delta, score) in
              enumerate(zip(event.deltas, event.scores))]
    return "分數變化 — " + "，".join(parts)


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
        """使用者撳咗「繼續」，先俾佢睇返新一局嘅決策 UI。"""
        self.pending_summary = None

    def _advance(self, action: Action | None) -> None:
        """行落去，直至輪到人類（seat 0）決策，或者成鋪 game 完。
        中途如果有一局完咗（糊/流局），停低要使用者確認咗個總結先再問下一步。
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
