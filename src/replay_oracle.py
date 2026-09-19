"""Phase 2 Part 1：一個「decide」callback，答返歷史牌譜實際做過嘅 action，
等 jansou.game.flow.deal_steps 可以逐步重演一場歷史牌局，唔使真係隨機行棋。

做法：維持一個行緊嘅 event pointer，jansou 問到邊個 decision，就睇返歷史
event stream 嗰陣做咗咩，揀返 legal actions 入面對應嗰個。揾唔到就拋
OracleMismatch，等驗證嗰陣即刻發現係邊度對唔上，唔會靜雞雞行錯。
"""

from jansou.core.hand import MeldType
from jansou.game.actions import (
    Action,
    AddedKan,
    Chii,
    ClosedKan,
    DeclareTenpai,
    NineTerminals,
    Nuki,
    OpenKan,
    Pass,
    Pon,
    Riichi,
    Ron,
    Tsumo,
)
from jansou.game.actions import Discard as ActionDiscard
from jansou.game.flow import DecisionKind
from jansou.io.paifu import Call, Discard as PaifuDiscard, DoraReveal, Draw, Kita, Ryuukyoku, RoundLog


class OracleMismatch(Exception):
    """歷史記錄同 jansou 提供嘅 legal action 對唔上，或者揾唔到對應嘅 event。"""


class HistoricalOracle:
    def __init__(self, round_log: RoundLog):
        self.events = round_log.events
        self.outcome = round_log.outcome
        self.ptr = 0

    def _peek(self):
        """睇下個 event，自動跳過 DoraReveal（jansou 個 engine 自己會 emit
        返呢啲，唔係一個要揀嘅決策，我哋淨係要跳過佢搵返下一個真正相關嘅 event）。"""
        while self.ptr < len(self.events) and isinstance(self.events[self.ptr], DoraReveal):
            self.ptr += 1
        return self.events[self.ptr] if self.ptr < len(self.events) else None

    def _agari_list(self):
        return self.outcome if isinstance(self.outcome, tuple) else ()

    def _pick_type(self, actions: list[Action], cls: type) -> Action:
        for action in actions:
            if isinstance(action, cls):
                return action
        raise OracleMismatch(f"actions={actions} 入面揾唔到 {cls.__name__}")

    def decide(self, seat: int, kind: DecisionKind, actions: list[Action]) -> Action:
        if kind == DecisionKind.SELF:
            return self._decide_self(seat, actions)
        if kind == DecisionKind.DISCARD_REACTION:
            return self._decide_reaction(seat, actions)
        if kind in (DecisionKind.ROBBED_KAN, DecisionKind.NORTH_REACTION):
            return self._decide_rob(seat, actions)
        if kind == DecisionKind.TENPAI:
            return self._decide_tenpai(seat, actions)
        raise OracleMismatch(f"未處理嘅 decision kind: {kind}")

    # --- SELF ---

    def _decide_self(self, seat: int, actions: list[Action]) -> Action:
        event = self._peek()
        if isinstance(event, Draw) and event.seat == seat:
            # 正常回合：先摸咗張牌。如果啱啱叫咗嘢（pon/chii）先嚟到呢個
            # SELF decision，中間冇摸牌，event 會直接係嗰個 Discard。
            self.ptr += 1
            event = self._peek()

        if isinstance(event, Call) and event.seat == seat and event.meld.type in (MeldType.ANKAN, MeldType.SHOUMINKAN):
            self.ptr += 1
            return self._match_self_kan(actions, event.meld)

        if isinstance(event, Kita) and event.seat == seat:
            self.ptr += 1
            return self._pick_type(actions, Nuki)

        if isinstance(event, PaifuDiscard) and event.seat == seat:
            self.ptr += 1
            cls = Riichi if event.riichi else ActionDiscard
            for action in actions:
                if isinstance(action, cls) and action.tile == event.tile and action.tsumogiri == event.tsumogiri:
                    return action
            raise OracleMismatch(
                f"揾唔到啱嘅 {cls.__name__}(tile={event.tile}, tsumogiri={event.tsumogiri})，actions={actions}"
            )

        # 冇更多 event 屬於呢個 seat：即係話呢鋪 tsumo 咗，或者九種九牌棄局
        for agari in self._agari_list():
            if agari.winner == seat and agari.is_tsumo:
                return self._pick_type(actions, Tsumo)
        if isinstance(self.outcome, Ryuukyoku) and self.outcome.kind == "yao9":
            return self._pick_type(actions, NineTerminals)
        raise OracleMismatch(f"SELF seat={seat} 揾唔到對應嘅歷史行為，下個 event={event!r}")

    def _match_self_kan(self, actions: list[Action], meld) -> Action:
        if meld.type is MeldType.ANKAN:
            kind = meld.tiles[0].kind
            for action in actions:
                if isinstance(action, ClosedKan) and action.kind == kind:
                    return action
        else:  # SHOUMINKAN
            for action in actions:
                if isinstance(action, AddedKan) and action.tile == meld.added:
                    return action
        raise OracleMismatch(f"揾唔到啱嘅 kan action，meld={meld}, actions={actions}")

    # --- DISCARD_REACTION ---

    def _decide_reaction(self, seat: int, actions: list[Action]) -> Action:
        event = self._peek()

        if isinstance(event, Call) and event.seat == seat:
            self.ptr += 1
            return self._match_call(actions, event.meld)

        if event is None:
            for agari in self._agari_list():
                if agari.winner == seat and not agari.is_tsumo:
                    return self._pick_type(actions, Ron)

        return self._pick_type(actions, Pass)

    def _match_call(self, actions: list[Action], meld) -> Action:
        used = list(meld.tiles)
        used.remove(meld.called)

        if meld.type is MeldType.DAIMINKAN:
            return self._pick_type(actions, OpenKan)

        used_pair = tuple(sorted(used[:2]))
        target_cls = Pon if meld.type is MeldType.PON else Chii
        for action in actions:
            if isinstance(action, target_cls) and tuple(sorted(action.tiles)) == used_pair:
                return action
        raise OracleMismatch(f"揾唔到啱嘅 {target_cls.__name__}，used={used_pair}, actions={actions}")

    # --- ROBBED_KAN / NORTH_REACTION（未喺已知測試 round 出現過）---

    def _decide_rob(self, seat: int, actions: list[Action]) -> Action:
        for agari in self._agari_list():
            if agari.winner == seat and not agari.is_tsumo:
                return self._pick_type(actions, Ron)
        return self._pick_type(actions, Pass)

    # --- TENPAI（流局公示聽牌，未測試過）---

    def _decide_tenpai(self, seat: int, actions: list[Action]) -> Action:
        declare = (
            isinstance(self.outcome, Ryuukyoku)
            and seat < len(self.outcome.tenpai)
            and self.outcome.tenpai[seat]
        )
        for action in actions:
            if isinstance(action, DeclareTenpai) and action.declare == declare:
                return action
        raise OracleMismatch(f"揾唔到啱嘅 DeclareTenpai(declare={declare})，actions={actions}")
