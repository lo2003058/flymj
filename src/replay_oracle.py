"""Phase 2 Part 1: a "decide" callback that answers with whatever action
the historical log actually took, so jansou.game.flow.deal_steps can
replay a historical match step by step instead of playing randomly.

How it works: keeps a pointer into the event stream being replayed. Every
time jansou asks for a decision, it checks what the historical event
stream did at that point and picks the matching legal action. If no match
is found, it raises OracleMismatch, so validation immediately surfaces
exactly where the replay diverges instead of silently going wrong.
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
    """The historical record doesn't match what jansou offers as a legal
    action, or no corresponding event could be found."""


class HistoricalOracle:
    def __init__(self, round_log: RoundLog):
        self.events = round_log.events
        self.outcome = round_log.outcome
        self.ptr = 0

    def _peek(self):
        """Look at the next event, automatically skipping DoraReveal (the
        jansou engine emits these on its own — they're not a decision to
        make, we just need to skip past them to find the next actually
        relevant event)."""
        while self.ptr < len(self.events) and isinstance(self.events[self.ptr], DoraReveal):
            self.ptr += 1
        return self.events[self.ptr] if self.ptr < len(self.events) else None

    def _agari_list(self):
        return self.outcome if isinstance(self.outcome, tuple) else ()

    def _pick_type(self, actions: list[Action], cls: type) -> Action:
        for action in actions:
            if isinstance(action, cls):
                return action
        raise OracleMismatch(f"couldn't find a {cls.__name__} among actions={actions}")

    def decide(self, seat: int, kind: DecisionKind, actions: list[Action]) -> Action:
        if kind == DecisionKind.SELF:
            return self._decide_self(seat, actions)
        if kind == DecisionKind.DISCARD_REACTION:
            return self._decide_reaction(seat, actions)
        if kind in (DecisionKind.ROBBED_KAN, DecisionKind.NORTH_REACTION):
            return self._decide_rob(seat, actions)
        if kind == DecisionKind.TENPAI:
            return self._decide_tenpai(seat, actions)
        raise OracleMismatch(f"unhandled decision kind: {kind}")

    # --- SELF ---

    def _decide_self(self, seat: int, actions: list[Action]) -> Action:
        event = self._peek()
        if isinstance(event, Draw) and event.seat == seat:
            # Normal turn: a tile was just drawn. If this SELF decision
            # came right after a call (pon/chii) with no draw in between,
            # the event will already be the Discard itself.
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
                f"couldn't find a matching {cls.__name__}(tile={event.tile}, tsumogiri={event.tsumogiri}), actions={actions}"
            )

        # No more events belong to this seat: means this hand ended in
        # tsumo, or a nine-terminals abort.
        for agari in self._agari_list():
            if agari.winner == seat and agari.is_tsumo:
                return self._pick_type(actions, Tsumo)
        if isinstance(self.outcome, Ryuukyoku) and self.outcome.kind == "yao9":
            return self._pick_type(actions, NineTerminals)
        raise OracleMismatch(f"SELF seat={seat} found no matching historical action, next event={event!r}")

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
        raise OracleMismatch(f"couldn't find a matching kan action, meld={meld}, actions={actions}")

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
        raise OracleMismatch(f"couldn't find a matching {target_cls.__name__}, used={used_pair}, actions={actions}")

    # --- ROBBED_KAN / NORTH_REACTION (not yet seen in a known test round) ---

    def _decide_rob(self, seat: int, actions: list[Action]) -> Action:
        for agari in self._agari_list():
            if agari.winner == seat and not agari.is_tsumo:
                return self._pick_type(actions, Ron)
        return self._pick_type(actions, Pass)

    # --- TENPAI (declared tenpai at an abortive draw, untested) ---

    def _decide_tenpai(self, seat: int, actions: list[Action]) -> Action:
        declare = (
            isinstance(self.outcome, Ryuukyoku)
            and seat < len(self.outcome.tenpai)
            and self.outcome.tenpai[seat]
        )
        for action in actions:
            if isinstance(action, DeclareTenpai) and action.declare == declare:
                return action
        raise OracleMismatch(f"couldn't find a matching DeclareTenpai(declare={declare}), actions={actions}")
