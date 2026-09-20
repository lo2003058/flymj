"""Phase 3: a mahjong table you can keep playing (fly-brain autoplay) — you
(seat 0) vs. 3 SmartEfficiencyAgent bots. The full match (four east + four
south rounds) is driven by jansou.game.environment.Environment itself.
Every one of your decisions comes with a suggestion from the Arm A (real
connectome) model, but what you actually press is always one of jansou's
genuinely computed legal options — never a guess.

Run: uv run streamlit run src/game_app.py
"""

import re

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
from jansou.game.actions import (
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

from action_inference import load_action_model, predict_action
from build_action_dataset import snapshot_state
from game_session import HUMAN_SEAT, new_session
from neuron_coords import load_all_coords
from safety import safety_label
from tile_ui import TILE_GLYPHS, TILE_LABELS

MASKS_PATH = "data/processed/masks.npz"
ACTION_MODEL_PATH = "data/processed/action_model_arm_a.pt"

WIND_NAMES = ["東", "南", "西", "北"]

st.set_page_config(page_title="FlyMJ", layout="wide")


@st.cache_resource
def load_everything():
    device = torch.device("mps")
    masks = np.load(MASKS_PATH)
    action_model = load_action_model(ACTION_MODEL_PATH, masks, device)
    coords = load_all_coords(masks)
    return action_model, masks, coords, device


def action_label(action) -> str:
    if isinstance(action, Riichi):
        tag = " (tsumogiri)" if action.tsumogiri else ""
        return f"Riichi + discard {TILE_LABELS[action.tile.kind.value]}{tag}"
    if isinstance(action, ActionDiscard):
        tag = " (tsumogiri)" if action.tsumogiri else ""
        return f"Discard {TILE_LABELS[action.tile.kind.value]}{tag}"
    if isinstance(action, Tsumo):
        return "🎉 Tsumo!"
    if isinstance(action, Ron):
        return "🎉 Ron!"
    if isinstance(action, Pon):
        return f"Pon (using {TILE_LABELS[action.tiles[0].kind.value]}+{TILE_LABELS[action.tiles[1].kind.value]})"
    if isinstance(action, Chii):
        return f"Chii (using {TILE_LABELS[action.tiles[0].kind.value]}+{TILE_LABELS[action.tiles[1].kind.value]})"
    if isinstance(action, OpenKan):
        return "Open kan"
    if isinstance(action, ClosedKan):
        return f"Closed kan {TILE_LABELS[action.kind.value]}"
    if isinstance(action, AddedKan):
        return f"Added kan {TILE_LABELS[action.tile.kind.value]}"
    if isinstance(action, NineTerminals):
        return "Nine terminals (abort)"
    if isinstance(action, Nuki):
        return "Nuki (draw North)"
    if isinstance(action, Pass):
        return "Pass"
    if isinstance(action, DeclareTenpai):
        return "Declare tenpai" if action.declare else "Declare not tenpai"
    return str(action)


def score_action(
    action, self_type_probs: dict, discard_probs: dict, reaction_probs: dict, *, for_ranking: bool = True
) -> float:
    """Look up how good the model thinks a legal action is. With
    `for_ranking=True` (the default), Tsumo/Ron get a +1.0 bonus so a
    winning move always sorts first; pass `for_ranking=False` to get the
    plain probability back for display (e.g. as a percentage)."""
    if isinstance(action, Riichi):
        return self_type_probs.get("RIICHI", 0.0) * discard_probs.get(action.tile.kind.value, 0.0)
    if isinstance(action, ActionDiscard):
        return self_type_probs.get("DISCARD", 0.0) * discard_probs.get(action.tile.kind.value, 0.0)
    if isinstance(action, Tsumo):
        base = self_type_probs.get("TSUMO", 0.0)
        return base + 1.0 if for_ranking else base
    if isinstance(action, Ron):
        base = reaction_probs.get("RON", 0.0)
        return base + 1.0 if for_ranking else base
    if isinstance(action, (ClosedKan, AddedKan)):
        return self_type_probs.get("KAN", 0.0)
    if isinstance(action, NineTerminals):
        return self_type_probs.get("KYUUSHU", 0.0)
    if isinstance(action, Pon):
        return reaction_probs.get("PON", 0.0)
    if isinstance(action, Chii):
        return reaction_probs.get("CHII", 0.0)
    if isinstance(action, OpenKan):
        return reaction_probs.get("OPEN_KAN", 0.0)
    if isinstance(action, Pass):
        return reaction_probs.get("PASS", 0.0)
    return 0.0


def render_brain_viz(coords: dict, activations: dict) -> None:
    fig = go.Figure()
    for name, key, size in [("PN", "pn", 7), ("KC", "kc", 3), ("MBON", "mbon", 9)]:
        xyz = coords[key]
        act = np.clip(activations[key], 0, None)
        valid = ~np.isnan(xyz).any(axis=1)
        act_max = act.max() if act.max() > 0 else 1.0
        fig.add_trace(
            go.Scatter(
                x=xyz[valid, 0],
                y=xyz[valid, 1],
                mode="markers",
                marker=dict(
                    size=size, color=(act[valid] / act_max), colorscale="Hot", cmin=0, cmax=1,
                    showscale=(key == "kc"),
                ),
                name=name,
            )
        )
    fig.update_layout(height=450, yaxis={"scaleanchor": "x"}, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig, width="stretch")


_TABLE_CSS = """
<style>
.mj-table {
    position: relative;
    background: radial-gradient(ellipse at center, #1d7a53 0%, #0b3d28 75%);
    border: 6px solid #6b4423;
    border-radius: 24px;
    padding: 22px;
    margin-bottom: 18px;
}
.mj-grid {
    display: grid;
    grid-template-columns: 1fr 2fr 1fr;
    grid-template-areas:
        ".    top   ."
        "left center right";
    gap: 14px;
    align-items: center;
}
.mj-top { grid-area: top; justify-self: center; }
.mj-left { grid-area: left; }
.mj-right { grid-area: right; }
.mj-center { grid-area: center; justify-self: center; }

.mj-seat-box { text-align: center; min-width: 210px; }
.mj-seat-name {
    color: #fff; font-weight: 700; font-size: 14px;
    background: rgba(0,0,0,0.35); border-radius: 6px; padding: 2px 8px; display: inline-block;
}
.mj-dealer-tag { color: #ffd66b; }
.mj-riichi-tag { color: #ff6b6b; }
.mj-seat-score { color: #ffe9a8; font-family: monospace; font-size: 13px; margin-top: 2px; }
.mj-river {
    display: flex; flex-wrap: wrap; gap: 3px; justify-content: center;
    max-width: 290px; margin: 6px auto 0;
}
.mj-melds { color: #cdeedd; font-size: 13px; margin-top: 4px; }

.mj-tile {
    display: inline-flex; align-items: center; justify-content: center;
    background: #fdfdf6; color: #222; border: 1px solid #8a8a7a; border-radius: 4px;
    box-shadow: 1px 2px 3px rgba(0,0,0,0.45);
    font-size: 26px; width: 44px; height: 60px; line-height: 1; overflow: hidden;
}
.mj-tile-sm { font-size: 22px; width: 38px; height: 52px; }
.mj-tile-lg { font-size: 36px; width: 58px; height: 80px; border-radius: 6px; margin: 2px; }

.mj-glyph {
    display: inline-block;
    transform: scale(1.5);
}

.mj-center-box {
    background: #0c2d1f; border: 3px solid #d4af37; border-radius: 14px;
    padding: 14px 22px; text-align: center; color: #fff; min-width: 170px;
}
.mj-center-round { font-size: 18px; font-weight: 700; color: #ffe9a8; }
.mj-center-row { font-size: 12px; color: #cdeedd; margin-top: 5px; }

.mj-human-bar {
    background: rgba(0,0,0,0.25); border-radius: 10px; padding: 10px 16px;
    color: #fff; margin-bottom: 8px;
}

.st-key-hand-row button {
    font-size: 32px !important;
    min-width: 48px !important;
    height: 66px !important;
    background: #fdfdf6 !important;
    color: #222 !important;
    border: 1px solid #8a8a7a !important;
    border-radius: 6px !important;
    box-shadow: 1px 2px 3px rgba(0,0,0,0.45) !important;
    padding: 0 !important;
    transform: scale(1.4);
}
.st-key-hand-row button:disabled {
    opacity: 0.35 !important;
}

.st-key-best-action button {
    border: 3px solid #39ff14 !important;
    box-shadow: 0 0 10px 2px rgba(57, 255, 20, 0.65) !important;
}

.st-key-drawn-tile button {
    border: 3px solid #4fc3f7 !important;
    box-shadow: 0 0 10px 2px rgba(79, 195, 247, 0.65) !important;
}

.mj-tile-current {
    border: 3px solid #ffca28 !important;
    box-shadow: 0 0 8px 3px rgba(255, 202, 40, 0.7) !important;
}
</style>
"""


def _clean_html(html: str) -> str:
    """Collapse everything to one line: leaving line breaks/indentation
    across lines makes Streamlit's markdown parser treat a blank line as
    ending the raw HTML block early, turning the trailing closing tags
    into plain markdown text (even a code block) and breaking the whole
    table's HTML."""
    html = re.sub(r"\s*\n\s*", " ", html.strip())
    return re.sub(r">\s+<", "><", html)


def _tile_span(kind: int, size: str = "") -> str:
    """Renders one tile glyph with a hover tooltip (its plain-text name),
    since the small Unicode tile glyphs alone can be hard to tell apart."""
    cls = f"mj-tile {size}".strip()
    return f'<span class="{cls}" title="{TILE_LABELS[kind]}"><span class="mj-glyph">{TILE_GLYPHS[kind]}</span></span>'


def _river_html(kinds: list[int], size: str = "mj-tile-sm", highlight_last: bool = False) -> str:
    """highlight_last marks the most recent discard in this river — the
    tile currently "in play" — so it's easy to spot at a glance."""
    if not kinds:
        return '<div class="mj-river">(no discards yet)</div>'
    spans = []
    for i, k in enumerate(kinds):
        extra = "mj-tile-current" if (highlight_last and i == len(kinds) - 1) else ""
        spans.append(_tile_span(k, f"{size} {extra}".strip()))
    return '<div class="mj-river">' + "".join(spans) + "</div>"


def _melds_html(state, seat: int) -> str:
    player = state.players[seat]
    if not player.melds:
        return ""
    melds_str = "　".join(" ".join(_tile_span(t.kind.value) for t in m.tiles) for m in player.melds)
    return f'<div class="mj-melds">Melds: {melds_str}</div>'


def _opponent_box_html(state, seat: int, label: str) -> str:
    player = state.players[seat]
    dealer = ' <span class="mj-dealer-tag">(Dealer)</span>' if seat == state.dealer else ""
    riichi = ' <span class="mj-riichi-tag">🀄 Riichi</span>' if player.is_riichi else ""
    river_kinds = [d.tile.kind.value for d in player.discards]
    is_last_discarder = state.last_discard is not None and state.last_discard[0] == seat
    return _clean_html(f"""<div class="mj-seat-box">
        <div class="mj-seat-name">{label}{dealer}{riichi}</div>
        <div class="mj-seat-score">{state.scores[seat]:,} pts</div>
        {_river_html(river_kinds, highlight_last=is_last_discarder)}
        {_melds_html(state, seat)}
    </div>""")


def _center_box_html(state) -> str:
    dora_kinds = [t.kind.value for t in state.wall.dora_indicators]
    return _clean_html(f"""<div class="mj-center-box">
        <div class="mj-center-round">{WIND_NAMES[state.round_wind.value]} {state.round_number}　Honba {state.honba}</div>
        <div class="mj-center-row">Tiles left {state.wall.live_draws_remaining}　|　Riichi sticks {state.deposit_pool // 1000}</div>
        <div class="mj-center-row">Dora indicator(s)</div>
        {_river_html(dora_kinds)}
    </div>""")


def render_table(state) -> None:
    table_html = _clean_html(f"""<div class="mj-table"><div class="mj-grid">
        <div class="mj-top">{_opponent_box_html(state, 2, "Across (seat 2)")}</div>
        <div class="mj-left">{_opponent_box_html(state, 3, "Left (seat 3)")}</div>
        <div class="mj-center">{_center_box_html(state)}</div>
        <div class="mj-right">{_opponent_box_html(state, 1, "Right (seat 1)")}</div>
    </div></div>""")
    st.markdown(table_html, unsafe_allow_html=True)


def render_human_panel(state) -> str:
    player = state.players[HUMAN_SEAT]
    dealer = ' <span class="mj-dealer-tag">(Dealer)</span>' if HUMAN_SEAT == state.dealer else ""
    riichi = ' <span class="mj-riichi-tag">🀄 Riichi</span>' if player.is_riichi else ""
    river_kinds = [d.tile.kind.value for d in player.discards]
    is_last_discarder = state.last_discard is not None and state.last_discard[0] == HUMAN_SEAT
    return _clean_html(f"""<div class="mj-human-bar">
        <span class="mj-seat-name">You (seat 0){dealer}{riichi}</span>
        &nbsp;&nbsp;<span class="mj-seat-score">{state.scores[HUMAN_SEAT]:,} pts</span>
        {_river_html(river_kinds, size="", highlight_last=is_last_discarder)}
        {_melds_html(state, HUMAN_SEAT)}
    </div>""")


def _hand_html(hand_by_suit: list[list]) -> str:
    groups = []
    for group in hand_by_suit:
        if not group:
            continue
        groups.append("".join(_tile_span(t.kind.value, "mj-tile-lg") for t in group))
    return '<div class="mj-river" style="max-width:100%; gap:10px;">' + "".join(groups) + "</div>"


def main() -> None:
    st.title("🦟 FlyMJ")
    st.caption("You're seat 0; the other three are jansou's built-in SmartEfficiencyAgent. "
               "Every one of your decisions comes with a suggestion from the Arm A model.")
    st.markdown(_TABLE_CSS, unsafe_allow_html=True)

    action_model, masks, coords, device = load_everything()

    if "session" not in st.session_state:
        st.session_state.session = None
        st.session_state.seed = 42

    with st.sidebar:
        if st.button("🎲 New game (random)", type="primary"):
            st.session_state.seed = int(np.random.default_rng().integers(0, 1_000_000))
            st.session_state.session = new_session(lambda k: TILE_LABELS[k], seed=st.session_state.seed)
            st.rerun()
        with st.expander("Replay a specific seed"):
            st.session_state.seed = st.number_input("Seed", value=st.session_state.seed, step=1)
            if st.button("🀄 Start with this seed"):
                st.session_state.session = new_session(lambda k: TILE_LABELS[k], seed=int(st.session_state.seed))
                st.rerun()
        with st.expander("Recent events"):
            if st.session_state.session:
                for line in st.session_state.session.event_log[-30:]:
                    st.caption(line)

    session = st.session_state.session
    if session is None:
        st.info('Click "New game" in the sidebar to start.')
        return

    if session.result is not None:
        st.header("🏁 Game over")
        if session.pending_summary:
            st.info(session.pending_summary)
        for seat in range(4):
            st.write(f"Seat {seat}: {session.result.scores[seat]} pts (rank #{session.result.ranking.index(seat) + 1})")
        return

    if session.pending_summary is not None:
        st.header("📋 Hand summary")
        st.info(session.pending_summary)
        if st.button("➡️ Continue (next hand)", type="primary"):
            session.acknowledge_summary()
            st.rerun()
        return

    state = session.env.state
    request = session.pending_request

    render_table(state)
    st.markdown(render_human_panel(state), unsafe_allow_html=True)

    hand = sorted(state.players[HUMAN_SEAT].as_hand(include_drawn=True).concealed, key=lambda t: t.sort_key)

    is_reaction = request.kind == DecisionKind.DISCARD_REACTION
    snap = snapshot_state(state, HUMAN_SEAT, state.player_count, is_reaction=is_reaction)
    predict_kwargs = {k: v for k, v in snap.items() if k != "trigger_seat"}
    result = predict_action(action_model, device, seat=HUMAN_SEAT, **predict_kwargs)

    self_type_probs = dict(result.self_type_ranked)
    discard_probs = dict(result.discard_ranked_hand_tiles)
    reaction_probs = dict(result.reaction_ranked)

    discard_actions: dict[tuple, ActionDiscard] = {}
    for a in request.actions:
        if isinstance(a, ActionDiscard):
            discard_actions.setdefault((a.tile.kind, a.tile.red), a)

    other_actions = sorted(
        (a for a in request.actions if not isinstance(a, ActionDiscard)),
        key=lambda a: score_action(a, self_type_probs, discard_probs, reaction_probs),
        reverse=True,
    )

    if discard_actions:
        st.write("**Your turn**: tap a hand tile to discard it, or pick a special option below. "
                 "The model's top pick is outlined in green; the tile you just drew is outlined in blue; "
                 "confidence is shown below each tile.")
        best_discard_tile = max(discard_probs, key=discard_probs.get) if discard_probs else None
        best_highlighted = False
        drawn_tile = state.players[HUMAN_SEAT].drawn
        drawn_key = (drawn_tile.kind, drawn_tile.red) if drawn_tile is not None else None
        drawn_highlighted = False
        with st.container(key="hand-row"):
            hand_cols = st.columns(len(hand))
            for i, (col, t) in enumerate(zip(hand_cols, hand)):
                action = discard_actions.get((t.kind, t.red))
                # Only the first matching tile gets the highlight container —
                # duplicate hand tiles of the same kind would otherwise both
                # try to register the same container key. Matched by value
                # (kind, red), not object identity, since jansou may hand
                # back a different Tile instance for what's logically the
                # same physical tile.
                is_best = t.kind.value == best_discard_tile and not best_highlighted
                if is_best:
                    best_highlighted = True
                is_drawn = drawn_key is not None and (t.kind, t.red) == drawn_key and not drawn_highlighted
                if is_drawn:
                    drawn_highlighted = True
                target = col
                if is_best:
                    target = target.container(key="best-action")
                if is_drawn:
                    target = target.container(key="drawn-tile")
                label = TILE_GLYPHS[t.kind.value]
                pct = discard_probs.get(t.kind.value, 0.0) * 100
                if action is not None:
                    if target.button(label, key=f"hand_{i}", help=TILE_LABELS[t.kind.value]):
                        session.choose(action)
                        st.rerun()
                    col.caption(f"{pct:.1f}%")
                    note = safety_label(snap["discard_counts"], t.kind.value, snap["riichi"])
                    if note:
                        col.caption(note)
                else:
                    target.button(label, key=f"hand_{i}", disabled=True, help="This tile can't be discarded right now")
    else:
        hand_by_suit: list[list] = [[], [], [], []]
        for t in hand:
            hand_by_suit[0 if t.kind.value < 9 else 1 if t.kind.value < 18 else 2 if t.kind.value < 27 else 3].append(t)
        st.markdown(_hand_html(hand_by_suit), unsafe_allow_html=True)
        st.write(f"**Your decision** ({request.kind.name}):")

    if other_actions:
        if discard_actions:
            st.caption("Special options")
        n_cols = min(len(other_actions), 6)
        action_cols = st.columns(n_cols)
        for i, action in enumerate(other_actions):
            col = action_cols[i % n_cols]
            is_best = i == 0 and not discard_actions
            target = col.container(key="best-action") if is_best else col
            label = action_label(action)
            if target.button(label, key=f"action_{i}_{action!r}"):
                session.choose(action)
                st.rerun()
            pct = score_action(action, self_type_probs, discard_probs, reaction_probs, for_ranking=False) * 100
            col.caption(f"{pct:.1f}%")

            if isinstance(action, Riichi):
                note = safety_label(snap["discard_counts"], action.tile.kind.value, snap["riichi"])
                if note:
                    col.caption(note)

    with st.expander("Fly-brain visualization (this decision's activations)"):
        render_brain_viz(coords, result.activations)


if __name__ == "__main__":
    main()
