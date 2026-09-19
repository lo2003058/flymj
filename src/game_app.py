"""Phase 3：可以一直玩落去嘅麻雀枱（果蠅代打）——你(seat 0) vs 3 個
SmartEfficiencyAgent bot。成鋪(東南各四局)由 jansou.game.environment.Environment
自己行，你嘅每個決策都有 Arm A（真 connectome）model 嘅建議做參考，但你
撳嘅永遠係 jansou 真正計出嚟嘅合法選項，唔會靠估。

跑法： uv run streamlit run src/game_app.py
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

st.set_page_config(page_title="果蠅代打", layout="wide")


@st.cache_resource
def load_everything():
    device = torch.device("mps")
    masks = np.load(MASKS_PATH)
    action_model = load_action_model(ACTION_MODEL_PATH, masks, device)
    coords = load_all_coords(masks)
    return action_model, masks, coords, device


def action_label(action) -> str:
    if isinstance(action, Riichi):
        tag = "（摸切）" if action.tsumogiri else ""
        return f"立直 + 掉 {TILE_LABELS[action.tile.kind.value]}{tag}"
    if isinstance(action, ActionDiscard):
        tag = "（摸切）" if action.tsumogiri else ""
        return f"掉 {TILE_LABELS[action.tile.kind.value]}{tag}"
    if isinstance(action, Tsumo):
        return "🎉 自摸！"
    if isinstance(action, Ron):
        return "🎉 榮和！"
    if isinstance(action, Pon):
        return f"碰（用 {TILE_LABELS[action.tiles[0].kind.value]}+{TILE_LABELS[action.tiles[1].kind.value]}）"
    if isinstance(action, Chii):
        return f"吃（用 {TILE_LABELS[action.tiles[0].kind.value]}+{TILE_LABELS[action.tiles[1].kind.value]}）"
    if isinstance(action, OpenKan):
        return "明槓"
    if isinstance(action, ClosedKan):
        return f"暗槓 {TILE_LABELS[action.kind.value]}"
    if isinstance(action, AddedKan):
        return f"加槓 {TILE_LABELS[action.tile.kind.value]}"
    if isinstance(action, NineTerminals):
        return "九種九牌（棄局）"
    if isinstance(action, Nuki):
        return "拔北"
    if isinstance(action, Pass):
        return "過"
    if isinstance(action, DeclareTenpai):
        return "宣言聽牌" if action.declare else "宣言未聽"
    return str(action)


def score_action(action, self_type_probs: dict, discard_probs: dict, reaction_probs: dict) -> float:
    """幫一個實際 legal action 攞返個 model 覺得幾好嘅分數，純粹排序/highlight 用。"""
    if isinstance(action, Riichi):
        return self_type_probs.get("RIICHI", 0.0) * discard_probs.get(action.tile.kind.value, 0.0)
    if isinstance(action, ActionDiscard):
        return self_type_probs.get("DISCARD", 0.0) * discard_probs.get(action.tile.kind.value, 0.0)
    if isinstance(action, Tsumo):
        return self_type_probs.get("TSUMO", 0.0) + 1.0  # 自摸永遠排最前
    if isinstance(action, Ron):
        return reaction_probs.get("RON", 0.0) + 1.0  # 食糊永遠排最前
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

.mj-seat-box { text-align: center; min-width: 150px; }
.mj-seat-name {
    color: #fff; font-weight: 700; font-size: 14px;
    background: rgba(0,0,0,0.35); border-radius: 6px; padding: 2px 8px; display: inline-block;
}
.mj-dealer-tag { color: #ffd66b; }
.mj-riichi-tag { color: #ff6b6b; }
.mj-seat-score { color: #ffe9a8; font-family: monospace; font-size: 13px; margin-top: 2px; }
.mj-river {
    display: flex; flex-wrap: wrap; gap: 2px; justify-content: center;
    max-width: 210px; margin: 6px auto 0;
}
.mj-melds { color: #cdeedd; font-size: 12px; margin-top: 4px; }

.mj-tile {
    display: inline-flex; align-items: center; justify-content: center;
    background: #fdfdf6; color: #222; border: 1px solid #8a8a7a; border-radius: 4px;
    box-shadow: 1px 2px 3px rgba(0,0,0,0.45);
    font-size: 16px; width: 22px; height: 30px; line-height: 1;
}
.mj-tile-sm { font-size: 12px; width: 16px; height: 22px; }
.mj-tile-lg { font-size: 30px; width: 40px; height: 56px; border-radius: 6px; margin: 2px; }

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
    font-size: 26px !important;
    min-width: 42px !important;
    height: 58px !important;
    background: #fdfdf6 !important;
    color: #222 !important;
    border: 1px solid #8a8a7a !important;
    border-radius: 6px !important;
    box-shadow: 1px 2px 3px rgba(0,0,0,0.45) !important;
    padding: 0 !important;
}
.st-key-hand-row button:disabled {
    opacity: 0.35 !important;
}
</style>
"""


def _clean_html(html: str) -> str:
    """摺埋成段做一行：留低跨行嘅縮排/空行會俾 Streamlit 嘅 markdown parser
    當做一個 blank line 提早收 HTML block，跟住手尾嘅 closing tag 就會變咗
    普通 markdown 文字（甚至觸發 code block），成個枱嘅 HTML 就散哂。"""
    html = re.sub(r"\s*\n\s*", " ", html.strip())
    return re.sub(r">\s+<", "><", html)


def _tile_span(glyph: str, size: str = "") -> str:
    cls = f"mj-tile {size}".strip()
    return f'<span class="{cls}">{glyph}</span>'


def _river_html(tiles: list[str], size: str = "mj-tile-sm") -> str:
    if not tiles:
        return '<div class="mj-river">（未出牌）</div>'
    return '<div class="mj-river">' + "".join(_tile_span(t, size) for t in tiles) + "</div>"


def _melds_html(state, seat: int) -> str:
    player = state.players[seat]
    if not player.melds:
        return ""
    melds_str = "　".join(" ".join(TILE_GLYPHS[t.kind.value] for t in m.tiles) for m in player.melds)
    return f'<div class="mj-melds">副露：{melds_str}</div>'


def _opponent_box_html(state, seat: int, label: str) -> str:
    player = state.players[seat]
    dealer = ' <span class="mj-dealer-tag">(莊)</span>' if seat == state.dealer else ""
    riichi = ' <span class="mj-riichi-tag">🀄立直</span>' if player.is_riichi else ""
    river_tiles = [TILE_GLYPHS[d.tile.kind.value] for d in player.discards]
    return _clean_html(f"""<div class="mj-seat-box">
        <div class="mj-seat-name">{label}{dealer}{riichi}</div>
        <div class="mj-seat-score">{state.scores[seat]:,} 分</div>
        {_river_html(river_tiles)}
        {_melds_html(state, seat)}
    </div>""")


def _center_box_html(state) -> str:
    dora_tiles = [TILE_GLYPHS[t.kind.value] for t in state.wall.dora_indicators]
    return _clean_html(f"""<div class="mj-center-box">
        <div class="mj-center-round">{WIND_NAMES[state.round_wind.value]}{state.round_number} 局　{state.honba} 本場</div>
        <div class="mj-center-row">牌山剩 {state.wall.live_draws_remaining}　|　供托 {state.deposit_pool // 1000} 本</div>
        <div class="mj-center-row">Dora 指示牌</div>
        {_river_html(dora_tiles)}
    </div>""")


def render_table(state) -> None:
    table_html = _clean_html(f"""<div class="mj-table"><div class="mj-grid">
        <div class="mj-top">{_opponent_box_html(state, 2, "對家 (seat 2)")}</div>
        <div class="mj-left">{_opponent_box_html(state, 3, "上家 (seat 3)")}</div>
        <div class="mj-center">{_center_box_html(state)}</div>
        <div class="mj-right">{_opponent_box_html(state, 1, "下家 (seat 1)")}</div>
    </div></div>""")
    st.markdown(table_html, unsafe_allow_html=True)


def render_human_panel(state) -> str:
    player = state.players[HUMAN_SEAT]
    dealer = ' <span class="mj-dealer-tag">(莊)</span>' if HUMAN_SEAT == state.dealer else ""
    riichi = ' <span class="mj-riichi-tag">🀄立直</span>' if player.is_riichi else ""
    river_tiles = [TILE_GLYPHS[d.tile.kind.value] for d in player.discards]
    return _clean_html(f"""<div class="mj-human-bar">
        <span class="mj-seat-name">你 (seat 0){dealer}{riichi}</span>
        &nbsp;&nbsp;<span class="mj-seat-score">{state.scores[HUMAN_SEAT]:,} 分</span>
        {_river_html(river_tiles, size="")}
        {_melds_html(state, HUMAN_SEAT)}
    </div>""")


def _hand_html(hand_by_suit: list[list]) -> str:
    groups = []
    for group in hand_by_suit:
        if not group:
            continue
        groups.append("".join(_tile_span(TILE_GLYPHS[t.kind.value], "mj-tile-lg") for t in group))
    return '<div class="mj-river" style="max-width:100%; gap:10px;">' + "".join(groups) + "</div>"


def main() -> None:
    st.title("🦟 果蠅代打")
    st.caption("你係 seat 0，另外三個係 jansou 自帶嘅 SmartEfficiencyAgent。你嘅每個決策都有 Arm A model 嘅建議。")
    st.markdown(_TABLE_CSS, unsafe_allow_html=True)

    action_model, masks, coords, device = load_everything()

    if "session" not in st.session_state:
        st.session_state.session = None
        st.session_state.seed = 42

    with st.sidebar:
        st.session_state.seed = st.number_input("開局 seed", value=st.session_state.seed, step=1)
        if st.button("🀄 開新局", type="primary"):
            st.session_state.session = new_session(lambda k: TILE_LABELS[k], seed=int(st.session_state.seed))
            st.rerun()
        with st.expander("最近事件"):
            if st.session_state.session:
                for line in st.session_state.session.event_log[-30:]:
                    st.caption(line)

    session = st.session_state.session
    if session is None:
        st.info("撳側邊欄「開新局」開始。")
        return

    if session.result is not None:
        st.header("🏁 Game 完")
        if session.pending_summary:
            st.info(session.pending_summary)
        for seat in range(4):
            st.write(f"Seat {seat}：{session.result.scores[seat]} 分（排名第 {session.result.ranking.index(seat) + 1}）")
        return

    if session.pending_summary is not None:
        st.header("📋 呢一局嘅總結")
        st.info(session.pending_summary)
        if st.button("➡️ 繼續（開始下一局）", type="primary"):
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
        st.write("**你嘅回合**：撳手牌打出邊隻，或者揀下面嘅特殊選項。")
        best_discard_tile = max(discard_probs, key=discard_probs.get) if discard_probs else None
        with st.container(key="hand-row"):
            hand_cols = st.columns(len(hand))
            for i, (col, t) in enumerate(zip(hand_cols, hand)):
                action = discard_actions.get((t.kind, t.red))
                prefix = "🌟" if t.kind.value == best_discard_tile else ""
                label = f"{prefix}{TILE_GLYPHS[t.kind.value]}"
                if action is not None:
                    if col.button(label, key=f"hand_{i}", help=TILE_LABELS[t.kind.value]):
                        session.choose(action)
                        st.rerun()
                    note = safety_label(snap["discard_counts"], t.kind.value, snap["riichi"])
                    if note:
                        col.caption(note)
                else:
                    col.button(label, key=f"hand_{i}", disabled=True, help="呢隻牌而家唔可以打")
    else:
        hand_by_suit: list[list] = [[], [], [], []]
        for t in hand:
            hand_by_suit[0 if t.kind.value < 9 else 1 if t.kind.value < 18 else 2 if t.kind.value < 27 else 3].append(t)
        st.markdown(_hand_html(hand_by_suit), unsafe_allow_html=True)
        st.write(f"**輪到你決策**（{request.kind.name}）：")

    if other_actions:
        if discard_actions:
            st.caption("特殊選項")
        n_cols = min(len(other_actions), 6)
        action_cols = st.columns(n_cols)
        for i, action in enumerate(other_actions):
            col = action_cols[i % n_cols]
            prefix = "🌟 " if i == 0 and not discard_actions else ""
            label = f"{prefix}{action_label(action)}"
            if col.button(label, key=f"action_{i}_{action!r}"):
                session.choose(action)
                st.rerun()

            if isinstance(action, Riichi):
                note = safety_label(snap["discard_counts"], action.tile.kind.value, snap["riichi"])
                if note:
                    col.caption(note)

    with st.expander("果蠅腦視覺化（呢個決策嘅 activation）"):
        render_brain_viz(coords, result.activations)


if __name__ == "__main__":
    main()
