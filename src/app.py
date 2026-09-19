"""Phase 1：本機麻雀掉牌 advisor（用真果蠅 connectome 拓撲嘅 Arm A model）。

跑法： uv run streamlit run src/app.py

淨係識答「呢手牌應該掉邊隻」，唔識叫牌/立直/食糊（Arm A model 訓練數據
淨係得 DISCARD/RIICHI_DISCARD，見 data/doc/writeup.md）。
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
from jansou.core.tiles import TileKind

from action_inference import load_action_model, predict_action
from call_options import chii_combos, open_kan_available, pon_available
from inference import build_state, load_model, predict
from neuron_coords import load_all_coords
from safety import safety_label
from tile_ui import FIVE_KINDS, TILE_GLYPHS, TILE_LABELS

MASKS_PATH = "data/processed/masks.npz"
MODEL_PATH = "data/processed/model_arm_a.pt"
ACTION_MODEL_PATH = "data/processed/action_model_arm_a.pt"

WIND_LABELS = ["東", "南", "西", "北"]


def _hand_from_kinds(kinds: list[int]) -> list[int]:
    counts = [0] * 34
    for k in kinds:
        counts[k] += 1
    return counts


#: 撳一撳就成手牌入哂去嘅例子，唔使逐隻撳。
EXAMPLE_HANDS: dict[str, list[int]] = {
    "明顯字牌（123456789萬 + 東東東 + 南南）": _hand_from_kinds(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 27, 27, 27, 28, 28]
    ),
    "清一色進行中 + 一隻雜牌（11 234 567 789 99萬 + 1筒）": _hand_from_kinds(
        [0, 0, 1, 2, 3, 4, 5, 6, 6, 7, 8, 8, 8, 9]
    ),
    "亂七八糟嘅開局（成手唔連唔對）": _hand_from_kinds(
        [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 28, 29, 30, 31]
    ),
    "已經聽牌（雙碰聽白/發）": _hand_from_kinds(
        [0, 1, 2, 3, 13, 14, 15, 24, 25, 26, 31, 31, 32, 32]
    ),
}

st.set_page_config(page_title="果蠅腦麻雀 Advisor", layout="wide")


@st.cache_resource
def load_everything():
    device = torch.device("mps")
    masks = np.load(MASKS_PATH)
    model = load_model(MODEL_PATH, masks, device)
    action_model = load_action_model(ACTION_MODEL_PATH, masks, device)
    coords = load_all_coords(masks)
    return model, action_model, masks, coords, device


def init_state() -> None:
    defaults = {
        "hand_counts": [0] * 34,
        "hand_red": dict.fromkeys(FIVE_KINDS, False),
        "discard_counts": [[0] * 34 for _ in range(4)],
        "meld_counts": [[0] * 34 for _ in range(4)],
        "riichi": [False] * 4,
        "dora_indicator": "(冇)",
        "round_wind": 0,
        "seat_wind": 0,
        "result": None,
        "chosen_discard": None,
        "decision_mode": "自己回合（啱啱摸咗牌）",
        "trigger_seat_rel": 1,
        "trigger_tile_label": TILE_LABELS[0],
        "action_result": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_tile_picker(counts: list[int], key_prefix: str, max_count: int = 4) -> None:
    """畫 34 個 tile button，撳一下加一隻（去到 max_count 就返去 0）。"""
    groups = [("萬", range(0, 9)), ("筒", range(9, 18)), ("索", range(18, 27)), ("字", range(27, 34))]
    for group_name, rng in groups:
        cols = st.columns(9)
        for i, kind in enumerate(rng):
            label = f"{TILE_GLYPHS[kind]}\n{counts[kind]}"
            if cols[i].button(label, key=f"{key_prefix}_{kind}"):
                counts[kind] = (counts[kind] + 1) % (max_count + 1)


def render_brain_viz(coords: dict[str, np.ndarray], activations: dict[str, np.ndarray]) -> None:
    fig = go.Figure()
    layer_specs = [("PN", "pn", 7), ("KC", "kc", 3), ("MBON", "mbon", 9)]
    for name, key, size in layer_specs:
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
                    size=size,
                    color=(act[valid] / act_max),
                    colorscale="Hot",
                    cmin=0,
                    cmax=1,
                    showscale=(key == "kc"),
                    colorbar=dict(title="activation") if key == "kc" else None,
                ),
                name=f"{name} ({valid.sum()})",
                hovertext=[f"{name} activation={a:.3f}" for a in act[valid]],
            )
        )
    fig.update_layout(
        title="果蠅腦活躍程度（真實神經元位置，顏色 = activation 強度）",
        xaxis_title="X（真實 soma 座標）",
        yaxis_title="Y（真實 soma 座標）",
        yaxis={"scaleanchor": "x"},
        height=600,
        legend={"orientation": "h"},
    )
    st.plotly_chart(fig, width="stretch")


def main() -> None:
    st.title("🦟 果蠅腦麻雀掉牌 Advisor（Arm A：真 connectome 拓撲）")
    st.caption("淨係識答「呢手牌應該掉邊隻」，唔包括叫牌/立直/食糊決策。")

    model, action_model, masks, coords, device = load_everything()
    init_state()

    with st.sidebar:
        st.header("場面資訊")
        st.session_state.round_wind = WIND_LABELS.index(
            st.selectbox("場風", WIND_LABELS, index=st.session_state.round_wind)
        )
        st.session_state.seat_wind = WIND_LABELS.index(
            st.selectbox("自風（你）", WIND_LABELS, index=st.session_state.seat_wind)
        )
        dora_options = ["(冇)"] + TILE_LABELS
        st.session_state.dora_indicator = st.selectbox(
            "Dora 指示牌", dora_options, index=dora_options.index(st.session_state.dora_indicator)
        )

        st.subheader("Riichi 狀態")
        seat_names = ["你 (seat 0)", "下家 (seat 1)", "對家 (seat 2)", "上家 (seat 3)"]
        for i, name in enumerate(seat_names):
            st.session_state.riichi[i] = st.checkbox(name, value=st.session_state.riichi[i], key=f"riichi_{i}")

        if st.button("🔄 全部重設"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    st.subheader("快速例子（撳一撳成手牌就入哂去）")
    example_cols = st.columns(len(EXAMPLE_HANDS))
    for col, (name, hand) in zip(example_cols, EXAMPLE_HANDS.items()):
        if col.button(name, key=f"example_{name}"):
            st.session_state.hand_counts = list(hand)
            st.session_state.result = None
            st.session_state.chosen_discard = None
            st.rerun()

    st.subheader("你嘅手牌")
    render_tile_picker(st.session_state.hand_counts, "hand")
    hand_total = sum(st.session_state.hand_counts)
    st.write(f"手牌總數：{hand_total} 隻（正常應該係 13 或 14，call 咗嘢會少啲）")

    red_cols = st.columns(len(FIVE_KINDS))
    for col, kind in zip(red_cols, FIVE_KINDS):
        st.session_state.hand_red[kind] = col.checkbox(
            f"{TILE_LABELS[kind]} 係紅五", value=st.session_state.hand_red[kind], key=f"red_{kind}"
        )

    st.divider()
    st.subheader("🀄 全部人嘅牌河（呢個對推薦好關鍵，唔好漏咗）")
    st.caption("Model 同安全度分析都靠呢度嘅資料嚟判斷邊隻牌人哋掉過、邊隻危險——留空即係當「乜都未掉過」，會令推薦唔準。")
    seat_labels = ["你自己", "下家 (seat 1)", "對家 (seat 2)", "上家 (seat 3)"]
    discard_tabs = st.tabs(seat_labels)
    for seat, tab in enumerate(discard_tabs):
        with tab:
            render_tile_picker(st.session_state.discard_counts[seat], f"discard_{seat}")

    with st.expander("已經 call 咗嘅牌（meld，四位都喺呢度）"):
        for seat, label in enumerate(seat_labels):
            st.write(f"**{label}**：")
            render_tile_picker(st.session_state.meld_counts[seat], f"meld_{seat}")

    st.divider()

    st.subheader("Phase 2：而家係咩情況？")
    mode_options = ["自己回合（啱啱摸咗牌）", "反應叫牌（有人啱啱掉咗隻牌）"]
    st.session_state.decision_mode = st.radio(
        "決策類型", mode_options, index=mode_options.index(st.session_state.decision_mode), horizontal=True
    )
    is_reaction_mode = st.session_state.decision_mode == mode_options[1]

    if is_reaction_mode:
        rcol1, rcol2 = st.columns(2)
        seat_rel_options = {"下家 (seat 1)": 1, "對家 (seat 2)": 2, "上家 (seat 3)": 3}
        chosen_seat_label = rcol1.selectbox(
            "邊個掉嘅",
            list(seat_rel_options.keys()),
            index=list(seat_rel_options.values()).index(st.session_state.trigger_seat_rel),
        )
        st.session_state.trigger_seat_rel = seat_rel_options[chosen_seat_label]
        st.session_state.trigger_tile_label = rcol2.selectbox(
            "掉咗邊隻", TILE_LABELS, index=TILE_LABELS.index(st.session_state.trigger_tile_label)
        )

    st.divider()

    if st.button("🔍 分析", type="primary", disabled=hand_total == 0):
        hand_red_counts = [0] * 34
        for kind, is_red in st.session_state.hand_red.items():
            if is_red and st.session_state.hand_counts[kind] > 0:
                hand_red_counts[kind] = 1

        dora_tiles = []
        if st.session_state.dora_indicator != "(冇)":
            indicator_kind = TILE_LABELS.index(st.session_state.dora_indicator)
            dora_kind = TileKind(indicator_kind).successor()
            dora_tiles = [dora_kind.value]

        state = build_state(
            hand_counts=st.session_state.hand_counts,
            hand_red_counts=hand_red_counts,
            meld_counts=st.session_state.meld_counts,
            discard_counts=st.session_state.discard_counts,
            dora_tiles=dora_tiles,
            round_wind=st.session_state.round_wind,
            seat_wind=st.session_state.seat_wind,
            riichi=st.session_state.riichi,
        )
        st.session_state.result = predict(model, state, device)
        st.session_state.chosen_discard = None

        trigger_tile = -1
        if is_reaction_mode:
            trigger_tile = TILE_LABELS.index(st.session_state.trigger_tile_label)

        st.session_state.action_result = predict_action(
            action_model,
            device,
            seat=0,
            hand_counts=st.session_state.hand_counts,
            hand_red_counts=hand_red_counts,
            meld_counts=st.session_state.meld_counts,
            discard_counts=st.session_state.discard_counts,
            riichi=st.session_state.riichi,
            dora_tiles=dora_tiles,
            round_wind=st.session_state.round_wind,
            seat_wind=st.session_state.seat_wind,
            trigger_tile=trigger_tile,
        )
        st.session_state.action_is_reaction = is_reaction_mode
        st.session_state.last_trigger_tile = trigger_tile

    result = st.session_state.result
    if result is not None:
        st.header("Phase 1 model（淨係識掉牌）")
        st.subheader("Model 推薦（分數由高到低）")
        labels = [TILE_LABELS[k] for k, _ in result.ranked_hand_tiles]
        probs = [p for _, p in result.ranked_hand_tiles]
        fig = go.Figure(go.Bar(x=labels, y=probs, marker_color="indianred"))
        fig.update_layout(yaxis_title="predicted probability", height=350)
        st.plotly_chart(fig, width="stretch")

        st.write("**安全度（現物判斷，見返上面「全部人嘅牌河」有冇填）**：")
        for kind, prob in result.ranked_hand_tiles:
            label = safety_label(st.session_state.discard_counts, kind, st.session_state.riichi)
            if label:
                st.caption(f"{TILE_LABELS[kind]}（{prob:.1%}）：{label}")

        st.write("撳你實際想掉嗰隻牌：")
        cols = st.columns(len(result.ranked_hand_tiles))
        for col, (kind, prob) in zip(cols, result.ranked_hand_tiles):
            label = f"{TILE_GLYPHS[kind]}\n{prob:.1%}"
            if col.button(label, key=f"choose_{kind}"):
                st.session_state.chosen_discard = kind

        if st.session_state.chosen_discard is not None:
            chosen = st.session_state.chosen_discard
            top = result.ranked_hand_tiles[0][0]
            rank = next(i for i, (k, _) in enumerate(result.ranked_hand_tiles, start=1) if k == chosen)
            if chosen == top:
                st.success(f"你揀咗 {TILE_LABELS[chosen]}，同 model 建議一致。")
            else:
                st.info(
                    f"你揀咗 {TILE_LABELS[chosen]}（model 排名第 {rank} 位），"
                    f"model 首選係 {TILE_LABELS[top]}。"
                )

        st.subheader("果蠅腦視覺化（Phase 1 model）")
        render_brain_viz(coords, result.activations)

    action_result = st.session_state.action_result
    if action_result is not None:
        st.divider()
        st.header("Phase 2 model（識埋叫牌/立直/自摸/槓）")

        if st.session_state.get("action_is_reaction"):
            st.subheader("反應排名（PASS / PON / CHII / OPEN_KAN / RON）")
            names = [n for n, _ in action_result.reaction_ranked]
            probs = [p for _, p in action_result.reaction_ranked]
            fig = go.Figure(go.Bar(x=names, y=probs, marker_color="seagreen"))
            fig.update_layout(yaxis_title="predicted probability", height=300)
            st.plotly_chart(fig, width="stretch")

            trigger_tile = st.session_state.get("last_trigger_tile", -1)
            if pon_available(st.session_state.hand_counts, trigger_tile):
                st.write(f"**PON**：用你手牌入面兩隻 {TILE_LABELS[trigger_tile]}")
            combos = chii_combos(st.session_state.hand_counts, trigger_tile)
            if combos:
                combo_strs = "、".join(f"{TILE_LABELS[t1]}+{TILE_LABELS[t2]}" for t1, t2 in combos)
                st.write(f"**CHII**：可以用 {combo_strs} 嚟吃")
            if open_kan_available(st.session_state.hand_counts, trigger_tile):
                st.write(f"**OPEN_KAN**：用你手牌入面三隻 {TILE_LABELS[trigger_tile]}")
            if not (pon_available(st.session_state.hand_counts, trigger_tile) or combos):
                st.caption("（你手牌實際上冇嘢可以碰/吃呢隻牌——如果 model 都係咁估，可能純粹反映緊「大部分時候都係 PASS」嘅先驗，唔代表呢鋪叫得牌）")
        else:
            st.subheader("自己回合揀邊種 action")
            names = [n for n, _ in action_result.self_type_ranked]
            probs = [p for _, p in action_result.self_type_ranked]
            fig = go.Figure(go.Bar(x=names, y=probs, marker_color="seagreen"))
            fig.update_layout(yaxis_title="predicted probability", height=300)
            st.plotly_chart(fig, width="stretch")

            st.subheader("如果係 discard/riichi，邊隻牌（分數由高到低）")
            labels = [TILE_LABELS[k] for k, _ in action_result.discard_ranked_hand_tiles]
            probs = [p for _, p in action_result.discard_ranked_hand_tiles]
            fig2 = go.Figure(go.Bar(x=labels, y=probs, marker_color="indianred"))
            fig2.update_layout(yaxis_title="predicted probability", height=300)
            st.plotly_chart(fig2, width="stretch")

            st.write("**安全度（現物判斷）**：")
            for kind, prob in action_result.discard_ranked_hand_tiles:
                label = safety_label(st.session_state.discard_counts, kind, st.session_state.riichi)
                if label:
                    st.caption(f"{TILE_LABELS[kind]}（{prob:.1%}）：{label}")

        st.subheader("果蠅腦視覺化（Phase 2 model）")
        render_brain_viz(coords, action_result.activations)


if __name__ == "__main__":
    main()
