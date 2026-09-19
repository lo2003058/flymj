"""Phase 2 Part 2：用已經驗證過嘅 wall reconstruction + oracle（見
validate_replay.py），逐步重演歷史牌局，喺每一個 decision point 攞低：
場面 feature、（jansou 算出嚟嘅）合法 action 種類、歷史實際揀咗邊種。

同 Step 3/6 嘅 discard-only dataset 唔同，呢度連 pon/chii/kan/riichi/ron/pass
呢啲決策都包埋，係 Phase 2「叫牌/立直」擴大嘅正式 dataset。

輸出 data/processed/action_dataset.parquet：
  - match_id, kyoku_index, decision_index, seat, decision_kind
  - hand_counts, hand_red_counts, meld_counts, discard_counts, riichi,
    dora_tiles, round_wind, seat_wind   （場面 feature，同 features.py 嘅
    FullDiscardState 對齊，方便重用個 encode() function）
  - trigger_seat, trigger_tile：DISCARD_REACTION 先有意義（-1 = 冇），
    trigger_seat 係相對於揀緊呢個決策嘅 seat（0=自己/1=下家/2=對家/3=上家）
  - action_type: DISCARD/RIICHI/TSUMO/KAN/KYUUSHU/PASS/PON/CHII/
    OPEN_KAN/RON/TENPAI_YES/TENPAI_NO
  - tile: 0-33（淨係 DISCARD/RIICHI 有意義，其他 -1）
  - split

跑法： uv run python src/build_action_dataset.py
"""

from pathlib import Path

import polars as pl
from jansou.core.rules import RIICHI_DEPOSIT
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
from jansou.game.flow import DecisionKind, IllegalActionError, Position, deal_steps, new_deal
from jansou.game.state import GameState
from jansou.io.mjai import parse_mjai

from replay_oracle import HistoricalOracle, OracleMismatch
from splits import assign_splits
from wall_reconstruction import reconstruct_wall

YEAR = 2009
PAIFU_DIR = Path(f"data/raw/paifu/{YEAR}")
OUT_PATH = Path("data/processed/action_dataset.parquet")

MAX_FILES = 3000  # 同 Step 6 discard-only dataset 一致嘅規模
PASS_KEEP_RATE = 0.2
PASS_SUBSAMPLE_SEED = 0


def categorize_action(action) -> tuple[str, int]:
    if isinstance(action, Riichi):
        return "RIICHI", action.tile.kind.value
    if isinstance(action, ActionDiscard):
        return "DISCARD", action.tile.kind.value
    if isinstance(action, Tsumo):
        return "TSUMO", -1
    if isinstance(action, (ClosedKan, AddedKan)):
        return "KAN", -1
    if isinstance(action, NineTerminals):
        return "KYUUSHU", -1
    if isinstance(action, Nuki):
        return "NUKI", -1
    if isinstance(action, Pass):
        return "PASS", -1
    if isinstance(action, Pon):
        return "PON", -1
    if isinstance(action, Chii):
        return "CHII", -1
    if isinstance(action, OpenKan):
        return "OPEN_KAN", -1
    if isinstance(action, Ron):
        return "RON", -1
    if isinstance(action, DeclareTenpai):
        return ("TENPAI_YES" if action.declare else "TENPAI_NO"), -1
    raise ValueError(f"未知 action type: {action!r}")


def snapshot_state(state: GameState, seat: int, player_count: int, *, is_reaction: bool) -> dict:
    """由 jansou 自己權威嘅 GameState 攞返一個決策點嘅場面 feature，
    唔使自己另外維護一份 tracker（依家個 GameState 已經驗證過準確）。"""
    hand_counts = [0] * 34
    hand_red_counts = [0] * 34
    for tile in state.players[seat].as_hand(include_drawn=True).concealed:
        hand_counts[tile.kind.value] += 1
        if tile.red:
            hand_red_counts[tile.kind.value] += 1

    meld_counts = [[0] * 34 for _ in range(player_count)]
    discard_counts = [[0] * 34 for _ in range(player_count)]
    riichi = [False] * player_count
    for s in range(player_count):
        for meld in state.players[s].melds:
            for tile in meld.tiles:
                meld_counts[s][tile.kind.value] += 1
        for discard in state.players[s].discards:
            discard_counts[s][discard.tile.kind.value] += 1
        riichi[s] = state.players[s].is_riichi

    dora_tiles = [indicator.kind.successor().value for indicator in state.wall.dora_indicators]
    seat_wind = (seat - state.dealer) % player_count

    # DISCARD_REACTION 決策：而家反應緊邊個座位掉出嚟嘅邊隻牌（SELF 決策冇呢樣嘢）。
    # 一定要用 is_reaction 呢個嚟自 point.kind 嘅明確 flag，唔可以淨係睇
    # state.last_discard is not None——冇人叫嗰陣 flow.py 唔會清返呢個欄位，
    # 留返上一鋪嘅殘值，會令 SELF 決策都誤判做「反應緊」。
    trigger_seat = -1
    trigger_tile = -1
    if is_reaction:
        assert state.last_discard is not None, "DISCARD_REACTION 但 state.last_discard 係 None"
        discarder, tile = state.last_discard
        trigger_seat = (discarder - seat) % player_count
        trigger_tile = tile.kind.value

    return {
        "hand_counts": hand_counts,
        "hand_red_counts": hand_red_counts,
        "meld_counts": meld_counts,
        "discard_counts": discard_counts,
        "riichi": riichi,
        "trigger_seat": trigger_seat,
        "trigger_tile": trigger_tile,
        "dora_tiles": dora_tiles,
        "round_wind": state.round_wind.value,
        "seat_wind": seat_wind,
    }


def process_round(round_log, player_count, rules, match_id: str, kyoku_index: int, split: str) -> list[dict] | None:
    """回傳呢局全部決策嘅 row，如果 replay 中途對唔上就回傳 None（跳過成局，
    唔留低部分/唔準確嘅 row）。"""
    wall = reconstruct_wall(round_log, player_count)
    position = Position(dealer=round_log.dealer, round_wind=round_log.round_wind, round_number=1, honba=round_log.honba)
    state = new_deal(rules, wall, position, list(round_log.scores), round_log.riichi_sticks * RIICHI_DEPOSIT)
    oracle = HistoricalOracle(round_log)

    rows: list[dict] = []
    decision_index = 0
    events_emitted: list = []

    steps = deal_steps(state, events_emitted.append)
    try:
        point = next(steps)
        while True:
            snap = snapshot_state(
                state, point.seat, player_count, is_reaction=(point.kind == DecisionKind.DISCARD_REACTION)
            )
            action = oracle.decide(point.seat, point.kind, list(point.actions))
            action_type, tile = categorize_action(action)

            rows.append(
                {
                    "match_id": match_id,
                    "kyoku_index": kyoku_index,
                    "decision_index": decision_index,
                    "seat": point.seat,
                    "decision_kind": point.kind.name,
                    "split": split,
                    "action_type": action_type,
                    "tile": tile,
                    **snap,
                }
            )
            decision_index += 1
            point = steps.send(action)
    except StopIteration:
        return rows
    except (OracleMismatch, IllegalActionError):
        return None


def main() -> None:
    all_files = sorted(PAIFU_DIR.glob("*.mjson"))
    files = all_files[:MAX_FILES]
    print(f"處理 {len(files)} 個檔（MAX_FILES={MAX_FILES}）")

    file_splits = assign_splits(len(files))

    all_rows: list[dict] = []
    skipped_rounds = 0
    total_rounds = 0

    for file_index, path in enumerate(files):
        match_id = path.stem
        split = file_splits[file_index]
        paifu = parse_mjai(path)
        for kyoku_index, round_log in enumerate(paifu.rounds):
            total_rounds += 1
            rows = process_round(round_log, paifu.player_count, paifu.rules, match_id, kyoku_index, split)
            if rows is None:
                skipped_rounds += 1
                continue
            all_rows.extend(rows)

        if (file_index + 1) % 100 == 0:
            print(f"已處理 {file_index + 1}/{len(files)} 個檔，累積 {len(all_rows)} 個決策，跳過 {skipped_rounds} 局")

    print(f"\n總共 {total_rounds} 局，跳過 {skipped_rounds} 局（{skipped_rounds / total_rounds:.2%}）")
    print(f"總共 {len(all_rows)} 個決策")

    df = pl.DataFrame(all_rows)
    print("\n=== action_type 分佈（subsample 之前）===")
    print(df.group_by("action_type").agg(pl.len().alias("count")).sort("count", descending=True))

    print("\n=== decision_kind 分佈 ===")
    print(df.group_by("decision_kind").agg(pl.len().alias("count")).sort("count", descending=True))

    rng_seed = PASS_SUBSAMPLE_SEED
    n_pass = df.filter(pl.col("action_type") == "PASS").height
    n_keep_pass = int(n_pass * PASS_KEEP_RATE)
    pass_df = df.filter(pl.col("action_type") == "PASS").sample(n=n_keep_pass, seed=rng_seed)
    other_df = df.filter(pl.col("action_type") != "PASS")
    df = pl.concat([other_df, pass_df]).sort(["match_id", "kyoku_index", "decision_index"])

    print(f"\nPASS 由 {n_pass} 個 subsample 到 {n_keep_pass} 個（PASS_KEEP_RATE={PASS_KEEP_RATE}）")
    print(f"Subsample 之後總共 {df.height} 個決策")

    print("\n=== action_type 分佈（subsample 之後）===")
    print(df.group_by("action_type").agg(pl.len().alias("count")).sort("count", descending=True))

    print("\n=== 決策數按 split ===")
    print(df.group_by("split").agg(pl.len().alias("count")))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT_PATH)
    print(f"\n已存 {OUT_PATH}")


if __name__ == "__main__":
    main()
