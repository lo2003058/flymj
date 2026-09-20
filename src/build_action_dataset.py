"""Phase 2 Part 2: using the already-validated wall reconstruction +
oracle (see validate_replay.py), replays historical matches step by step,
recording at every decision point: the game-state features, the legal
action types (as computed by jansou), and which one was actually chosen
historically.

Unlike the Step 3/6 discard-only dataset, this one also includes
pon/chii/kan/riichi/ron/pass decisions — it's the official expanded
Phase 2 "calls/riichi" dataset.

Writes data/processed/action_dataset.parquet:
  - match_id, kyoku_index, decision_index, seat, decision_kind
  - hand_counts, hand_red_counts, meld_counts, discard_counts, riichi,
    dora_tiles, round_wind, seat_wind   (game-state features, aligned with
    features.py's FullDiscardState so the encode() function can be reused)
  - trigger_seat, trigger_tile: only meaningful for DISCARD_REACTION
    (-1 = none); trigger_seat is relative to the seat making this
    decision (0=self/1=right/2=across/3=left)
  - action_type: DISCARD/RIICHI/TSUMO/KAN/KYUUSHU/PASS/PON/CHII/
    OPEN_KAN/RON/TENPAI_YES/TENPAI_NO
  - tile: 0-33 (only meaningful for DISCARD/RIICHI, -1 otherwise)
  - split

Run: uv run python src/build_action_dataset.py
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

MAX_FILES = 3000  # matches the scale of the Step 6 discard-only dataset
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
    raise ValueError(f"Unknown action type: {action!r}")


def snapshot_state(state: GameState, seat: int, player_count: int, *, is_reaction: bool) -> dict:
    """Read a decision point's game-state features straight from jansou's
    authoritative GameState, instead of maintaining our own separate
    tracker (this GameState has already been validated as accurate)."""
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

    # DISCARD_REACTION decision: which seat's discard, and which tile, is
    # currently being reacted to (SELF decisions have none of this). Must
    # use the explicit is_reaction flag derived from point.kind, not just
    # check whether state.last_discard is not None — when nobody calls,
    # flow.py doesn't clear that field, leaving a stale value from the
    # previous turn that would incorrectly flag SELF decisions as reactions too.
    trigger_seat = -1
    trigger_tile = -1
    if is_reaction:
        assert state.last_discard is not None, "DISCARD_REACTION but state.last_discard is None"
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
    """Returns every row of decisions for this round, or None if the
    replay diverges partway through (skip the whole round rather than
    keeping partial/inaccurate rows)."""
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
    print(f"Processing {len(files)} files (MAX_FILES={MAX_FILES})")

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
            print(f"Processed {file_index + 1}/{len(files)} files, {len(all_rows)} decisions so far, {skipped_rounds} rounds skipped")

    print(f"\n{total_rounds} rounds total, {skipped_rounds} skipped ({skipped_rounds / total_rounds:.2%})")
    print(f"{len(all_rows)} decisions total")

    df = pl.DataFrame(all_rows)
    print("\n=== action_type distribution (before subsampling) ===")
    print(df.group_by("action_type").agg(pl.len().alias("count")).sort("count", descending=True))

    print("\n=== decision_kind distribution ===")
    print(df.group_by("decision_kind").agg(pl.len().alias("count")).sort("count", descending=True))

    rng_seed = PASS_SUBSAMPLE_SEED
    n_pass = df.filter(pl.col("action_type") == "PASS").height
    n_keep_pass = int(n_pass * PASS_KEEP_RATE)
    pass_df = df.filter(pl.col("action_type") == "PASS").sample(n=n_keep_pass, seed=rng_seed)
    other_df = df.filter(pl.col("action_type") != "PASS")
    df = pl.concat([other_df, pass_df]).sort(["match_id", "kyoku_index", "decision_index"])

    print(f"\nPASS subsampled from {n_pass} to {n_keep_pass} (PASS_KEEP_RATE={PASS_KEEP_RATE})")
    print(f"Total decisions after subsampling: {df.height}")

    print("\n=== action_type distribution (after subsampling) ===")
    print(df.group_by("action_type").agg(pl.len().alias("count")).sort("count", descending=True))

    print("\n=== decision count by split ===")
    print(df.group_by("split").agg(pl.len().alias("count")))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT_PATH)
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
