"""Phase 2 Part 1 (validation): reconstructs the wall from a historical
log, replays a real match step by step with jansou.game.flow.deal_steps,
and checks whether the final result (who won/drew, who was dealer) matches
the historical record.

Run: uv run python src/validate_replay.py
"""

from jansou.core.rules import RIICHI_DEPOSIT
from jansou.game.flow import IllegalActionError, Position, deal_steps, new_deal
from jansou.io.mjai import parse_mjai
from jansou.io.paifu import Agari, Ryuukyoku

from replay_oracle import HistoricalOracle, OracleMismatch
from wall_reconstruction import reconstruct_wall

TEST_FILE = "data/raw/paifu/2009/2009022011gm-00a9-0000-d7935c6d.mjson"
TEST_ROUND_INDEX = 0


def validate_round(round_log, player_count, rules) -> bool:
    wall = reconstruct_wall(round_log, player_count)
    position = Position(
        dealer=round_log.dealer,
        round_wind=round_log.round_wind,
        round_number=1,
        honba=round_log.honba,
    )
    state = new_deal(rules, wall, position, list(round_log.scores), round_log.riichi_sticks * RIICHI_DEPOSIT)

    oracle = HistoricalOracle(round_log)
    events_emitted = []

    steps = deal_steps(state, events_emitted.append)
    try:
        point = next(steps)
        while True:
            action = oracle.decide(point.seat, point.kind, list(point.actions))
            point = steps.send(action)
    except StopIteration as stop:
        outcome = stop.value
    except (OracleMismatch, IllegalActionError) as e:
        print(f"❌ Replay failed: {type(e).__name__}: {e}")
        return False

    print(f"Replayed outcome: winners={outcome.winners}  is_draw={outcome.is_draw}")

    hist_outcome = round_log.outcome
    if isinstance(hist_outcome, Ryuukyoku):
        expected_winners = ()
        expected_is_draw = True
    else:
        expected_winners = tuple(a.winner for a in hist_outcome)
        expected_is_draw = False
    print(f"Historical outcome:  winners={expected_winners}  is_draw={expected_is_draw}")

    ok = set(outcome.winners) == set(expected_winners) and outcome.is_draw == expected_is_draw
    print("✅ Match" if ok else "❌ Mismatch")
    return ok


def main() -> None:
    paifu = parse_mjai(TEST_FILE)
    round_log = paifu.rounds[TEST_ROUND_INDEX]
    print(f"Validating {TEST_FILE} round {TEST_ROUND_INDEX}")
    print(f"Event count: {len(round_log.events)}")
    validate_round(round_log, paifu.player_count, paifu.rules)


if __name__ == "__main__":
    main()
