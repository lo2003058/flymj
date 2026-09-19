"""Phase 2 Part 1（驗證）：由歷史牌譜逆推 wall，用 jansou.game.flow.deal_steps
逐步重演一場真實牌局，睇吓最後個結果（邊個糊/流局、邊個係莊）啱唔啱歷史記錄。

跑法： uv run python src/validate_replay.py
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
        print(f"❌ 重演失敗: {type(e).__name__}: {e}")
        return False

    print(f"重演出嚟嘅 outcome: winners={outcome.winners}  is_draw={outcome.is_draw}")

    hist_outcome = round_log.outcome
    if isinstance(hist_outcome, Ryuukyoku):
        expected_winners = ()
        expected_is_draw = True
    else:
        expected_winners = tuple(a.winner for a in hist_outcome)
        expected_is_draw = False
    print(f"歷史記錄嘅 outcome:   winners={expected_winners}  is_draw={expected_is_draw}")

    ok = set(outcome.winners) == set(expected_winners) and outcome.is_draw == expected_is_draw
    print("✅ 一致" if ok else "❌ 唔一致")
    return ok


def main() -> None:
    paifu = parse_mjai(TEST_FILE)
    round_log = paifu.rounds[TEST_ROUND_INDEX]
    print(f"驗證緊 {TEST_FILE} 第 {TEST_ROUND_INDEX} 局")
    print(f"事件數: {len(round_log.events)}")
    validate_round(round_log, paifu.player_count, paifu.rules)


if __name__ == "__main__":
    main()
