import json
import time

from bubbleshooter import decider, strategy
from bubbleshooter.config import DeciderSettings, StrategyWeights
from bubbleshooter.sim.engine import SimGame

WEIGHTS = StrategyWeights()
SETTINGS = DeciderSettings(decider="model", model_timeout_s=0.2)


class FakeClient:
    def __init__(self, answer=None, delay_s=0.0, error=None):
        self.answer, self.delay_s, self.error = answer, delay_s, error
        self.prompts = []

    def complete(self, system, user, timeout_s):
        self.prompts.append(user)
        time.sleep(self.delay_s)
        if self.error:
            raise self.error
        return self.answer(user) if callable(self.answer) else self.answer


def board(seed=1):
    game = SimGame(seed)
    return game.shooter_color(), game.bubbles()


def rule_choice(shooter, bubbles):
    return strategy.choose_target(shooter, bubbles, weights=WEIGHTS)


def other_candidate(shooter, bubbles):
    candidates = strategy.list_candidates(shooter, bubbles, weights=WEIGHTS)
    best = strategy.best_candidate(candidates)
    return next(c for c in candidates if c.cell != best.cell)


def answer_for(candidate):
    row, col = decider.cell_address(candidate.cell)
    return json.dumps({"row": row, "col": col})


def test_choose_target_matches_best_candidate_on_many_boards():
    for seed in range(1, 9):
        game = SimGame(seed)
        for _ in range(25):
            if game.is_over:
                break
            shooter, bubbles = game.shooter_color(), game.bubbles()
            candidates = strategy.list_candidates(shooter, bubbles, weights=WEIGHTS)
            target = rule_choice(shooter, bubbles)
            assert target == strategy.best_candidate(candidates).aim()
            assert len({c.cell for c in candidates}) == len(candidates)
            game.shoot(target[:2])


def test_cell_address_round_trips_through_layout_position():
    for row in range(4):
        for col in (0, 5, 16):
            x, y = SimGame.layout_position((col, row))
            assert decider.cell_address((x, y, row)) == (row, col)


def test_rule_decider_returns_the_rule_choice():
    shooter, bubbles = board()
    decision = decider.RuleDecider(WEIGHTS).decide(shooter, bubbles)
    assert (decision.x, decision.y, decision.reason) == rule_choice(shooter, bubbles)
    assert decision.source == "rule" and decision.fallback is None


def test_model_answer_is_used_and_aim_comes_from_code():
    shooter, bubbles = board()
    pick = other_candidate(shooter, bubbles)
    model = decider.ModelDecider(WEIGHTS, SETTINGS, client=FakeClient(answer_for(pick)))
    decision = model.decide(shooter, bubbles)
    assert (decision.x, decision.y, decision.reason) == pick.aim()
    assert decision.source == "model" and decision.fallback is None and decision.agrees_with_rule is False
    assert model.model_shots == 1 and model.disagreements == 1


def test_prompt_lists_every_reachable_cell_without_scores():
    shooter, bubbles = board()
    client = FakeClient('{"row": 0, "col": 0}')
    decider.ModelDecider(WEIGHTS, SETTINGS, client=client).decide(shooter, bubbles)
    candidates = strategy.list_candidates(shooter, bubbles, weights=WEIGHTS)
    prompt = client.prompts[0]
    for candidate in candidates:
        row, col = decider.cell_address(candidate.cell)
        assert f"- row {row} col {col}: {candidate.why}" in prompt
    assert "score" not in prompt.lower()


def assert_falls_back(client, expected):
    shooter, bubbles = board()
    model = decider.ModelDecider(WEIGHTS, SETTINGS, client=client)
    decision = model.decide(shooter, bubbles)
    assert (decision.x, decision.y, decision.reason) == rule_choice(shooter, bubbles)
    assert decision.source == "rule" and decision.fallback == expected
    assert model.fallbacks == {expected: 1} and model.model_shots == 0


def test_malformed_json_falls_back():
    assert_falls_back(FakeClient("row 3, col 4"), "invalid json")


def test_wrong_field_types_fall_back():
    assert_falls_back(FakeClient('{"row": "3", "col": 4}'), "invalid fields")
    assert_falls_back(FakeClient('{"row": true, "col": 4}'), "invalid fields")
    assert_falls_back(FakeClient('{"row": 3}'), "invalid fields")


def test_unreachable_cell_falls_back():
    assert_falls_back(FakeClient('{"row": 0, "col": 0}'), "unreachable cell")


def test_timeout_falls_back_within_the_deadline():
    started = time.perf_counter()
    assert_falls_back(FakeClient('{"row": 0, "col": 0}', delay_s=1.0), "timeout")
    assert time.perf_counter() - started < 0.6


def test_server_down_falls_back():
    assert_falls_back(FakeClient(error=ConnectionRefusedError("refused")), "request failed: ConnectionRefusedError")


def test_decisions_are_logged(tmp_path):
    shooter, bubbles = board()
    log = decider.DecisionLog(tmp_path)
    model = decider.ModelDecider(WEIGHTS, SETTINGS, client=FakeClient("nope"), log=log)
    model.decide(shooter, bubbles)
    record = json.loads(log.path.read_text().splitlines()[0])
    assert record["fallback"] == "invalid json" and record["source"] == "rule" and "latency_ms" in record


def test_make_decider_defaults_to_rule():
    from bubbleshooter.config import Settings

    assert decider.make_decider(Settings()).name == "rule"
