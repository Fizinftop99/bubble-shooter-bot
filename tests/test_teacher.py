import copy

from bubbleshooter.config import StrategyWeights
from bubbleshooter.sim import teacher
from bubbleshooter.sim.engine import SimGame

WEIGHTS = StrategyWeights()
CFG = teacher.TeacherConfig(top_k=3, rollouts=2, depth=2)


def snapshot(game):
    return (
        dict(game.board), game.score, game.shots, game.pushes, game.lives, game.max_lives,
        game.ready, game.queue, game.won, game.lost, game.last_cell, game.rng.getstate(),
    )


def decide(seed, cfg=CFG):
    game = SimGame(seed)
    bot = teacher.RolloutTeacher(WEIGHTS, cfg, seed).bind(game)
    return game, bot, bot.decide(game.shooter_color(), game.bubbles())


def test_rollouts_do_not_mutate_the_real_game():
    game = SimGame(3)
    before = copy.deepcopy(snapshot(game))
    bot = teacher.RolloutTeacher(WEIGHTS, CFG, 3).bind(game)
    bot.decide(game.shooter_color(), game.bubbles())
    assert snapshot(game) == before


def test_same_seed_and_settings_give_the_same_choice():
    _, _, first = decide(5)
    _, _, second = decide(5)
    assert (first.x, first.y, first.reason) == (second.x, second.y, second.reason)


def test_choice_is_one_of_the_top_candidates():
    for seed in (1, 2):
        game, bot, decision = decide(seed)
        aims = {c.aim() for c in bot.top_candidates(game.shooter_color(), game.bubbles())}
        assert (decision.x, decision.y, decision.reason) in aims
        assert len(aims) <= CFG.top_k


def test_rollout_futures_differ_between_seeds():
    game = SimGame(4)
    twins = [teacher.clone_game(game, seed) for seed in (1, 2)]
    draws = [[twin.rng.random() for _ in range(4)] for twin in twins]
    assert draws[0] != draws[1]
    assert twins[0].board == game.board and twins[0].board is not game.board


def test_losing_is_valued_below_surviving():
    alive, dead = SimGame(1), SimGame(1)
    dead.lost = True
    assert teacher.outcome_value(dead, 0, 0, CFG) < teacher.outcome_value(alive, 0, 0, CFG)


def test_decide_without_bind_is_an_error():
    game = SimGame(1)
    try:
        teacher.RolloutTeacher(WEIGHTS, CFG).decide(game.shooter_color(), game.bubbles())
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError")


def test_play_teacher_reports_a_full_result():
    result = teacher.play_teacher(WEIGHTS, 1, max_shots=3, cfg=CFG)
    assert result.shots == 3 and result.outcome == "limit"
    assert result.model_calls == 3 and result.latency_ms_mean > 0
