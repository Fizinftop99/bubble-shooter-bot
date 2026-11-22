"""strategy.list_candidates through strategy_fast (numba) must return exactly the pure Python candidates."""
import random

import pytest

pytest.importorskip("numba")

from bubbleshooter import strategy, strategy_fast  # noqa: E402
from bubbleshooter.config import StrategyWeights  # noqa: E402
from bubbleshooter.sim.engine import SimGame  # noqa: E402
from bubbleshooter.sim.teacher import RolloutTeacher, TeacherConfig  # noqa: E402

GAME_SEEDS = range(1, 51)
SHOTS_PER_GAME = 60
# the pure reference costs ~70 ms a call: default weights on every 2nd position, all weight sets on every 10th
CHECK_EVERY = 2
OTHER_WEIGHTS_EVERY = 10
WEIGHT_SETS = [
    StrategyWeights(),
    StrategyWeights(wipe_free_bubbles=5, pressure_row=3, front_weight=2.5),
    StrategyWeights(drop_weight=3.3, depth_weight=0.7, miss_base=-1.1, friend_weight=1.3, growth_penalty=0.2,
                    pressure_row=0, color_wipe_penalty=7.7, wipe_free_bubbles=0),
]


def pure(shooter, bubbles, weights):
    return strategy._list_candidates_pure(shooter, bubbles, weights=weights)


def exact(candidates):
    """Candidates down to the types and the bits of every float (repr round-trips doubles exactly)."""
    return [(repr(c), [type(v) for v in c.cell], [type(d) for d in c.degs], type(c.score)) for c in candidates]


def assert_same(shooter, bubbles, weights):
    expected = pure(shooter, bubbles, weights)
    got = strategy_fast.list_candidates(shooter, bubbles, weights=weights)
    assert got == expected
    assert exact(got) == exact(expected)
    return len(expected)


def test_list_candidates_uses_fast_path(monkeypatch):
    assert strategy._fast_module() is strategy_fast or strategy._FAST == [None]
    bubbles, shooter = SimGame(3).bubbles(), SimGame(3).shooter_color()
    assert strategy.list_candidates(shooter, bubbles) == pure(shooter, bubbles, StrategyWeights())


def test_games_match_along_the_game():
    positions = candidates = 0
    for seed in GAME_SEEDS:
        game = SimGame(seed)
        rng = random.Random(seed)
        while not game.is_over and game.shots < SHOTS_PER_GAME:
            bubbles, shooter = game.bubbles(), game.shooter_color()
            if game.shots % CHECK_EVERY == 0:
                for weights in WEIGHT_SETS if game.shots % OTHER_WEIGHTS_EVERY == 0 else WEIGHT_SETS[:1]:
                    candidates += assert_same(shooter, bubbles, weights)
                positions += 1
            # mostly the rule's shot, sometimes a random one so the boards get ragged
            if rng.random() < 0.8:
                target = strategy_fast.choose_target(shooter, bubbles)
            else:
                target = (rng.uniform(0, 560), rng.uniform(40, 480))
            game.shoot(target[:2])
    assert positions > 50 * 25
    assert candidates > positions * 10


def test_edited_boards_and_other_shooters():
    """Holes, islands, single bubbles, odd top parity, colours not on the board, near-matching shades."""
    rng = random.Random(23)
    checked = 0
    for i in range(300):
        game = SimGame(1000 + i)
        for _ in range(rng.randint(0, 25)):
            if game.is_over:
                break
            game.shoot((rng.uniform(0, 560), rng.uniform(40, 480)))
        bubbles = [b for b in game.bubbles() if rng.random() < rng.choice((0.3, 0.7, 1.0))]
        if not bubbles:
            continue
        if rng.random() < 0.3:
            for b in bubbles:
                b["row"] += 1
                b["shifted"] = not b["shifted"]
        bubbles = [dict(b, color=tuple(max(0, min(255, c + rng.randint(-25, 25))) for c in b["color"]))
                   if rng.random() < 0.2 else b for b in bubbles]
        shooter = rng.choice([game.shooter_color(), (255, 255, 255), bubbles[0]["color"]])
        origin = rng.choice([(278, 516), (300, 500)])
        expected = strategy._list_candidates_pure(shooter, bubbles, origin, WEIGHT_SETS[i % 3])
        got = strategy_fast.list_candidates(shooter, bubbles, origin, WEIGHT_SETS[i % 3])
        assert got == expected and exact(got) == exact(expected)
        checked += 1
    assert checked > 250


def test_empty_and_unknown_shooter():
    bubbles = SimGame(1).bubbles()
    assert strategy_fast.list_candidates((0, 0, 0), []) == []
    assert strategy_fast.list_candidates(None, bubbles) == []
    assert strategy.list_candidates(None, bubbles) == []


def test_non_float_weights_fall_back_to_python():
    bubbles, shooter = SimGame(2).bubbles(), SimGame(2).shooter_color()
    weights = StrategyWeights(drop_weight=10, pressure_row=8)
    assert strategy_fast.list_candidates(shooter, bubbles, weights=weights) is None
    assert exact(strategy.list_candidates(shooter, bubbles, weights=weights)) == exact(pure(shooter, bubbles, weights))


def test_teacher_values_match(monkeypatch):
    cfg = TeacherConfig(top_k=3, rollouts=2, depth=2, fast_rule=True)
    weights = StrategyWeights()
    decisions = 0
    for seed in range(1, 9):
        runs = []
        for fast in (True, False):
            monkeypatch.setattr(strategy, "_FAST", [strategy_fast if fast else None])
            game = SimGame(seed)
            teacher = RolloutTeacher(weights, cfg, seed).bind(game)
            log = []
            while not game.is_over and game.shots < 40:
                d = teacher.decide(game.shooter_color(), game.bubbles())
                log.append(((d.x, d.y, d.reason), [(repr(c), v) for c, v in teacher.last_values]))
                game.shoot((d.x, d.y))
            runs.append((log, list(game.board.items()), game.score))
        assert runs[0] == runs[1]
        decisions += len(runs[0][0])
    assert decisions > 200


def test_pure_env_var_disables_fast_candidates():
    import os
    import subprocess
    import sys

    env = dict(os.environ, BUBBLE_SIM_PURE="1")
    out = subprocess.run(
        [sys.executable, "-c", "from bubbleshooter import strategy; print(strategy._fast_module() is None)"],
        env=env, capture_output=True, text=True, check=True, cwd=os.path.dirname(os.path.dirname(__file__)),
    )
    assert out.stdout.strip() == "True"
