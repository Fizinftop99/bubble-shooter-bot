"""The numba path of SimGame (engine_fast) must give bit-identical games to the pure Python path."""
import copy
import math
import os
import random
import subprocess
import sys

import pytest

pytest.importorskip("numba")

from bubbleshooter import strategy_fast  # noqa: E402
from bubbleshooter.config import StrategyWeights  # noqa: E402
from bubbleshooter.sim import engine, engine_fast  # noqa: E402
from bubbleshooter.sim.engine import SimGame  # noqa: E402
from bubbleshooter.sim.teacher import RolloutTeacher, TeacherConfig  # noqa: E402

GAME_SEEDS = range(1, 51)
MAX_GAME_SHOTS = 2000
WEIGHTS = StrategyWeights()


@pytest.fixture
def kernels(monkeypatch):
    """Switch engine.FAST per call: use(True) for numba, use(False) for pure Python."""

    def use(fast):
        monkeypatch.setattr(engine, "FAST", engine_fast if fast else None)

    yield use


def state(game):
    return (
        list(game.board.items()),  # dict order too: bubbles() and so the rule depend on it
        game.score, game.shots, game.lives, game.max_lives, game.pushes, game.last_cell,
        game.won, game.lost, game.ready, game.queue, game.rng.getstate(),
    )


def twin(game):
    other = copy.copy(game)
    other.board = dict(game.board)
    other.rng = random.Random()
    other.rng.setstate(game.rng.getstate())
    return other


def shoot_both(kernels, fast_game, pure_game, click):
    """Shoot the same click on both games; returns the outcome (exception type or popped flag)."""
    results = []
    for game, fast in ((fast_game, True), (pure_game, False)):
        kernels(fast)
        try:
            results.append(game.shoot(click))
        except RuntimeError as e:
            results.append(type(e))
    assert results[0] == results[1]
    assert state(fast_game) == state(pure_game)
    return results[1]


# --- the collision test ---------------------------------------------------------------------------


def boundary_pairs(rng, count):
    """(dx, dy) within a few ulps of hypot == COLLISION_DIST, in every quadrant."""
    r = engine.COLLISION_DIST
    for _ in range(count):
        dy = rng.uniform(0, r)
        dx = math.sqrt(r * r - dy * dy)
        for k in range(-4, 5):
            x = dx
            for _ in range(abs(k)):
                x = math.nextafter(x, math.inf if k > 0 else -math.inf)
            yield rng.choice((1, -1)) * x, rng.choice((1, -1)) * dy
            yield rng.choice((1, -1)) * dy, rng.choice((1, -1)) * x


def test_hypot_port_is_bit_identical_to_cpython():
    rng = random.Random(5)
    pairs = list(boundary_pairs(rng, 20000))
    pairs += [(rng.uniform(-40, 40), rng.uniform(-40, 40)) for _ in range(50000)]
    pairs += [(rng.uniform(-1, 1) * 10.0 ** rng.randint(-300, 300), rng.uniform(-1, 1) * 10.0 ** rng.randint(-300, 300)) for _ in range(20000)]
    pairs += [(0.0, 0.0), (0.0, 24.0), (24.0, 0.0), (-24.0, -0.0), (5e-324, 0.0), (1e308, 1e308)]
    inside = 0
    for dx, dy in pairs:
        expected = math.hypot(dx, dy)
        got = engine_fast._cpython_hypot(dx, dy)
        assert got == expected or (math.isinf(got) and math.isinf(expected)), (dx, dy, got, expected)
        hit = expected < engine.COLLISION_DIST
        assert engine_fast._collides(dx, dy) == hit, (dx, dy)
        inside += hit
    assert 0 < inside < len(pairs)


def test_squared_distance_alone_would_not_match():
    """Shows why the port is needed: near the boundary dx*dx + dy*dy < r*r disagrees with hypot < r."""
    r = engine.COLLISION_DIST
    disagree = sum(
        (dx * dx + dy * dy < r * r) != (math.hypot(dx, dy) < r) for dx, dy in boundary_pairs(random.Random(6), 5000)
    )
    assert disagree > 0


# --- whole games ----------------------------------------------------------------------------------


def test_rule_games_are_identical_after_every_shot(kernels):
    total = 0
    for seed in GAME_SEEDS:
        fast_game, pure_game = SimGame(seed), SimGame(seed)
        assert state(fast_game) == state(pure_game)
        while not pure_game.is_over and pure_game.shots < MAX_GAME_SHOTS:
            target = strategy_fast.choose_target(pure_game.shooter_color(), pure_game.bubbles(), weights=WEIGHTS)
            shoot_both(kernels, fast_game, pure_game, target[:2])
            total += 1
    assert total > 50 * 100


def mid_game_positions(kernels, seeds, every=15, limit=120):
    kernels(False)
    for seed in seeds:
        game = SimGame(seed)
        while not game.is_over and game.shots < limit:
            if game.shots % every == 0:
                yield twin(game)
            game.shoot(strategy_fast.choose_target(game.shooter_color(), game.bubbles(), weights=WEIGHTS)[:2])


def random_click(rng):
    """Anywhere on and around the screen: steep, flat, clamped below the horizon, far off to either side."""
    kind = rng.random()
    if kind < 0.6:
        deg = rng.uniform(5, 175)
        rad = math.radians(deg)
        dist = rng.uniform(20, 600)
        x, y = engine.SHOOTER_X + dist * math.cos(rad), engine.SHOOTER_Y - dist * math.sin(rad)
    elif kind < 0.9:
        x, y = rng.uniform(-200, 800), rng.uniform(-200, 700)
    else:
        x, y = rng.choice((-5000.0, 5000.0)), engine.SHOOTER_Y + rng.uniform(-50, 50)
    return x * engine.SCREEN_SCALE, y * engine.SCREEN_SCALE


def test_random_clicks_match(kernels):
    rng = random.Random(11)
    shots = 0
    for position in mid_game_positions(kernels, range(1, 21)):
        for _ in range(40):
            if position.is_over:
                break
            shoot_both(kernels, twin(position), twin(position), random_click(rng))
            shots += 1
    assert shots > 3000


def landing_key(kernels, game, click):
    g = twin(game)
    kernels(False)
    try:
        g.shoot(click)
    except RuntimeError:
        return None
    return g.last_cell, len(g.board)


def test_near_tangent_shots_match(kernels):
    """Bisect the click between two that land differently down to adjacent doubles: these shots graze a
    bubble (or a snap boundary) as closely as floating point allows. Check them and their neighbours."""
    rng = random.Random(13)
    edges = 0
    for position in mid_game_positions(kernels, range(1, 11), every=20):
        if position.is_over:
            continue
        for _ in range(6):
            y = rng.uniform(100, 480) * engine.SCREEN_SCALE
            lo, hi = rng.uniform(0, 300), rng.uniform(300, 600)
            k_lo, k_hi = landing_key(kernels, position, (lo, y)), landing_key(kernels, position, (hi, y))
            if k_lo == k_hi:
                continue
            while math.nextafter(lo, hi) != hi:
                mid = (lo + hi) / 2
                if mid in (lo, hi):
                    break
                if landing_key(kernels, position, (mid, y)) == k_lo:
                    lo = mid
                else:
                    hi = mid
            edges += 1
            x = lo
            for _ in range(8):
                x = math.nextafter(x, -math.inf)
            for _ in range(17):
                shoot_both(kernels, twin(position), twin(position), (x, y))
                x = math.nextafter(x, math.inf)
    assert edges > 30


def test_wall_bounces_match(kernels):
    """Flat shots that bounce off both walls several times before landing."""
    shots = 0
    for position in mid_game_positions(kernels, range(1, 6), every=30):
        for deg in [15 + 0.37 * i for i in range(30)] + [165 - 0.37 * i for i in range(30)]:
            if position.is_over:
                break
            rad = math.radians(deg)
            click = (
                (engine.SHOOTER_X + 300 * math.cos(rad)) * engine.SCREEN_SCALE,
                (engine.SHOOTER_Y - 300 * math.sin(rad)) * engine.SCREEN_SCALE,
            )
            shoot_both(kernels, twin(position), twin(position), click)
            shots += 1
    assert shots > 500


def test_methods_match_on_edited_boards(kernels):
    """_fly, _resolve_landing and _drop_hanging on boards with holes, islands and rows far down."""
    rng = random.Random(17)
    checked = 0
    for _ in range(400):
        board = {}
        for by in range(rng.randint(1, 22)):
            for bx in range(engine.BOARD_WIDTH):
                if rng.random() < 0.7:
                    board[(bx, by)] = rng.randrange(engine.COLOR_COUNT)
        vx, vy = SimGame._launch_velocity(rng.uniform(0, 640), rng.uniform(0, 540))
        pick = rng.random()
        results = []
        for fast in (True, False):
            kernels(fast)
            game = SimGame(0)
            game.board = dict(board)
            try:
                cell = game._fly(vx, vy)
            except RuntimeError:
                cell = None
            dropped = game._drop_hanging()
            cell2 = list(game.board)[int(pick * len(game.board))] if game.board else None
            popped = game._resolve_landing(cell2, game.board[cell2]) if cell2 else None
            results.append((cell, dropped, cell2, popped, list(game.board.items()), game.score))
        assert results[0] == results[1]
        checked += results[1][3] is not None
    assert checked > 300


def test_teacher_values_match(kernels):
    cfg = TeacherConfig(top_k=3, rollouts=2, depth=2, fast_rule=True)
    decisions = 0
    for seed in range(1, 9):
        runs = []
        for fast in (True, False):
            kernels(fast)
            game = SimGame(seed)
            teacher = RolloutTeacher(WEIGHTS, cfg, seed).bind(game)
            log = []
            while not game.is_over and game.shots < 40:
                d = teacher.decide(game.shooter_color(), game.bubbles())
                log.append(((d.x, d.y, d.reason), [(repr(c), v) for c, v in teacher.last_values]))
                game.shoot((d.x, d.y))
            runs.append((log, state(game)))
        assert runs[0] == runs[1]
        decisions += len(runs[0][0])
    assert decisions > 200


def test_pure_env_var_disables_kernels():
    env = dict(os.environ, BUBBLE_SIM_PURE="1")
    out = subprocess.run(
        [sys.executable, "-c", "from bubbleshooter.sim import engine; print(engine.FAST is None)"],
        env=env, capture_output=True, text=True, check=True, cwd=os.path.dirname(os.path.dirname(__file__)),
    )
    assert out.stdout.strip() == "True"
