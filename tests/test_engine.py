import math

from bubbleshooter.sim import engine
from bubbleshooter.sim.engine import SimGame


def empty_game(seed=0):
    game = SimGame(seed)
    game.board = {}
    game.ready, game.queue = 0, 1
    game.lives = game.max_lives = engine.START_LIVES
    return game


def click_for_angle(deg):
    rad = math.radians(deg)
    return (
        (engine.SHOOTER_X + 300 * math.cos(rad)) * engine.SCREEN_SCALE,
        (engine.SHOOTER_Y - 300 * math.sin(rad)) * engine.SCREEN_SCALE,
    )


def test_initial_board_is_nine_full_rows():
    game = SimGame(3)
    assert len(game.board) == engine.INITIAL_ROWS * engine.BOARD_WIDTH
    assert {by for _, by in game.board} == set(range(engine.INITIAL_ROWS))


def test_same_seed_gives_same_game():
    a, b = SimGame(7), SimGame(7)
    assert a.board == b.board and a.ready == b.ready and a.queue == b.queue


def test_straight_shot_snaps_under_ceiling():
    game = empty_game()
    cell = game._fly(0, -engine.LAUNCH_SPEED)
    assert cell[1] == 0


def test_angle_is_clamped_to_fifteen_degrees_above_horizontal():
    vx, vy = SimGame._launch_velocity(2000, engine.SHOOTER_Y - 1)
    assert vx > 0
    assert abs(math.degrees(math.atan2(-vy, vx)) - 15) < 1e-6


def test_wall_bounce_reverses_direction():
    game = empty_game()
    cell = game._fly(*SimGame._launch_velocity(*click_for_angle(20)))
    assert cell[1] <= 3
    assert cell[0] < engine.BOARD_WIDTH


def test_pop_of_three_scores_thirty():
    game = empty_game()
    game.board = {(5, 0): 0, (6, 0): 0, (7, 0): 0, (6, 1): 2}
    assert game._resolve_landing((7, 0), 0)
    assert game.score == 30 + engine.HANG_SCORE
    assert game.board == {}


def test_score_per_removed_bubble_steps_every_three():
    game = empty_game()
    game.board = {(x, 0): 0 for x in range(6)}
    assert game._resolve_landing((0, 0), 0)
    assert game.score == 10 * 3 + 20 * 3


def test_pair_does_not_pop():
    game = empty_game()
    game.board = {(5, 0): 0, (6, 0): 0}
    assert not game._resolve_landing((5, 0), 0)
    assert game.score == 0 and len(game.board) == 2


def test_hanging_bubbles_score_hundred_each():
    game = empty_game()
    game.board = {(0, 0): 0, (1, 0): 0, (2, 0): 0, (1, 1): 3, (1, 2): 4}
    assert game._resolve_landing((0, 0), 0)
    assert game.score == 30 + 2 * engine.HANG_SCORE
    assert game.board == {}


def test_miss_costs_lives_then_pushes_one_row():
    game = empty_game()
    game.board = {(5, 0): 0, (6, 0): 1, (7, 0): 2, (8, 0): 3, (9, 0): 4, (10, 0): 5}
    for _ in range(engine.START_LIVES):
        game._lose_life()
    assert game.lives == 0 and game.pushes == 0
    game._lose_life()
    assert game.pushes == 1
    assert max(by for _, by in game.board) == 1
    assert game.lives == game.max_lives == engine.START_LIVES - 1


def test_push_adds_a_row_per_missing_colour():
    game = empty_game()
    game.ready, game.queue = 0, 0
    game.board = {(5, 0): 0, (6, 0): 1}
    game._push_rows()
    assert max(by for _, by in game.board) == 5
    assert game.max_lives == 1


def test_loss_when_a_bubble_reaches_row_fifteen():
    game = empty_game()
    game.board = {(0, engine.LOSS_ROW): 0}
    game._finish_turn()
    assert game.lost and not game.won


def test_win_doubles_the_score():
    game = empty_game()
    game.score = 500
    game._finish_turn()
    assert game.won and game.final_score == 1000


def test_new_shooter_colours_come_from_the_board():
    game = SimGame(5)
    game.board = {(0, 0): 2, (1, 0): 2}
    game.ready = 2
    for _ in range(20):
        game._promote()
        assert game.queue == 2


GOLDEN_RANDOM_GAMES = {
    0: (25, 290, 4, 221),
    1: (17, 30, 3, 217),
    2: (19, 30, 4, 235),
}


def test_compiled_engine_reproduces_reference_games():
    import random

    for seed, expected in GOLDEN_RANDOM_GAMES.items():
        game = SimGame(seed)
        rng = random.Random(seed * 7919)
        while not game.is_over and game.shots < 400:
            game.shoot(click_for_angle(rng.uniform(5, 175)))
        assert (game.shots, game.score, game.pushes, len(game.board)) == expected
        assert game.lost and not game.won
