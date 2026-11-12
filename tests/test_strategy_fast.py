import pytest

from bubbleshooter import strategy, strategy_fast
from bubbleshooter.config import StrategyWeights
from bubbleshooter.sim.engine import SimGame

SEEDS = range(6)
SHOTS_PER_SEED = 120
SPARSE_WEIGHTS = StrategyWeights(wipe_free_bubbles=5, pressure_row=3, front_weight=2.5)


@pytest.mark.parametrize("weights", [StrategyWeights(), SPARSE_WEIGHTS])
def test_fast_choice_matches_reference(weights):
    compared = 0
    for seed in SEEDS:
        game = SimGame(seed)
        for _ in range(SHOTS_PER_SEED):
            if game.is_over:
                break
            bubbles, shooter = game.bubbles(), game.shooter_color()
            expected = strategy.choose_target(shooter, bubbles, weights=weights)
            assert strategy_fast.choose_target(shooter, bubbles, weights=weights) == expected
            game.shoot(expected[:2])
            compared += 1
    assert compared > 300


def test_fast_handles_empty_board_and_unknown_shooter():
    assert strategy_fast.choose_target((0, 0, 0), []) is None
    bubbles = SimGame(1).bubbles()
    assert strategy_fast.choose_target(None, bubbles) == strategy.choose_target(None, bubbles)
