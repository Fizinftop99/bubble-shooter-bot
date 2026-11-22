from bubbleshooter.config import StrategyWeights
from bubbleshooter.sim import runner


def test_no_jitter_and_any_click_keep_the_old_game():
    a = runner.play(StrategyWeights(), 3, max_shots=60)
    b = runner.play(StrategyWeights(), 3, max_shots=60, jitter=0, click_area=False)
    assert a == b


def test_jitter_is_reproducible_and_changes_the_game():
    a = runner.play(StrategyWeights(), 3, max_shots=60, jitter=5)
    b = runner.play(StrategyWeights(), 3, max_shots=60, jitter=5)
    exact = runner.play(StrategyWeights(), 3, max_shots=60)
    assert a == b
    assert a != exact
