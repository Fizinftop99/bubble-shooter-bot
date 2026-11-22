from dataclasses import replace

from bubbleshooter.sim import teacher
from bubbleshooter.sim.engine import SimGame


def test_every_preset_builds_a_config_and_base_changes_nothing():
    for name, overrides in teacher.OUTCOMES.items():
        assert replace(teacher.TeacherConfig(), **overrides)
    assert teacher.OUTCOMES["base"] == {}


def test_noscore_preset_ignores_score_and_win_bonus():
    game = SimGame(1)
    cfg = replace(teacher.TeacherConfig(), **teacher.OUTCOMES["noscore"])
    game.score = 50_000
    value = teacher.outcome_value(game, 0, 0, cfg)
    assert abs(value - teacher.outcome_value(game, 50_000, 0, cfg)) < 1e-6
    base = teacher.outcome_value(game, 0, 0, teacher.TeacherConfig())
    assert base < value or base != value  # the base function does see the score
