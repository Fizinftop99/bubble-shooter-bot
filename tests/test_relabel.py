import json
import statistics
from dataclasses import replace

import pytest

from bubbleshooter.config import StrategyWeights
from bubbleshooter.sim import dataset, relabel, teacher
from bubbleshooter.sim.engine import SimGame

CFG = teacher.TeacherConfig(top_k=4, rollouts=3, depth=3, fast_rule=True, **teacher.OUTCOMES["noscore"])


def old_outcome_value(game, start_score, start_shots, cfg):
    """The formula as it was before the outcome was split from its weighting (kept here as the reference)."""
    gained = (game.final_score - start_score) / cfg.score_unit
    if game.lost:
        return gained - cfg.loss_penalty + cfg.survive_weight * (game.shots - start_shots)
    if game.won:
        return gained + cfg.win_bonus
    front = max(by for _, by in game.board)
    return gained - cfg.row_weight * front - cfg.bubble_weight * len(game.board) + cfg.life_weight * game.lives


def game_states():
    alive = SimGame(1)
    for _ in range(6):
        alive.shoot((300, 200))
    lost = SimGame(2)
    lost.shots, lost.score, lost.lost = 7, 1230, True
    won = SimGame(3)
    won.board, won.score, won.shots, won.won = {}, 4560, 9, True
    return alive, lost, won


@pytest.mark.parametrize("preset", sorted(teacher.OUTCOMES))
def test_value_of_matches_the_old_formula(preset):
    cfg = relabel.config_for(preset)
    for game in game_states():
        for start_score, start_shots in ((0, 0), (500, 3)):
            outcome = teacher.rollout_outcome(game, start_score, start_shots)
            assert teacher.value_of(outcome, cfg) == old_outcome_value(game, start_score, start_shots, cfg)
            assert teacher.outcome_value(game, start_score, start_shots, cfg) == old_outcome_value(game, start_score, start_shots, cfg)


def test_rollout_outcome_fields():
    alive, lost, won = game_states()
    assert teacher.rollout_outcome(lost, 0, 0)[1] == teacher.LOST
    assert teacher.rollout_outcome(won, 0, 0) == (2 * 4560, teacher.WON, 9, 0, 0, won.lives)
    points, status, shots, front, bubbles, lives = teacher.rollout_outcome(alive, 0, 0)
    assert status == teacher.ALIVE and shots == 6 and bubbles == len(alive.board) and front == max(by for _, by in alive.board)


def test_teacher_values_are_the_mean_of_its_stored_outcomes():
    game = SimGame(11)
    bot = teacher.RolloutTeacher(StrategyWeights(), CFG, 11).bind(game)
    for _ in range(5):
        decision = bot.decide(game.shooter_color(), game.bubbles())
        if bot.last_values:
            assert len(bot.last_outcomes) == len(bot.last_values)
            for (_, value), runs in zip(bot.last_values, bot.last_outcomes):
                assert len(runs) == CFG.rollouts
                assert value == statistics.fmean([teacher.value_of(r, CFG) for r in runs])
        game.shoot((decision.x, decision.y))


def records(seed=1001, shots=12, cfg=CFG):
    return dataset.play_game(seed, shots, cfg)[2]


def test_records_carry_raw_rollouts_that_reproduce_the_labels():
    recs = records()
    assert recs
    for r in recs:
        assert r["schema"] == dataset.SCHEMA == relabel.SCHEMA
        for c in r["candidates"]:
            assert len(c["rollouts"]) == CFG.rollouts
            for run in c["rollouts"]:
                assert len(run) == 6 and all(isinstance(x, int) for x in run) and run[1] in (0, 1, 2)
        values = relabel.candidate_values(r, CFG)
        assert values == [c["teacher_value"] for c in r["candidates"]]
        assert relabel.first_argmax(values) == r["chosen"]
        assert all(se >= 0 for se in relabel.standard_errors(r, CFG))


def test_labels_survive_a_json_round_trip():
    for r in records(shots=6):
        again = json.loads(dataset.encode(r))
        assert relabel.candidate_values(again, CFG) == [c["teacher_value"] for c in r["candidates"]]


def test_another_outcome_function_gives_other_values_where_points_were_scored():
    recs = records(shots=15)
    scored = [r for r in recs if any(run[0] for c in r["candidates"] for run in c["rollouts"])]
    assert scored, "no rollout scored points; pick a longer game"
    base = relabel.config_for("base")
    assert any(relabel.candidate_values(r, base) != relabel.candidate_values(r, CFG) for r in scored)


def test_old_schema_is_refused():
    record = records(shots=3)[0]
    del record["schema"]
    with pytest.raises(ValueError, match="schema"):
        relabel.candidate_values(record, CFG)


def test_meta_round_trips_to_the_generating_config(tmp_path):
    meta = dataset.make_meta(CFG, "noscore", [4001, 4008], 1200)
    assert meta["schema"] == dataset.SCHEMA and meta["seeds"] == [4001, 4008] and meta["outcome"] == "noscore"
    path = tmp_path / "x.meta.json"
    path.write_text(json.dumps(meta), encoding="utf-8")
    assert relabel.config_from_meta(path) == CFG
    assert dataset.meta_path(tmp_path / "teacher_v2_s1-8.jsonl").name == "teacher_v2_s1-8.meta.json"


def test_dataset_default_plays_with_noscore():
    assert dataset.DATASET_CONFIG == replace(
        teacher.TeacherConfig(top_k=4, rollouts=10, depth=8, fast_rule=True), **teacher.OUTCOMES["noscore"]
    )
