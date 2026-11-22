import json

from bubbleshooter.config import StrategyWeights
from bubbleshooter.decider import build_prompt, cell_address
from bubbleshooter.sim import dataset
from bubbleshooter.sim.engine import SimGame
from bubbleshooter.sim.prompts import answer_from_record, prompt_from_record
from bubbleshooter.sim.teacher import RolloutTeacher, TeacherConfig

CFG = TeacherConfig(top_k=4, rollouts=2, depth=2, fast_rule=True)


def play_and_compare(seed, shots):
    """At every decision: the prompt rebuilt from the record must equal the prompt the live decider would build."""
    game = SimGame(seed)
    teacher = RolloutTeacher(StrategyWeights(), CFG, seed).bind(game)
    compared = 0
    while not game.is_over and game.shots < shots:
        teacher.last_values = []
        top = teacher.top_candidates(game.shooter_color(), game.bubbles())
        live = build_prompt(game.shooter_color(), game.bubbles(), top, game.next_color(), game.lives)
        decision = teacher.decide(game.shooter_color(), game.bubbles())
        record = dataset.make_record(seed, game, teacher)
        if record is not None:
            assert prompt_from_record(record) == live
            compared += 1
        game.shoot((decision.x, decision.y))
    return compared


def test_prompt_from_record_equals_the_live_prompt():
    assert sum(play_and_compare(seed, 40) for seed in (1001, 1002, 1003)) > 30


def test_answer_names_a_listed_cell_and_is_valid_json():
    _, _, records, _ = dataset.play_game(1001, 10, CFG)
    for r in records:
        answer = json.loads(answer_from_record(r))
        listed = {(c["row"], c["col"]) for c in r["candidates"]}
        assert (answer["row"], answer["col"]) in listed
        assert f"- row {answer['row']} col {answer['col']}:" in prompt_from_record(r)


def test_candidate_cells_are_empty_and_addresses_round_trip():
    _, _, records, _ = dataset.play_game(1002, 20, CFG)
    for r in records:
        taken = {(row, col) for row, col, _ in r["board"]}
        for c in r["candidates"]:
            assert (c["row"], c["col"]) not in taken
    assert cell_address((39 + 30 * 3, 37, 0)) == (0, 3)
