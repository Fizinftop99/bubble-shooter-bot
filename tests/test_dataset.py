import json

from bubbleshooter.config import StrategyWeights
from bubbleshooter.decider import cell_address
from bubbleshooter.sim import dataset
from bubbleshooter.sim.engine import SimGame
from bubbleshooter.sim.teacher import RolloutTeacher, TeacherConfig

CFG = TeacherConfig(top_k=3, rollouts=2, depth=2, fast_rule=True)
FIELDS = {
    "schema", "seed", "shot", "lives", "max_row", "bubble_count", "board", "shifted_rows", "shooter", "next", "candidates",
    "chosen", "rule_choice",
}
CANDIDATE_FIELDS = {"row", "col", "why", "rule_score", "wiped", "teacher_value", "click", "rollouts"}


def first_game(seed=1001, shots=6):
    return dataset.play_game(seed, shots, CFG)


def test_records_have_all_fields_and_valid_indices():
    _, _, records, _ = first_game()
    assert records
    for r in records:
        assert set(r) == FIELDS
        assert 2 <= len(r["candidates"]) <= CFG.top_k
        assert all(set(c) == CANDIDATE_FIELDS for c in r["candidates"])
        assert 0 <= r["chosen"] < len(r["candidates"])
        assert 0 <= r["rule_choice"] < len(r["candidates"])
        scores = [c["rule_score"] for c in r["candidates"]]
        assert scores[r["rule_choice"]] == max(scores)
        assert r["bubble_count"] == len(r["board"])


def test_same_seed_gives_identical_records():
    assert first_game() == first_game()


def test_board_matches_the_game_at_that_moment():
    game = SimGame(1001)
    teacher = RolloutTeacher(StrategyWeights(), CFG, 1001).bind(game)
    teacher.decide(game.shooter_color(), game.bubbles())
    record = dataset.make_record(1001, game, teacher)
    expected = {cell_address((*game.layout_position(cell), cell[1])) + (colour,) for cell, colour in game.board.items()}
    assert {tuple(t) for t in record["board"]} == expected
    assert {(row, col) for row, col, _ in record["board"]} == {(by, bx) for bx, by in game.board}
    assert record["shooter"] == game.ready and record["next"] == game.queue
    assert record["max_row"] == max(by for _, by in game.board)


def test_single_candidate_positions_are_skipped():
    game = SimGame(1001)
    teacher = RolloutTeacher(StrategyWeights(), CFG, 1001).bind(game)
    teacher.last_values = []
    assert dataset.make_record(1001, game, teacher) is None
    _, shots, records, skipped = dataset.play_game(1001, 6, TeacherConfig(top_k=1, rollouts=1, depth=1, fast_rule=True))
    assert records == [] and skipped == shots


def test_written_file_parses_line_by_line(tmp_path):
    _, _, records, _ = first_game(shots=3)
    path = tmp_path / "part.jsonl"
    path.write_text("".join(dataset.encode(r) + "\n" for r in records), encoding="utf-8")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(records)
    assert [json.loads(line) for line in lines] == json.loads(json.dumps(records))
