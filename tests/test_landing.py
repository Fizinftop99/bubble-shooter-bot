from collections import Counter

from bubbleshooter import landing, strategy_fast
from bubbleshooter.landing import check
from bubbleshooter.sim.engine import SimGame

RED, BLUE = (188, 1, 0), (0, 21, 188)


def bubble(x, y, color=RED):
    return {"x": x, "y": y, "color": color, "row": 0, "shifted": False}


def test_reason_parsing():
    p = landing.Prediction.from_reason("pop 5 drop 2 at cell (204, 307) angle 109.5")
    assert p == landing.Prediction((204, 307), True, 6)
    p = landing.Prediction.from_reason("no pop (joins 1 same-colour) at cell (414, 307) angle 57.0")
    assert p == landing.Prediction((414, 307), False, 0)
    assert landing.Prediction.from_reason("fallback: no simulated shot, aiming at top row").cell is None


def test_synthetic_outcomes():
    board = [bubble(39, 37), bubble(69, 37), bubble(99, 37, BLUE)]
    stuck = board + [bubble(54, 67)]
    assert check(board, stuck, "no pop (joins 0) at cell (54, 67) angle 90")["outcome"] == "as_predicted"
    other = check(board, stuck, "no pop (joins 0) at cell (84, 67) angle 90")
    assert other["outcome"] == "other_cell" and other["actual_cell"] == (54, 67) and other["miss_px"] == 30
    assert check(board, stuck, "pop 3 drop 0 at cell (54, 67) angle 90")["outcome"] == "missed_pop"
    popped = [bubble(99, 37, BLUE)]
    assert check(board, popped, "pop 3 drop 0 at cell (54, 67) angle 90")["outcome"] == "as_predicted"
    assert check(board, popped, "pop 4 drop 0 at cell (54, 67) angle 90")["outcome"] == "pop_count_off"
    assert check(board, popped, "no pop (joins 1) at cell (54, 67) angle 90")["outcome"] == "unexpected_pop"
    assert check(board, board, "pop 3 drop 0 at cell (54, 67) angle 90")["outcome"] == "no_change"
    pushed = [bubble(39 + 30 * i, 37, BLUE) for i in range(17)] + [bubble(54 + 30 * i, 67) for i in range(3)]
    assert check(board, pushed, "no pop (joins 0) at cell (54, 97) angle 90")["outcome"] == "push"


def test_matches_simulator_truth():
    """On simulated games the check must recover exactly what the engine did with each shot."""
    outcomes = Counter()
    for seed in range(1, 9):
        game = SimGame(seed)
        while not game.is_over and game.shots < 300:
            before = game.bubbles()
            x, y, reason = strategy_fast.choose_target(game.shooter_color(), before)
            pushes = game.pushes
            popped = game.shoot((x, y))
            after = game.bubbles()
            record = check(before, after, reason)
            outcomes[record["outcome"]] += 1
            if game.pushes != pushes:
                assert record["outcome"] == "push"
            elif not popped:
                assert record["actual_cell"] == game.layout_position(game.last_cell)
                assert record["removed"] == 0
            else:
                assert record["added"] == 0 and record["removed"] == len(before) - len(after)
                assert record["outcome"] in ("as_predicted", "pop_count_off", "unexpected_pop")
    # The rule's ray model and the engine's frame-by-frame flight disagree on about 11 % of shots
    # (other_cell / missed_pop), so this is a floor on the rule, not on the check.
    judged = sum(outcomes.values()) - outcomes["push"]
    assert outcomes["as_predicted"] / judged > 0.85, outcomes
    assert outcomes["no_change"] == 0 and outcomes["unclear"] == 0


def test_log_writes_rows_and_frames(tmp_path):
    from PIL import Image

    log = landing.LandingLog(tmp_path)
    img = Image.new("RGB", (10, 10))
    board = [bubble(39, 37)]
    log.write(1, check(board, board + [bubble(54, 67)], "no pop at cell (54, 67) angle 90"), img, img, reason="r")
    log.write(2, check(board, board, "no pop at cell (54, 67) angle 90"), img, img, reason="r")
    rows = log.path.read_text().splitlines()
    assert len(rows) == 2 and '"frames"' in rows[1] and '"frames"' not in rows[0]
    assert sorted(p.name for p in log.frames_dir.iterdir()) == ["shot0002_after.png", "shot0002_before.png"]
    assert "50.0% as predicted of 2 judged" in log.summary()
