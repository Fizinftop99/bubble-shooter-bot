from bubbleshooter.sim import noise

CFG = noise.TeacherConfig(top_k=3, rollouts=2, depth=2, fast_rule=False)


def test_records_hold_two_independent_labellings_of_the_same_candidates():
    outcome, shots, records = noise.play_game(3, 12, CFG)
    assert shots == 12 or outcome != "limit"
    assert records
    for r in records:
        assert len(r["values_a"]) == len(r["values_b"]) == len(r["cells"]) == len(r["rule_scores"]) == 3
        assert r["chosen_a"] == noise.first_argmax(r["values_a"])
        assert r["chosen_b"] == noise.first_argmax(r["values_b"])
    # independent rollouts: the two value vectors must not be identical everywhere
    assert any(r["values_a"] != r["values_b"] for r in records)


def test_same_seed_gives_the_same_records():
    assert noise.play_game(2, 8, CFG)[2] == noise.play_game(2, 8, CFG)[2]


def test_first_argmax_takes_the_first_of_equal_values():
    assert noise.first_argmax([1.0, 3.0, 3.0, 0.0]) == 1
