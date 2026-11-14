"""Teacher dataset: one JSON line per decision with the board, the top-K candidates and the teacher's values.

Each record is enough to re-render a prompt and to build soft targets without the simulator.
Positions with a single candidate carry no learning signal and are not written (they are counted).

    uv run python -m bubbleshooter.sim.dataset --games 12 --seed-start 1001 --workers 12 --out datasets/teacher_s1001-1012.jsonl
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from ..decider import cell_address
from .engine import SimGame
from .runner import DEFAULT_MAX_SHOTS, weights_from_overrides
from .teacher import RolloutTeacher, TeacherConfig

DATASET_CONFIG = TeacherConfig(top_k=4, rollouts=5, depth=8, fast_rule=True)


def board_triples(game):
    """Occupied cells as sorted (row, col, colour_index), addressed like decider.cell_address."""
    return sorted([*cell_address((*game.layout_position(cell), cell[1])), colour] for cell, colour in game.board.items())


def make_record(seed, game, teacher):
    """Record for the position the teacher has just decided on (call before shooting). None for single-candidate positions."""
    values = teacher.last_values
    if len(values) < 2:
        return None
    scores = [candidate.score for candidate, _ in values]
    chosen = 0
    for i in range(1, len(values)):  # same tie-break as RolloutTeacher.decide with margin 0
        if values[i][1] > values[chosen][1]:
            chosen = i
    candidates = []
    for candidate, value in values:
        row, col = cell_address(candidate.cell)
        click_x, click_y, _ = candidate.aim()
        candidates.append(
            {
                "row": row,
                "col": col,
                "why": candidate.why,
                "rule_score": candidate.score,
                "wiped": candidate.wiped,
                "teacher_value": value,
                "click": [click_x, click_y],
            }
        )
    return {
        "seed": seed,
        "shot": game.shots,
        "lives": game.lives,
        "max_row": max(by for _, by in game.board),
        "bubble_count": len(game.board),
        "board": board_triples(game),
        "shifted_rows": sorted({by for _, by in game.board if by % 2 == 1}),
        "shooter": game.ready,
        "next": game.queue,
        "candidates": candidates,
        "chosen": chosen,
        "rule_choice": scores.index(max(scores)),
    }


def play_game(seed, max_shots=DEFAULT_MAX_SHOTS, cfg=DATASET_CONFIG):
    """Play one teacher game. Returns (outcome, shots, records, skipped_single_candidate)."""
    weights = weights_from_overrides({})
    game = SimGame(seed)
    teacher = RolloutTeacher(weights, cfg, seed).bind(game)
    records, skipped = [], 0
    while not game.is_over and game.shots < max_shots:
        teacher.last_values = []
        decision = teacher.decide(game.shooter_color(), game.bubbles())
        if decision is None:
            break
        record = make_record(seed, game, teacher)
        if record is None:
            skipped += 1
        else:
            chosen = record["candidates"][record["chosen"]]["click"]
            assert (decision.x, decision.y) == tuple(chosen), "chosen candidate does not match the teacher's shot"
            records.append(record)
        game.shoot((decision.x, decision.y))
    outcome = "won" if game.won else "lost" if game.lost else "limit"
    return outcome, game.shots, records, skipped


def _job(args):
    seed, max_shots, cfg = args
    started = time.time()
    outcome, shots, records, skipped = play_game(seed, max_shots, cfg)
    return seed, outcome, shots, records, skipped, time.time() - started


def encode(record):
    return json.dumps(record, separators=(",", ":"))


def main():
    parser = argparse.ArgumentParser(description="Generate teacher-labelled positions as JSON lines")
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-shots", type=int, default=2000)
    parser.add_argument("--top-k", type=int, default=DATASET_CONFIG.top_k)
    parser.add_argument("--rollouts", type=int, default=DATASET_CONFIG.rollouts)
    parser.add_argument("--depth", type=int, default=DATASET_CONFIG.depth)
    parser.add_argument("--slow-rule", action="store_true", help="use the pure-python rule inside rollouts instead of numba")
    parser.add_argument("--out", required=True, help="output .jsonl; include the seed range in the name; never overwritten")
    args = parser.parse_args()
    cfg = TeacherConfig(top_k=args.top_k, rollouts=args.rollouts, depth=args.depth, fast_rule=not args.slow_rule)
    seeds = list(range(args.seed_start, args.seed_start + args.games))
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"{out}: seeds {seeds[0]}-{seeds[-1]}, K={cfg.top_k} N={cfg.rollouts} D={cfg.depth} fast={cfg.fast_rule}, "
          f"max {args.max_shots} shots, {args.workers} workers", flush=True)
    started = time.time()
    outcomes, total, skipped_total, shots_total = {}, 0, 0, 0
    with out.open("x", encoding="utf-8", newline="\n") as fh, ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_job, (seed, args.max_shots, cfg)) for seed in seeds]
        for future in as_completed(futures):
            seed, outcome, shots, records, skipped, seconds = future.result()
            fh.write("".join(encode(r) + "\n" for r in records))
            fh.flush()
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
            total += len(records)
            skipped_total += skipped
            shots_total += shots
            print(f"  seed {seed}: {outcome:5} shots {shots:4d} records {len(records):4d} ({seconds:.0f}s)", flush=True)
    print(f"games {len(seeds)}, shots {shots_total}, records {total}, single-candidate skipped {skipped_total}, "
          f"outcomes {outcomes}, {time.time() - started:.0f}s wall -> {out}", flush=True)


if __name__ == "__main__":
    main()
