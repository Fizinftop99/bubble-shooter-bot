"""Teacher dataset: one JSON line per decision with the board, the top-K candidates and the teacher's values.

Each record is enough to re-render a prompt and to build soft targets without the simulator.
Positions with a single candidate carry no learning signal and are not written (they are counted).

Schema 2: every candidate also keeps the raw result of each rollout, [points, status, shots, front, bubbles, lives]
(teacher.rollout_outcome), so labels can be recomputed for any outcome function and their noise estimated from the
spread of rollouts (sim/relabel.py). A <out>.meta.json next to the file records how it was made.

    uv run --group fast python -m bubbleshooter.sim.dataset --games 8 --seed-start 4001 --workers 8 --max-shots 1200 \
        --out datasets/teacher_v2_noscore_n10_s4001-4008.jsonl
"""
import argparse
import dataclasses
import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from ..decider import cell_address
from .engine import SimGame
from .runner import DEFAULT_MAX_SHOTS, weights_from_overrides
from .teacher import OUTCOMES, RolloutTeacher, TeacherConfig

SCHEMA = 2
DATASET_OUTCOME = "noscore"  # what the teacher optimises while playing the games; labels can be recomputed for others
DATASET_CONFIG = TeacherConfig(top_k=4, rollouts=10, depth=8, fast_rule=True, **OUTCOMES[DATASET_OUTCOME])


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
    for i, (candidate, value) in enumerate(values):
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
                "rollouts": [list(outcome) for outcome in teacher.last_outcomes[i]],
            }
        )
    return {
        "schema": SCHEMA,
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


def meta_path(out):
    out = Path(out)
    return out.with_name(out.stem + ".meta.json")


def _git_state():
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return commit, dirty


def make_meta(cfg, outcome, seeds, max_shots):
    commit, dirty = _git_state()
    return {
        "schema": SCHEMA,
        "outcome": outcome,
        "teacher_config": dataclasses.asdict(cfg),
        "seeds": [seeds[0], seeds[-1]],
        "max_shots": max_shots,
        "git_commit": commit,
        "git_dirty": dirty,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


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
    parser.add_argument("--outcome", choices=sorted(OUTCOMES), default=DATASET_OUTCOME,
                        help="outcome function the teacher plays with (raw rollouts are stored either way)")
    parser.add_argument("--out", required=True, help="output .jsonl; include the seed range in the name; never overwritten")
    args = parser.parse_args()
    cfg = TeacherConfig(
        top_k=args.top_k, rollouts=args.rollouts, depth=args.depth, fast_rule=not args.slow_rule, **OUTCOMES[args.outcome]
    )
    seeds = list(range(args.seed_start, args.seed_start + args.games))
    out = Path(args.out)
    meta = meta_path(out)
    for path in (out, meta):
        if path.exists():
            raise SystemExit(f"refusing to overwrite {path}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with meta.open("x", encoding="utf-8") as fh:
        json.dump(make_meta(cfg, args.outcome, seeds, args.max_shots), fh, indent=2)
    print(f"{out}: seeds {seeds[0]}-{seeds[-1]}, K={cfg.top_k} N={cfg.rollouts} D={cfg.depth} fast={cfg.fast_rule}, "
          f"outcome {args.outcome}, max {args.max_shots} shots, {args.workers} workers", flush=True)
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
