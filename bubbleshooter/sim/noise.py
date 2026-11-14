"""Measure label noise of the rollout teacher by labelling every position twice.

Teacher A plays the game. At each decision an independent teacher B (different random futures in its
rollouts, same top-K candidates, same settings) values the same candidates. Comparing the two value
vectors shows how often the teacher's choice flips from rollout noise alone.

    uv run python -m bubbleshooter.sim.noise --games 12 --seed-start 2001 --max-shots 500 --workers 12 --out datasets/noise_s2001-2012.jsonl
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from ..decider import cell_address
from .dataset import DATASET_CONFIG
from .engine import SimGame
from .runner import weights_from_overrides
from .teacher import RolloutTeacher, TeacherConfig

SECOND_TEACHER_SEED_OFFSET = 90_000


def first_argmax(values):
    best = 0
    for i in range(1, len(values)):
        if values[i] > values[best]:
            best = i
    return best


def make_record(seed, game, teacher_a, teacher_b):
    """Both teachers must have just decided on the current position. None for single-candidate positions."""
    a, b = teacher_a.last_values, teacher_b.last_values
    if len(a) < 2:
        return None
    cells_a = [cell_address(c.cell) for c, _ in a]
    cells_b = [cell_address(c.cell) for c, _ in b]
    assert cells_a == cells_b, "the two teachers saw different candidates"
    values_a, values_b = [v for _, v in a], [v for _, v in b]
    scores = [c.score for c, _ in a]
    return {
        "seed": seed,
        "shot": game.shots,
        "cells": [list(c) for c in cells_a],
        "why": [c.why for c, _ in a],
        "rule_scores": scores,
        "values_a": values_a,
        "values_b": values_b,
        "chosen_a": first_argmax(values_a),
        "chosen_b": first_argmax(values_b),
        "rule_choice": scores.index(max(scores)),
    }


def play_game(seed, max_shots, cfg=DATASET_CONFIG):
    weights = weights_from_overrides({})
    game = SimGame(seed)
    teacher_a = RolloutTeacher(weights, cfg, seed).bind(game)
    teacher_b = RolloutTeacher(weights, cfg, seed + SECOND_TEACHER_SEED_OFFSET).bind(game)
    records = []
    while not game.is_over and game.shots < max_shots:
        teacher_a.last_values = []
        teacher_b.last_values = []
        shooter, bubbles = game.shooter_color(), game.bubbles()
        decision = teacher_a.decide(shooter, bubbles)
        if decision is None:
            break
        teacher_b.decide(shooter, bubbles)
        record = make_record(seed, game, teacher_a, teacher_b)
        if record is not None:
            records.append(record)
        game.shoot((decision.x, decision.y))
    outcome = "won" if game.won else "lost" if game.lost else "limit"
    return outcome, game.shots, records


def _job(args):
    seed, max_shots, cfg = args
    started = time.time()
    outcome, shots, records = play_game(seed, max_shots, cfg)
    return seed, outcome, shots, records, time.time() - started


def main():
    parser = argparse.ArgumentParser(description="Label every position twice with independent rollouts to measure label noise")
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-shots", type=int, default=500, help="two teachers per decision: keep games short")
    parser.add_argument("--top-k", type=int, default=DATASET_CONFIG.top_k)
    parser.add_argument("--rollouts", type=int, default=DATASET_CONFIG.rollouts)
    parser.add_argument("--depth", type=int, default=DATASET_CONFIG.depth)
    parser.add_argument("--out", required=True, help="output .jsonl; never overwritten")
    args = parser.parse_args()
    cfg = TeacherConfig(top_k=args.top_k, rollouts=args.rollouts, depth=args.depth, fast_rule=True)
    seeds = list(range(args.seed_start, args.seed_start + args.games))
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"{out}: seeds {seeds[0]}-{seeds[-1]}, K={cfg.top_k} N={cfg.rollouts} D={cfg.depth}, max {args.max_shots} shots, "
          f"{args.workers} workers", flush=True)
    started, total = time.time(), 0
    with out.open("x", encoding="utf-8", newline="\n") as fh, ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_job, (seed, args.max_shots, cfg)) for seed in seeds]
        for future in as_completed(futures):
            seed, outcome, shots, records, seconds = future.result()
            fh.write("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in records))
            fh.flush()
            total += len(records)
            print(f"  seed {seed}: {outcome:5} shots {shots:4d} records {len(records):4d} ({seconds:.0f}s)", flush=True)
    print(f"games {len(seeds)}, records {total}, {time.time() - started:.0f}s wall -> {out}", flush=True)


if __name__ == "__main__":
    main()
