"""Rollout teacher: picks a shot by playing each of the rule's top candidates forward.

For the current position the teacher takes the top-K candidates by rule score. Each one is
shot on a copy of the live game, the copy then plays D more shots with the rule, and this is
repeated N times with different random futures. The candidate with the best mean outcome wins.

The copies get a fresh RNG, so the teacher never sees the real game's upcoming colours beyond
what the player can see too (the bubble in hand and the next one in the queue).
"""
import argparse
import copy
import csv
import random
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from .. import strategy
from ..decider import Decision
from .engine import SimGame
from .runner import CELL_PATTERN, DEFAULT_MAX_SHOTS, GameResult, play, summarize, weights_from_overrides

RESULTS_DIR = Path("sim_results")
SEED_STRIDE = 1_000_003


@dataclass(frozen=True)
class TeacherConfig:
    top_k: int = 4
    rollouts: int = 4
    depth: int = 8
    margin: float = 0.0
    score_unit: float = 100.0
    row_weight: float = 3.0
    bubble_weight: float = 0.1
    life_weight: float = 0.5
    loss_penalty: float = 200.0
    win_bonus: float = 200.0
    survive_weight: float = 2.0


def clone_game(game, seed):
    twin = copy.copy(game)
    twin.board = dict(game.board)
    twin.rng = random.Random(seed)
    return twin


def outcome_value(game, start_score, start_shots, cfg):
    """Higher is better: score gained, a higher front line, fewer bubbles, more lives; losing is heavily punished."""
    gained = (game.final_score - start_score) / cfg.score_unit
    if game.lost:
        return gained - cfg.loss_penalty + cfg.survive_weight * (game.shots - start_shots)
    if game.won:
        return gained + cfg.win_bonus
    front = max(by for _, by in game.board)
    return gained - cfg.row_weight * front - cfg.bubble_weight * len(game.board) + cfg.life_weight * game.lives


def rollout(game, click, seed, weights, cfg, memo=None):
    """Shoot `click` on a reseeded copy, then play `depth` rule shots. `memo` reuses rule answers for identical positions."""
    twin = clone_game(game, seed)
    twin.shoot(click)
    for _ in range(cfg.depth):
        if twin.is_over:
            break
        key = (twin.ready, frozenset(twin.board.items())) if memo is not None else None
        target = memo.get(key) if memo is not None else None
        if target is None:
            target = strategy.choose_target(twin.shooter_color(), twin.bubbles(), weights=weights)
            if target is None:
                break
            if memo is not None:
                memo[key] = target
        twin.shoot(target[:2])
    return outcome_value(twin, game.score, game.shots, cfg)


class RolloutTeacher:
    name = "teacher"

    def __init__(self, weights, cfg=TeacherConfig(), seed=0):
        self._weights = weights
        self._cfg = cfg
        self._seed = seed
        self._game = None
        self.latencies_ms = []
        self.fallbacks = {}
        self.model_shots = 0
        self.disagreements = 0
        self.last_values = []

    def bind(self, game):
        self._game = game
        return self

    def top_candidates(self, shooter_color, bubbles):
        candidates = strategy.list_candidates(shooter_color, bubbles, weights=self._weights)
        return sorted(candidates, key=lambda c: -c.score)[: self._cfg.top_k]

    def decide(self, shooter_color, bubbles):
        game = self._game
        if game is None:
            raise RuntimeError("RolloutTeacher needs bind(game) before decide()")
        started = time.perf_counter()
        top = self.top_candidates(shooter_color, bubbles)
        if not top:
            result = strategy.choose_target(shooter_color, bubbles, weights=self._weights)
            return None if result is None else Decision(*result)
        if len(top) == 1:
            self.last_values = []
            return Decision(*top[0].aim(), source="rule", agrees_with_rule=True)
        base = (self._seed * SEED_STRIDE + game.shots) * SEED_STRIDE
        values, memo = [], {}
        for candidate in top:
            click = candidate.aim()[:2]
            runs = [rollout(game, click, base + i, self._weights, self._cfg, memo) for i in range(self._cfg.rollouts)]
            values.append(statistics.fmean(runs))
        best = 0
        for i in range(1, len(top)):
            if values[i] > values[best]:
                best = i
        if values[best] - values[0] <= self._cfg.margin:
            best = 0
        self.last_values = list(zip(top, values))
        latency_ms = (time.perf_counter() - started) * 1000
        self.latencies_ms.append(latency_ms)
        self.model_shots += 1
        self.disagreements += best != 0
        return Decision(*top[best].aim(), source="teacher", latency_ms=latency_ms, agrees_with_rule=best == 0)


def play_teacher(weights, seed, max_shots=DEFAULT_MAX_SHOTS, cfg=TeacherConfig()):
    """Same loop and bookkeeping as runner.play, but the decider is bound to the live game."""
    game = SimGame(seed)
    teacher = RolloutTeacher(weights, cfg, seed).bind(game)
    pops = wipes = checked = landing_bad = pop_bad = max_row = 0
    colors = set(game.board.values())
    while not game.is_over and game.shots < max_shots:
        decision = teacher.decide(game.shooter_color(), game.bubbles())
        popped = game.shoot((decision.x, decision.y))
        pops += popped
        match = CELL_PATTERN.search(decision.reason)
        if match:
            checked += 1
            landing_bad += (int(match[1]), int(match[2])) != game.layout_position(game.last_cell)
            pop_bad += decision.reason.startswith("pop") != popped
        now = set(game.board.values())
        wipes += len(colors - now)
        colors = now
        if game.board:
            max_row = max(max_row, max(by for _, by in game.board))
    outcome = "won" if game.won else "lost" if game.lost else "limit"
    latencies = sorted(teacher.latencies_ms)
    stats = {}
    if latencies:
        stats = {
            "model_calls": len(latencies),
            "model_shots": teacher.model_shots,
            "differs_from_rule": teacher.disagreements,
            "latency_ms_mean": round(statistics.mean(latencies), 1),
            "latency_ms_p95": round(latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))], 1),
        }
    return GameResult(
        seed, outcome, game.shots, game.final_score, max_row, game.pushes, pops, wipes, checked, landing_bad, pop_bad, **stats
    )


def _job(job):
    kind, seed, max_shots, cfg = job
    weights = weights_from_overrides({})
    started = time.time()
    result = play(weights, seed, max_shots) if kind == "rule" else play_teacher(weights, seed, max_shots, cfg)
    return kind, result, time.time() - started


def run(seeds, max_shots, cfg, workers, with_rule=True):
    jobs = [("teacher", seed, max_shots, cfg) for seed in seeds]
    if with_rule:
        jobs = [("rule", seed, max_shots, cfg) for seed in seeds] + jobs
    results = {"rule": [], "teacher": []}
    if workers <= 1:
        done = map(_job, jobs)
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        done = (future.result() for future in as_completed([pool.submit(_job, job) for job in jobs]))
    for kind, result, seconds in done:
        results[kind].append(result)
        if kind == "teacher":
            print(
                f"  seed {result.seed:3d}: {result.outcome:5} shots {result.shots:4d} score {result.score:7d} "
                f"differs {result.differs_from_rule:3d} {result.latency_ms_mean:6.0f} ms/decision ({seconds:.0f}s)",
                flush=True,
            )
    for group in results.values():
        group.sort(key=lambda r: r.seed)
    return results


def save_csv(name, results):
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with path.open("x", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(results[0])))
        writer.writeheader()
        writer.writerows(asdict(r) for r in results)
    return path


def print_summary(label, results):
    print(f"[{label}]")
    for key, value in summarize(results).items():
        print(f"  {key:18} {value}")


def print_paired(rule, teacher):
    by_seed = {r.seed: r for r in rule}
    print("paired by seed (rule -> teacher):")
    longer = shorter = higher = lower = 0
    for t in teacher:
        r = by_seed.get(t.seed)
        if r is None:
            continue
        longer += t.shots > r.shots
        shorter += t.shots < r.shots
        higher += t.score > r.score
        lower += t.score < r.score
        verdict = "teacher longer" if t.shots > r.shots else "rule longer" if t.shots < r.shots else "same length"
        print(
            f"  seed {t.seed:3d}: {r.outcome:5} {r.shots:4d} shots {r.score:7d}  ->  "
            f"{t.outcome:5} {t.shots:4d} shots {t.score:7d}   {verdict}, max row {r.max_row} -> {t.max_row}, "
            f"pushes {r.pushes} -> {t.pushes}"
        )
    n = len(teacher)
    print(f"  teacher lasted longer in {longer} games, shorter in {shorter}, equal in {n - longer - shorter}")
    print(f"  teacher scored higher in {higher} games, lower in {lower}, equal in {n - higher - lower}")


def main():
    parser = argparse.ArgumentParser(description="Play the rollout teacher in the headless simulator and compare it with the rule")
    parser.add_argument("--games", type=int, default=12)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--max-shots", type=int, default=400)
    parser.add_argument("--top-k", type=int, default=TeacherConfig.top_k)
    parser.add_argument("--rollouts", type=int, default=TeacherConfig.rollouts)
    parser.add_argument("--depth", type=int, default=TeacherConfig.depth)
    parser.add_argument("--margin", type=float, default=TeacherConfig.margin, help="keep the rule's shot unless the teacher's is better by more than this")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--no-rule", action="store_true", help="skip the rule run on the same seeds")
    args = parser.parse_args()
    cfg = TeacherConfig(top_k=args.top_k, rollouts=args.rollouts, depth=args.depth, margin=args.margin)
    seeds = range(args.seed_start, args.seed_start + args.games)
    label = f"teacher_k{cfg.top_k}_n{cfg.rollouts}_d{cfg.depth}" + (f"_m{cfg.margin:g}" if cfg.margin else "")
    print(f"{label}: seeds {seeds.start}-{seeds.stop - 1}, max {args.max_shots} shots, {args.workers} workers", flush=True)
    started = time.time()
    results = run(seeds, args.max_shots, cfg, args.workers, with_rule=not args.no_rule)
    elapsed = time.time() - started
    if results["rule"]:
        print_summary("rule", results["rule"])
        print(f"  csv: {save_csv('teacher_rulebase', results['rule'])}")
    teacher = results["teacher"]
    print_summary(label, teacher)
    print(f"  csv: {save_csv(label, teacher)}  ({elapsed:.0f}s wall)")
    if results["rule"]:
        print_paired(results["rule"], teacher)


if __name__ == "__main__":
    main()
