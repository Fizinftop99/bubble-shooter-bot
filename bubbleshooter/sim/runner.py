import re
import statistics
from dataclasses import dataclass, replace

from .. import strategy
from ..config import StrategyWeights, _from_env_values
from .engine import SimGame

DEFAULT_MAX_SHOTS = 2000
CELL_PATTERN = re.compile(r"at cell \((\d+), (\d+)\)")


@dataclass
class GameResult:
    seed: int
    outcome: str
    shots: int
    score: int
    max_row: int
    pushes: int
    pops: int
    wipes: int
    landing_checked: int
    landing_mismatch: int
    pop_mismatch: int


def weights_from_overrides(overrides, base=None):
    return _from_env_values(base or StrategyWeights(), overrides)


def play(weights, seed, max_shots=DEFAULT_MAX_SHOTS):
    game = SimGame(seed)
    pops = wipes = checked = landing_bad = pop_bad = max_row = 0
    colors = {c for c in game.board.values()}
    while not game.is_over and game.shots < max_shots:
        bubbles = game.bubbles()
        x, y, reason = strategy.choose_target(game.shooter_color(), bubbles, weights=weights)
        popped = game.shoot((x, y))
        pops += popped
        match = CELL_PATTERN.search(reason)
        if match:
            checked += 1
            landing_bad += (int(match[1]), int(match[2])) != game.layout_position(game.last_cell)
            pop_bad += reason.startswith("pop") != popped
        now = set(game.board.values())
        wipes += len(colors - now)
        colors = now
        if game.board:
            max_row = max(max_row, max(by for _, by in game.board))
    outcome = "won" if game.won else "lost" if game.lost else "limit"
    return GameResult(seed, outcome, game.shots, game.final_score, max_row, game.pushes, pops, wipes, checked, landing_bad, pop_bad)


def run_job(job):
    overrides, seed, max_shots = job
    return play(weights_from_overrides(overrides), seed, max_shots)


def summarize(results):
    n = len(results)
    shots = [r.shots for r in results]
    return {
        "games": n,
        "won": sum(r.outcome == "won" for r in results),
        "lost": sum(r.outcome == "lost" for r in results),
        "limit": sum(r.outcome == "limit" for r in results),
        "median_shots": statistics.median(shots),
        "mean_shots": round(statistics.mean(shots), 1),
        "mean_score": round(statistics.mean(r.score for r in results)),
        "lost_before_100": sum(r.outcome == "lost" and r.shots < 100 for r in results),
        "pop_rate": round(sum(r.pops for r in results) / max(1, sum(r.shots for r in results)), 3),
        "landing_mismatch": round(sum(r.landing_mismatch for r in results) / max(1, sum(r.landing_checked for r in results)), 3),
        "pop_mismatch": round(sum(r.pop_mismatch for r in results) / max(1, sum(r.landing_checked for r in results)), 3),
    }
