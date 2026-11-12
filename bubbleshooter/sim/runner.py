import re
import statistics
from dataclasses import dataclass, replace

from .. import strategy
from ..config import StrategyWeights, _from_env_values
from ..decider import DecisionLog, ModelDecider
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
    model_calls: int = 0
    model_shots: int = 0
    fallbacks: int = 0
    differs_from_rule: int = 0
    latency_ms_mean: float = 0.0
    latency_ms_p95: float = 0.0


def weights_from_overrides(overrides, base=None):
    return _from_env_values(base or StrategyWeights(), overrides)


def _model_stats(decider):
    latencies = sorted(getattr(decider, "latencies_ms", []))
    if not latencies:
        return {}
    return {
        "model_calls": len(latencies),
        "model_shots": decider.model_shots,
        "fallbacks": sum(decider.fallbacks.values()),
        "differs_from_rule": decider.disagreements,
        "latency_ms_mean": round(statistics.mean(latencies), 1),
        "latency_ms_p95": round(latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))], 1),
    }


def play(weights, seed, max_shots=DEFAULT_MAX_SHOTS, decider=None):
    game = SimGame(seed)
    pops = wipes = checked = landing_bad = pop_bad = max_row = 0
    colors = {c for c in game.board.values()}
    while not game.is_over and game.shots < max_shots:
        bubbles = game.bubbles()
        if decider is None:
            x, y, reason = strategy.choose_target(game.shooter_color(), bubbles, weights=weights)
        else:
            decision = decider.decide(game.shooter_color(), bubbles)
            x, y, reason = decision.x, decision.y, decision.reason
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
    return GameResult(
        seed, outcome, game.shots, game.final_score, max_row, game.pushes, pops, wipes, checked, landing_bad, pop_bad,
        **_model_stats(decider),
    )


def run_job(job):
    overrides, seed, max_shots, decider_settings = job
    weights = weights_from_overrides(overrides)
    decider = None
    if decider_settings is not None and decider_settings.decider == "model":
        decider = ModelDecider(weights, decider_settings, log=DecisionLog(decider_settings.decision_log_dir, f"sim_seed{seed}"))
    return play(weights, seed, max_shots, decider)


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
        **_model_summary(results),
    }


def _model_summary(results):
    calls = sum(r.model_calls for r in results)
    if not calls:
        return {}
    used = sum(r.model_shots for r in results)
    return {
        "model_calls": calls,
        "fallback_rate": round(sum(r.fallbacks for r in results) / calls, 3),
        "differs_from_rule": round(sum(r.differs_from_rule for r in results) / max(1, used), 3),
        "latency_ms_mean": round(sum(r.latency_ms_mean * r.model_calls for r in results) / calls),
        "latency_ms_p95_max": round(max(r.latency_ms_p95 for r in results)),
    }
