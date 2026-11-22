import random
import re
import statistics
from dataclasses import dataclass, replace

from .. import bot, strategy
from ..config import StrategyWeights, _from_env_values
from ..decider import DecisionLog, ModelDecider
from .engine import SimGame

DEFAULT_MAX_SHOTS = 2000
STUCK_LIMIT = 8  # game.STALL_LIMIT: the real bot gives up after this many clicks that change nothing
JITTER_SEED_SALT = 0x5EED_C11C  # the click jitter has its own generator so the game's colours stay the same
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
    ignored_clicks: int = 0
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


def play(weights, seed, max_shots=DEFAULT_MAX_SHOTS, decider=None, jitter=0, click_area=False):
    """One game. jitter > 0 moves every click by up to that many pixels, as bot.HumanAimer does in the real game.
    click_area=True drops clicks outside the game's clickable rectangle; STUCK_LIMIT of them in a row end the
    game as "stuck"."""
    game = SimGame(seed, click_area=click_area)
    ignored = ignored_streak = 0
    click_rng = random.Random(seed ^ JITTER_SEED_SALT)
    pops = wipes = checked = landing_bad = pop_bad = max_row = 0
    colors = {c for c in game.board.values()}
    while not game.is_over and game.shots < max_shots:
        bubbles = game.bubbles()
        if decider is None:
            x, y, reason = strategy.choose_target(game.shooter_color(), bubbles, weights=weights)
        else:
            decision = decider.decide(game.shooter_color(), bubbles, game.next_color(), game.lives)
            x, y, reason = decision.x, decision.y, decision.reason
        if jitter:
            x, y = bot.jittered_point({"x": 0, "y": 0}, (x, y), jitter, click_rng)
        popped = game.shoot((x, y))
        if popped is None:
            ignored += 1
            ignored_streak += 1
            if ignored_streak >= STUCK_LIMIT:
                break
            continue
        ignored_streak = 0
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
    outcome = "won" if game.won else "lost" if game.lost else "stuck" if ignored_streak else "limit"
    return GameResult(
        seed, outcome, game.shots, game.final_score, max_row, game.pushes, pops, wipes, checked, landing_bad, pop_bad,
        ignored_clicks=ignored, **_model_stats(decider),
    )


def run_job(job):
    overrides, seed, max_shots, decider_settings, jitter, click_area = job
    weights = weights_from_overrides(overrides)
    decider = None
    if decider_settings is not None and decider_settings.decider == "model":
        decider = ModelDecider(weights, decider_settings, log=DecisionLog(decider_settings.decision_log_dir, f"sim_seed{seed}"))
    return play(weights, seed, max_shots, decider, jitter, click_area)


def summarize(results):
    n = len(results)
    shots = [r.shots for r in results]
    return {
        "games": n,
        "won": sum(r.outcome == "won" for r in results),
        "lost": sum(r.outcome == "lost" for r in results),
        "stuck": sum(r.outcome == "stuck" for r in results),
        "limit": sum(r.outcome == "limit" for r in results),
        "loss_per_1000": round(1000 * sum(r.outcome in ("lost", "stuck") for r in results) / max(1, sum(shots)), 3),
        "ignored_clicks": sum(r.ignored_clicks for r in results),
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
