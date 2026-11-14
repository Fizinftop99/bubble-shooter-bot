import argparse
import csv
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

from ..config import StrategyWeights, load_settings
from ..decider import ModelDecider
from .runner import DEFAULT_MAX_SHOTS, run_job, summarize

RESULTS_DIR = Path("sim_results")


def parse_overrides(text):
    pairs = [part for part in text.split(",") if part.strip()]
    return dict(part.strip().split("=", 1) for part in pairs)


def run_variant(overrides, seeds, max_shots, workers, decider_settings=None):
    jobs = [(overrides, seed, max_shots, decider_settings) for seed in seeds]
    if workers <= 1:
        return [run_job(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run_job, jobs, chunksize=1))


def save_csv(name, results):
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(results[0])))
        writer.writeheader()
        writer.writerows(asdict(r) for r in results)
    return path


def print_summary(label, results):
    print(f"[{label}]")
    for key, value in summarize(results).items():
        print(f"  {key:18} {value}")


def print_paired(label_a, a, label_b, b):
    by_seed = {r.seed: r for r in b}
    diffs = [by_seed[r.seed].shots - r.shots for r in a if r.seed in by_seed]
    better = sum(d > 0 for d in diffs)
    worse = sum(d < 0 for d in diffs)
    print(f"paired by seed: {label_b} lasted longer in {better} games, shorter in {worse}, equal in {len(diffs) - better - worse}")


def main():
    parser = argparse.ArgumentParser(description="Run the bot's strategy against a headless copy of the game")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--workers", type=int, default=None, help="parallel games (default 8 for the rule, 1 for the model)")
    parser.add_argument("--decider", choices=("rule", "model"), default="rule", help="who picks each shot")
    parser.add_argument("--model-name", help="override MODEL_NAME for --decider model")
    parser.add_argument("--max-shots", type=int, default=DEFAULT_MAX_SHOTS)
    parser.add_argument("--set", default="", help="comma separated StrategyWeights overrides, e.g. FRONT_WEIGHT=0,DROP_WEIGHT=5")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"), help="two override sets run on the same seeds")
    args = parser.parse_args()
    seeds = range(args.seed_start, args.seed_start + args.games)
    variants = [("A", args.compare[0]), ("B", args.compare[1])] if args.compare else [("run", args.set)]
    decider_settings = None
    if args.decider == "model":
        decider_settings = replace(load_settings().decider, decider="model")
        if args.model_name:
            decider_settings = replace(decider_settings, model_name=args.model_name)
        problem = ModelDecider(StrategyWeights(), decider_settings).warm_up()
        if problem:
            print(f"[warn] model not reachable ({problem}); every shot will fall back to the rule")
    workers = args.workers or (1 if decider_settings else 8)
    all_results = {}
    for label, text in variants:
        started = time.time()
        results = run_variant(parse_overrides(text), seeds, args.max_shots, workers, decider_settings)
        all_results[label] = results
        print_summary(f"{label}: {text or 'defaults'}{' [model decider]' if decider_settings else ''}", results)
        print(f"  csv: {save_csv(label, results)}  ({time.time() - started:.0f}s)")
    if args.compare:
        print_paired("A", all_results["A"], "B", all_results["B"])


if __name__ == "__main__":
    main()
