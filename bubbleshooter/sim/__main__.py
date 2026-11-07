import argparse
import csv
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from ..config import StrategyWeights
from .runner import DEFAULT_MAX_SHOTS, run_job, summarize

RESULTS_DIR = Path("sim_results")


def parse_overrides(text):
    pairs = [part for part in text.split(",") if part.strip()]
    return dict(part.strip().split("=", 1) for part in pairs)


def run_variant(overrides, seeds, max_shots, workers, label="run"):
    jobs = [(overrides, seed, max_shots) for seed in seeds]
    if workers <= 1:
        return [run_job(job) for job in tqdm(jobs, desc=label, unit="game")]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(tqdm(pool.map(run_job, jobs, chunksize=1), total=len(jobs), desc=label, unit="game"))


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
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-shots", type=int, default=DEFAULT_MAX_SHOTS)
    parser.add_argument("--set", default="", help="comma separated StrategyWeights overrides, e.g. FRONT_WEIGHT=0,DROP_WEIGHT=5")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"), help="two override sets run on the same seeds")
    args = parser.parse_args()
    seeds = range(args.seed_start, args.seed_start + args.games)
    variants = [("A", args.compare[0]), ("B", args.compare[1])] if args.compare else [("run", args.set)]
    all_results = {}
    for label, text in variants:
        started = time.time()
        results = run_variant(parse_overrides(text), seeds, args.max_shots, args.workers, f"{label} {text}".strip())
        all_results[label] = results
        print_summary(f"{label}: {text or 'defaults'}", results)
        print(f"  csv: {save_csv(label, results)}  ({time.time() - started:.0f}s)")
    if args.compare:
        print_paired("A", all_results["A"], "B", all_results["B"])


if __name__ == "__main__":
    main()
