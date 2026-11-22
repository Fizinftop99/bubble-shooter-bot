"""Check where each live shot actually landed against what the decider predicted.

After a shot the next frame is compared with the frame the shot was chosen on. A bubble that sticks shows
up as exactly one new grid position; a pop shows up as positions that disappeared. The decision's reason
("pop 5 drop 1 at cell (204, 307) angle 109.5" or "no pop (...) at cell (x, y) ...") gives the prediction.

    python -m bubbleshooter.landing logs/landing_*.jsonl   # summary of one or more live runs
"""
import argparse
import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from . import layout

CELL_PATTERN = re.compile(r"at cell \((\d+), (\d+)\)")
POP_PATTERN = re.compile(r"^pop (\d+) drop (\d+)")
PUSH_MIN_ADDED = layout.COLS // 2  # a push brings a whole new top row and shifts every bubble down
MAX_SAVED_FRAMES = 60  # frame pairs kept per run for shots that did not go as predicted

OUTCOMES = (
    "as_predicted",   # stuck at the predicted cell, or popped exactly the predicted number of bubbles
    "other_cell",     # no pop predicted, stuck somewhere else
    "missed_pop",     # pop predicted, the bubble just stuck (see actual_cell)
    "unexpected_pop", # no pop predicted, bubbles disappeared
    "pop_count_off",  # popped, but a different number of bubbles than predicted
    "no_change",      # nothing appeared or disappeared: the click did not fire or the frame is stale
    "push",           # new rows came down; the landing cannot be told apart
    "unclear",        # anything else (usually a vision misread)
)
SURPRISES = {"other_cell", "missed_pop", "unexpected_pop", "pop_count_off", "no_change", "unclear"}


@dataclass(frozen=True)
class Prediction:
    cell: tuple | None
    pops: bool
    removed: int  # bubbles already on the board that should disappear (cluster without the shot bubble + drops)

    @classmethod
    def from_reason(cls, reason):
        cell = CELL_PATTERN.search(reason)
        pop = POP_PATTERN.match(reason)
        return cls(
            cell=(int(cell[1]), int(cell[2])) if cell else None,
            pops=pop is not None,
            removed=int(pop[1]) - 1 + int(pop[2]) if pop else 0,
        )


def _positions(bubbles):
    return {(b["x"], b["y"]): b["color"] for b in bubbles}


def check(before, after, reason):
    """Compare two frames' bubbles (vision.find_grid_bubbles) with the shot's reason. Returns a dict."""
    prediction = Prediction.from_reason(reason)
    was, now = _positions(before), _positions(after)
    added = sorted(set(now) - set(was))
    removed = sorted(set(was) - set(now))
    actual = added[0] if len(added) == 1 else None
    if len(added) >= PUSH_MIN_ADDED:
        outcome = "push"
    elif not added and not removed:
        outcome = "no_change"
    elif actual and not removed:
        if prediction.pops:
            outcome = "missed_pop"
        elif prediction.cell is None or actual == prediction.cell:
            outcome = "as_predicted" if prediction.cell else "unclear"
        else:
            outcome = "other_cell"
    elif removed and not added:
        if not prediction.pops:
            outcome = "unexpected_pop"
        elif len(removed) == prediction.removed:
            outcome = "as_predicted"
        else:
            outcome = "pop_count_off"
    else:
        outcome = "unclear"
    record = {
        "outcome": outcome,
        "predicted_cell": prediction.cell,
        "predicted_pop": prediction.pops,
        "predicted_removed": prediction.removed,
        "actual_cell": actual,
        "added": len(added),
        "removed": len(removed),
    }
    if actual and prediction.cell:
        record["miss_px"] = round(math.dist(actual, prediction.cell), 1)
    if 0 < len(added) < PUSH_MIN_ADDED:
        record["added_cells"] = added
    return record


class LandingLog:
    """JSONL record per shot plus before/after frames for the first MAX_SAVED_FRAMES surprises."""

    def __init__(self, directory):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        folder = Path(directory)
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / f"landing_{stamp}.jsonl"
        self.frames_dir = folder / f"landing_{stamp}"
        self.counts = Counter()
        self._saved = 0

    def write(self, shot, record, before_img=None, after_img=None, **extra):
        self.counts[record["outcome"]] += 1
        row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "shot": shot, **extra, **record}
        if record["outcome"] in SURPRISES and before_img is not None and after_img is not None and self._saved < MAX_SAVED_FRAMES:
            self.frames_dir.mkdir(exist_ok=True)
            before_img.save(self.frames_dir / f"shot{shot:04d}_before.png")
            after_img.save(self.frames_dir / f"shot{shot:04d}_after.png")
            row["frames"] = str(self.frames_dir / f"shot{shot:04d}_*.png")
            self._saved += 1
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

    def summary(self):
        return summarize(self.counts)


def summarize(counts):
    total = sum(counts.values())
    if not total:
        return "landing: no shots checked"
    parts = [f"{name} {counts[name]} ({counts[name] / total:.1%})" for name in OUTCOMES if counts[name]]
    judged = total - counts["push"]
    hit = f"{counts['as_predicted'] / judged:.1%} as predicted of {judged} judged" if judged else "nothing judged"
    return f"landing: {total} shots, {hit}; " + ", ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Summarise landing logs written by the live bot")
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()
    counts, by_pred, miss_px = Counter(), Counter(), []
    for path in args.files:
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            counts[row["outcome"]] += 1
            by_pred[("pop" if row["predicted_pop"] else "no pop", row["outcome"])] += 1
            if row["outcome"] in ("other_cell", "missed_pop") and "miss_px" in row:
                miss_px.append(row["miss_px"])
    print(summarize(counts))
    for kind in ("pop", "no pop"):
        n = sum(v for (k, _), v in by_pred.items() if k == kind)
        if n:
            print(f"  predicted {kind}: {n} -> " + ", ".join(f"{o} {by_pred[(kind, o)]}" for o in OUTCOMES if by_pred[(kind, o)]))
    if miss_px:
        miss_px.sort()
        print(f"  stuck elsewhere: {len(miss_px)} shots, distance to the predicted cell median {miss_px[len(miss_px) // 2]} px, "
              f"neighbour (<= {layout.NEIGHBOR_DIST} px) {sum(d <= layout.NEIGHBOR_DIST for d in miss_px)}")


if __name__ == "__main__":
    main()
