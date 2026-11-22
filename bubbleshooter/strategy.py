import math
import os
from dataclasses import dataclass

from . import layout
from .config import StrategyWeights
from .vision import colors_match

CONTACT_DIST = 29
MAX_BOUNCES = 2
MAX_SNAP_DIST = 20
MIN_POP = 3
# The game clamps the launch to 15 degrees above the horizontal (Cannon.cannonFired), and below about 10 degrees
# the click at CLICK_DISTANCE falls outside the clickable area and fires nothing; so aim only where the game shoots.
ANGLE_MIN_DEG = 15
ANGLE_MAX_DEG = 165
ANGLE_STEP_DEG = 0.5
CLICK_DISTANCE = 250
WIDTH_BONUS = 0.05
WIDTH_BONUS_CAP = 6


def _neighbors(pos, others):
    return [o for o in others if o is not pos and math.hypot(o["x"] - pos["x"], o["y"] - pos["y"]) <= layout.NEIGHBOR_DIST]


def _lattice(bubbles):
    top_shift = next(b["shifted"] ^ (b["row"] % 2 == 1) for b in bubbles)
    last_row = int((layout.FIELD_BOTTOM_Y - layout.GRID_ROW0_Y) // layout.ROW_PITCH)
    cells = []
    for row in range(last_row + 1):
        shifted = top_shift ^ (row % 2 == 1)
        x0 = layout.COL0_X + (layout.ODD_ROW_SHIFT if shifted else 0)
        for col in range(layout.COLS):
            cells.append((x0 + col * layout.COL_PITCH, layout.GRID_ROW0_Y + row * layout.ROW_PITCH, row))
    return cells


def _trace(bubbles, origin, angle):
    px, py = origin
    dx, dy = math.cos(angle), -math.sin(angle)
    for _ in range(MAX_BOUNCES + 1):
        t_wall = math.inf
        if dx > 1e-9:
            t_wall = (layout.WALL_MAX_X - px) / dx
        elif dx < -1e-9:
            t_wall = (layout.WALL_MIN_X - px) / dx
        t_top = (layout.GRID_ROW0_Y - py) / dy if dy < -1e-9 else math.inf
        t_hit, hit = math.inf, False
        for b in bubbles:
            rx, ry = b["x"] - px, b["y"] - py
            t = rx * dx + ry * dy
            if t <= 0:
                continue
            h2 = rx * rx + ry * ry - t * t
            if h2 < CONTACT_DIST ** 2:
                th = t - math.sqrt(CONTACT_DIST ** 2 - h2)
                if 0 < th < t_hit:
                    t_hit = th
        t_end = min(t_hit, t_top, t_wall)
        if t_end == math.inf:
            return None
        if t_end == t_wall and t_wall < min(t_hit, t_top):
            px, py = px + dx * t_end, py + dy * t_end
            dx = -dx
            continue
        return px + dx * t_end, py + dy * t_end
    return None


def _landing_cell(point, occupied, cells):
    best = None
    for cx, cy, row in cells:
        if (cx, cy) in occupied:
            continue
        d = math.hypot(cx - point[0], cy - point[1])
        if best is None or d < best[0]:
            best = (d, cx, cy, row)
    return best if best and best[0] <= MAX_SNAP_DIST else None


def _colour_labels(bubbles):
    reps, labels = [], {}
    for b in bubbles:
        for i, rep in enumerate(reps):
            if colors_match(b["color"], rep):
                labels[id(b)] = i
                break
        else:
            reps.append(b["color"])
            labels[id(b)] = len(reps) - 1
    return labels


def _wiped_colours(labels, removed_ids, remaining_total, weights):
    if remaining_total <= weights.wipe_free_bubbles:
        return 0
    before = set(labels.values())
    after = {label for bid, label in labels.items() if bid not in removed_ids}
    return len(before - after)


def _evaluate(cell, bubbles, shooter_color, max_row, weights, labels):
    cx, cy, row = cell
    placed = {"x": cx, "y": cy, "color": shooter_color, "row": row}
    everything = bubbles + [placed]
    cluster, frontier = {id(placed)}, [placed]
    while frontier:
        cur = frontier.pop()
        for n in _neighbors(cur, everything):
            if id(n) not in cluster and colors_match(n["color"], shooter_color):
                cluster.add(id(n))
                frontier.append(n)
    if len(cluster) >= MIN_POP:
        rest = [b for b in bubbles if id(b) not in cluster]
        anchored = {id(b) for b in rest if b["row"] == 0}
        frontier = [b for b in rest if id(b) in anchored]
        while frontier:
            cur = frontier.pop()
            for n in _neighbors(cur, rest):
                if id(n) not in anchored:
                    anchored.add(id(n))
                    frontier.append(n)
        dropped = len(rest) - len(anchored)
        score = len(cluster) + weights.drop_weight * dropped + weights.depth_weight * row
        removed_ids = cluster | {id(b) for b in rest if id(b) not in anchored}
        wiped = _wiped_colours(labels, removed_ids, len(anchored), weights)
        score -= weights.color_wipe_penalty * wiped
        if max_row >= weights.pressure_row:
            new_max_row = max((b["row"] for b in rest if id(b) in anchored), default=0)
            score += weights.front_weight * (max_row - new_max_row)
        return score, f"pop {len(cluster)} drop {dropped}", wiped
    friends = len(cluster) - 1
    growth = max(0, row - max_row)
    penalty = weights.growth_penalty + (weights.front_weight if max_row >= weights.pressure_row else 0)
    score = weights.miss_base + weights.friend_weight * friends - penalty * growth
    return score, f"no pop (joins {friends} same-colour)", 0


@dataclass(frozen=True)
class Candidate:
    cell: tuple
    degs: tuple
    score: float
    why: str
    wiped: int

    def aim(self, origin=layout.SHOOTER_POS):
        mid = math.radians(self.degs[len(self.degs) // 2])
        x = origin[0] + CLICK_DISTANCE * math.cos(mid)
        y = origin[1] - CLICK_DISTANCE * math.sin(mid)
        return round(x), round(y), f"{self.why} at cell {self.cell[:2]} angle {math.degrees(mid):.1f}"


def list_candidates(shooter_color, bubbles, origin=layout.SHOOTER_POS, weights=StrategyWeights()):
    """Every reachable cell with its score. Uses strategy_fast (numba, identical candidates) when available."""
    if not bubbles or shooter_color is None:
        return []
    fast = _fast_module()
    if fast is not None:
        result = fast.list_candidates(shooter_color, bubbles, origin, weights)
        if result is not None:
            return result
    return _list_candidates_pure(shooter_color, bubbles, origin, weights)


_FAST = []


def _fast_module():
    """strategy_fast, or None without numba or with BUBBLE_SIM_PURE=1. Imported lazily: it imports this module."""
    if not _FAST:
        module = None
        if os.environ.get("BUBBLE_SIM_PURE") != "1":
            try:
                from . import strategy_fast as module
            except ImportError:
                module = None
        _FAST.append(module)
    return _FAST[0]


def _list_candidates_pure(shooter_color, bubbles, origin=layout.SHOOTER_POS, weights=StrategyWeights()):
    cells = _lattice(bubbles)
    occupied = {(b["x"], b["y"]) for b in bubbles}
    max_row = max(b["row"] for b in bubbles)
    labels = _colour_labels(bubbles)
    by_cell = {}
    steps = int((ANGLE_MAX_DEG - ANGLE_MIN_DEG) / ANGLE_STEP_DEG) + 1
    for i in range(steps):
        deg = ANGLE_MIN_DEG + i * ANGLE_STEP_DEG
        point = _trace(bubbles, origin, math.radians(deg))
        if point is None:
            continue
        cell = _landing_cell(point, occupied, cells)
        if cell:
            by_cell.setdefault(cell[1:], []).append(deg)
    candidates = []
    for cell, degs in by_cell.items():
        score, why, wiped = _evaluate(cell, bubbles, shooter_color, max_row, weights, labels)
        score += min(len(degs), WIDTH_BONUS_CAP) * WIDTH_BONUS
        candidates.append(Candidate(cell, tuple(degs), score, why, wiped))
    return candidates


def best_candidate(candidates):
    best = None
    for candidate in candidates:
        if best is None or candidate.score > best.score:
            best = candidate
    return best


def choose_target(shooter_color, bubbles, origin=layout.SHOOTER_POS, weights=StrategyWeights()):
    if not bubbles:
        return None
    best = best_candidate(list_candidates(shooter_color, bubbles, origin, weights))
    if best:
        return best.aim(origin)
    top_row = min(b["row"] for b in bubbles)
    b = min((b for b in bubbles if b["row"] == top_row), key=lambda b: abs(b["x"] - origin[0]))
    return b["x"], b["y"], "fallback: no simulated shot, aiming at top row"


def find_best_target(shooter_color, bubbles):
    result = choose_target(shooter_color, bubbles)
    return None if result is None else result[:2]
