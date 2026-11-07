import math

import numpy as np
from numba import njit

from . import layout
from .config import StrategyWeights
from .strategy import (
    ANGLE_MAX_DEG,
    ANGLE_MIN_DEG,
    ANGLE_STEP_DEG,
    CLICK_DISTANCE,
    CONTACT_DIST,
    MAX_BOUNCES,
    MAX_SNAP_DIST,
    MIN_POP,
    WIDTH_BONUS,
    WIDTH_BONUS_CAP,
    _colour_labels,
    _lattice,
)
from .vision import colors_match

RAY_EPS = 1e-9
NO_CELL = -1


@njit(cache=True)
def _trace(xs, ys, ox, oy, angle):
    px, py = ox, oy
    dx, dy = math.cos(angle), -math.sin(angle)
    contact2 = CONTACT_DIST ** 2
    for _ in range(MAX_BOUNCES + 1):
        t_wall = math.inf
        if dx > RAY_EPS:
            t_wall = (layout.WALL_MAX_X - px) / dx
        elif dx < -RAY_EPS:
            t_wall = (layout.WALL_MIN_X - px) / dx
        t_top = (layout.GRID_ROW0_Y - py) / dy if dy < -RAY_EPS else math.inf
        t_hit = math.inf
        for i in range(xs.shape[0]):
            rx, ry = xs[i] - px, ys[i] - py
            t = rx * dx + ry * dy
            if t <= 0:
                continue
            h2 = rx * rx + ry * ry - t * t
            if h2 < contact2:
                th = t - math.sqrt(contact2 - h2)
                if 0 < th < t_hit:
                    t_hit = th
        t_end = min(t_hit, t_top, t_wall)
        if t_end == math.inf:
            return False, 0.0, 0.0
        if t_end == t_wall and t_wall < min(t_hit, t_top):
            px, py = px + dx * t_end, py + dy * t_end
            dx = -dx
            continue
        return True, px + dx * t_end, py + dy * t_end
    return False, 0.0, 0.0


@njit(cache=True)
def _landing_cells(xs, ys, cell_x, cell_y, free, ox, oy, angles):
    out = np.full(angles.shape[0], NO_CELL, dtype=np.int64)
    for k in range(angles.shape[0]):
        ok, qx, qy = _trace(xs, ys, ox, oy, angles[k])
        if not ok:
            continue
        best_d, best_i = math.inf, NO_CELL
        for c in range(cell_x.shape[0]):
            if not free[c]:
                continue
            d = math.hypot(cell_x[c] - qx, cell_y[c] - qy)
            if best_i == NO_CELL or d < best_d:
                best_d, best_i = d, c
        if best_i != NO_CELL and best_d <= MAX_SNAP_DIST:
            out[k] = best_i
    return out


@njit(cache=True)
def _adjacency(xs, ys):
    n = xs.shape[0]
    adj = np.zeros((n, n), dtype=np.bool_)
    for i in range(n):
        for j in range(n):
            if i != j and math.hypot(xs[j] - xs[i], ys[j] - ys[i]) <= layout.NEIGHBOR_DIST:
                adj[i, j] = True
    return adj


@njit(cache=True)
def _evaluate(cx, cy, row, xs, ys, rows, same, labels, adj, max_row,
              drop_weight, depth_weight, miss_base, friend_weight, growth_penalty,
              pressure_row, front_weight, wipe_penalty, wipe_free):
    n = xs.shape[0]
    in_cluster = np.zeros(n, dtype=np.bool_)
    stack = np.empty(n, dtype=np.int64)
    top = 0
    size = 1
    for i in range(n):
        if same[i] and math.hypot(xs[i] - cx, ys[i] - cy) <= layout.NEIGHBOR_DIST:
            in_cluster[i] = True
            stack[top] = i
            top += 1
            size += 1
    while top > 0:
        top -= 1
        cur = stack[top]
        for j in range(n):
            if adj[cur, j] and same[j] and not in_cluster[j]:
                in_cluster[j] = True
                stack[top] = j
                top += 1
                size += 1
    pressure = max_row >= pressure_row
    if size >= MIN_POP:
        anchored = np.zeros(n, dtype=np.bool_)
        top = 0
        for i in range(n):
            if not in_cluster[i] and rows[i] == 0:
                anchored[i] = True
                stack[top] = i
                top += 1
        while top > 0:
            top -= 1
            cur = stack[top]
            for j in range(n):
                if adj[cur, j] and not in_cluster[j] and not anchored[j]:
                    anchored[j] = True
                    stack[top] = j
                    top += 1
        rest = 0
        kept = 0
        new_max_row = 0
        for i in range(n):
            if not in_cluster[i]:
                rest += 1
            if anchored[i]:
                kept += 1
                if rows[i] > new_max_row:
                    new_max_row = rows[i]
        dropped = rest - kept
        score = size + drop_weight * dropped + depth_weight * row
        if kept > wipe_free:
            seen_before = np.zeros(n + 1, dtype=np.bool_)
            seen_after = np.zeros(n + 1, dtype=np.bool_)
            for i in range(n):
                seen_before[labels[i]] = True
                if anchored[i]:
                    seen_after[labels[i]] = True
            wiped = 0
            for k in range(n + 1):
                if seen_before[k] and not seen_after[k]:
                    wiped += 1
            score -= wipe_penalty * wiped
        if pressure:
            score += front_weight * (max_row - new_max_row)
        return score, size, dropped
    friends = size - 1
    growth = max(0, row - max_row)
    penalty = growth_penalty + (front_weight if pressure else 0.0)
    return miss_base + friend_weight * friends - penalty * growth, size, 0


def choose_target(shooter_color, bubbles, origin=layout.SHOOTER_POS, weights=StrategyWeights()):
    if not bubbles:
        return None
    if shooter_color is not None:
        cells = _lattice(bubbles)
        cell_x = np.array([c[0] for c in cells], dtype=np.float64)
        cell_y = np.array([c[1] for c in cells], dtype=np.float64)
        cell_row = [c[2] for c in cells]
        occupied = {(b["x"], b["y"]) for b in bubbles}
        free = np.array([(c[0], c[1]) not in occupied for c in cells], dtype=np.bool_)
        xs = np.array([b["x"] for b in bubbles], dtype=np.float64)
        ys = np.array([b["y"] for b in bubbles], dtype=np.float64)
        rows = np.array([b["row"] for b in bubbles], dtype=np.int64)
        same = np.array([colors_match(b["color"], shooter_color) for b in bubbles], dtype=np.bool_)
        label_map = _colour_labels(bubbles)
        labels = np.array([label_map[id(b)] for b in bubbles], dtype=np.int64)
        max_row = int(rows.max())
        adj = _adjacency(xs, ys)
        steps = int((ANGLE_MAX_DEG - ANGLE_MIN_DEG) / ANGLE_STEP_DEG) + 1
        degs = [ANGLE_MIN_DEG + i * ANGLE_STEP_DEG for i in range(steps)]
        angles = np.array([math.radians(d) for d in degs], dtype=np.float64)
        landing = _landing_cells(xs, ys, cell_x, cell_y, free, float(origin[0]), float(origin[1]), angles)
        by_cell = {}
        for deg, c in zip(degs, landing):
            if c != NO_CELL:
                by_cell.setdefault(int(c), []).append(deg)
        best = None
        for c, cell_degs in by_cell.items():
            score, size, dropped = _evaluate(
                cell_x[c], cell_y[c], cell_row[c], xs, ys, rows, same, labels, adj, max_row,
                weights.drop_weight, weights.depth_weight, weights.miss_base, weights.friend_weight,
                weights.growth_penalty, weights.pressure_row, weights.front_weight,
                weights.color_wipe_penalty, weights.wipe_free_bubbles,
            )
            score += min(len(cell_degs), WIDTH_BONUS_CAP) * WIDTH_BONUS
            if best is None or score > best[0]:
                why = f"pop {size} drop {dropped}" if size >= MIN_POP else f"no pop (joins {size - 1} same-colour)"
                best = (score, c, cell_degs, why)
        if best:
            _, c, cell_degs, why = best
            mid = math.radians(cell_degs[len(cell_degs) // 2])
            x = origin[0] + CLICK_DISTANCE * math.cos(mid)
            y = origin[1] - CLICK_DISTANCE * math.sin(mid)
            cell = (int(cell_x[c]), int(cell_y[c]))
            return round(x), round(y), f"{why} at cell {cell} angle {math.degrees(mid):.1f}"
    top_row = min(b["row"] for b in bubbles)
    b = min((b for b in bubbles if b["row"] == top_row), key=lambda b: abs(b["x"] - origin[0]))
    return b["x"], b["y"], "fallback: no simulated shot, aiming at top row"
