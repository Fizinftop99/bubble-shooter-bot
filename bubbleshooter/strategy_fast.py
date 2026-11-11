import math
from functools import lru_cache

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
    _lattice,
)
from .vision import colors_match

RAY_EPS = 1e-9
NO_CELL = -1
MAX_NEIGHBORS = 8
DEGS = [ANGLE_MIN_DEG + i * ANGLE_STEP_DEG for i in range(int((ANGLE_MAX_DEG - ANGLE_MIN_DEG) / ANGLE_STEP_DEG) + 1)]
ANGLES = np.array([math.radians(d) for d in DEGS], dtype=np.float64)


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
            # a cell further than MAX_SNAP_DIST on either axis can never be accepted below
            if not free[c] or abs(cell_y[c] - qy) > MAX_SNAP_DIST or abs(cell_x[c] - qx) > MAX_SNAP_DIST:
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
    nbr = np.empty((n, MAX_NEIGHBORS), dtype=np.int64)
    cnt = np.zeros(n, dtype=np.int64)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(ys[j] - ys[i]) > layout.NEIGHBOR_DIST or abs(xs[j] - xs[i]) > layout.NEIGHBOR_DIST:
                continue
            if math.hypot(xs[j] - xs[i], ys[j] - ys[i]) <= layout.NEIGHBOR_DIST and cnt[i] < MAX_NEIGHBORS and cnt[j] < MAX_NEIGHBORS:
                nbr[i, cnt[i]] = j
                cnt[i] += 1
                nbr[j, cnt[j]] = i
                cnt[j] += 1
    return nbr, cnt


@njit(cache=True)
def _evaluate(cx, cy, row, xs, ys, rows, same, labels, nbr, cnt, max_row,
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
        for m in range(cnt[cur]):
            j = nbr[cur, m]
            if same[j] and not in_cluster[j]:
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
            for m in range(cnt[cur]):
                j = nbr[cur, m]
                if not in_cluster[j] and not anchored[j]:
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


@njit(cache=True)
def _choose(xs, ys, rows, same, labels, cell_x, cell_y, cell_row, angles, ox, oy,
            drop_weight, depth_weight, miss_base, friend_weight, growth_penalty,
            pressure_row, front_weight, wipe_penalty, wipe_free):
    n = xs.shape[0]
    n_cells = cell_x.shape[0]
    free = np.ones(n_cells, dtype=np.bool_)
    max_row = 0
    for i in range(n):
        if rows[i] > max_row:
            max_row = rows[i]
        for c in range(n_cells):
            if cell_x[c] == xs[i] and cell_y[c] == ys[i]:
                free[c] = False
                break
    landing = _landing_cells(xs, ys, cell_x, cell_y, free, ox, oy, angles)
    counts = np.zeros(n_cells, dtype=np.int64)
    order = np.empty(n_cells, dtype=np.int64)
    n_order = 0
    for k in range(landing.shape[0]):
        c = landing[k]
        if c != NO_CELL:
            if counts[c] == 0:
                order[n_order] = c
                n_order += 1
            counts[c] += 1
    nbr, cnt = _adjacency(xs, ys)
    best_c, best_score, best_size, best_dropped = NO_CELL, 0.0, 0, 0
    for o in range(n_order):
        c = order[o]
        score, size, dropped = _evaluate(
            cell_x[c], cell_y[c], cell_row[c], xs, ys, rows, same, labels, nbr, cnt, max_row,
            drop_weight, depth_weight, miss_base, friend_weight, growth_penalty,
            pressure_row, front_weight, wipe_penalty, wipe_free,
        )
        score += min(counts[c], WIDTH_BONUS_CAP) * WIDTH_BONUS
        if best_c == NO_CELL or score > best_score:
            best_c, best_score, best_size, best_dropped = c, score, size, dropped
    mid_k = NO_CELL
    if best_c != NO_CELL:
        target, seen = counts[best_c] // 2, 0
        for k in range(landing.shape[0]):
            if landing[k] == best_c:
                if seen == target:
                    mid_k = k
                    break
                seen += 1
    return best_c, mid_k, best_size, best_dropped


@lru_cache(maxsize=None)
def _lattice_arrays(top_shift):
    cells = _lattice([{"shifted": top_shift, "row": 0}])
    return (
        np.array([c[0] for c in cells], dtype=np.float64),
        np.array([c[1] for c in cells], dtype=np.float64),
        np.array([c[2] for c in cells], dtype=np.int64),
    )


@lru_cache(maxsize=65536)
def _match(a, b):
    return colors_match(a, b)


def _bubble_arrays(shooter_color, bubbles):
    xs, ys, rows, same, labels = [], [], [], [], []
    reps, label_of, same_of = [], {}, {}
    for b in bubbles:
        color = b["color"]
        label = label_of.get(color)
        if label is None:
            for i, rep in enumerate(reps):
                if _match(color, rep):
                    label = i
                    break
            else:
                reps.append(color)
                label = len(reps) - 1
            label_of[color] = label
            same_of[color] = _match(color, shooter_color)
        xs.append(b["x"])
        ys.append(b["y"])
        rows.append(b["row"])
        same.append(same_of[color])
        labels.append(label)
    return (
        np.array(xs, dtype=np.float64),
        np.array(ys, dtype=np.float64),
        np.array(rows, dtype=np.int64),
        np.array(same, dtype=np.bool_),
        np.array(labels, dtype=np.int64),
    )


def choose_target(shooter_color, bubbles, origin=layout.SHOOTER_POS, weights=StrategyWeights()):
    if not bubbles:
        return None
    if shooter_color is not None:
        first = bubbles[0]
        cell_x, cell_y, cell_row = _lattice_arrays(bool(first["shifted"] ^ (first["row"] % 2 == 1)))
        xs, ys, rows, same, labels = _bubble_arrays(shooter_color, bubbles)
        c, mid_k, size, dropped = _choose(
            xs, ys, rows, same, labels, cell_x, cell_y, cell_row, ANGLES, float(origin[0]), float(origin[1]),
            weights.drop_weight, weights.depth_weight, weights.miss_base, weights.friend_weight,
            weights.growth_penalty, weights.pressure_row, weights.front_weight,
            weights.color_wipe_penalty, weights.wipe_free_bubbles,
        )
        if c != NO_CELL:
            why = f"pop {size} drop {dropped}" if size >= MIN_POP else f"no pop (joins {size - 1} same-colour)"
            mid = math.radians(DEGS[mid_k])
            x = origin[0] + CLICK_DISTANCE * math.cos(mid)
            y = origin[1] - CLICK_DISTANCE * math.sin(mid)
            cell = (int(cell_x[c]), int(cell_y[c]))
            return round(x), round(y), f"{why} at cell {cell} angle {math.degrees(mid):.1f}"
    top_row = min(b["row"] for b in bubbles)
    b = min((b for b in bubbles if b["row"] == top_row), key=lambda b: abs(b["x"] - origin[0]))
    return b["x"], b["y"], "fallback: no simulated shot, aiming at top row"
