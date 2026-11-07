import math

from . import layout
from .vision import colors_match

CONTACT_DIST = 29
MAX_BOUNCES = 2
MAX_SNAP_DIST = 20
MIN_POP = 3
ANGLE_MIN_DEG = 8
ANGLE_MAX_DEG = 172
ANGLE_STEP_DEG = 0.5
CLICK_DISTANCE = 250
DROP_WEIGHT = 2.0
DEPTH_WEIGHT = 0.15
MISS_BASE = -3.0
FRIEND_WEIGHT = 0.4
GROWTH_PENALTY = 1.5
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


def _evaluate(cell, bubbles, shooter_color, max_row):
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
        score = len(cluster) + DROP_WEIGHT * dropped + DEPTH_WEIGHT * row
        return score, f"pop {len(cluster)} drop {dropped}"
    friends = len(cluster) - 1
    score = MISS_BASE + FRIEND_WEIGHT * friends - GROWTH_PENALTY * max(0, row - max_row)
    return score, f"no pop (joins {friends} same-colour)"


def choose_target(shooter_color, bubbles, origin=layout.SHOOTER_POS):
    if not bubbles:
        return None
    if shooter_color is not None:
        cells = _lattice(bubbles)
        occupied = {(b["x"], b["y"]) for b in bubbles}
        max_row = max(b["row"] for b in bubbles)
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
        best = None
        for cell, degs in by_cell.items():
            score, why = _evaluate(cell, bubbles, shooter_color, max_row)
            score += min(len(degs), WIDTH_BONUS_CAP) * WIDTH_BONUS
            if best is None or score > best[0]:
                best = (score, cell, degs, why)
        if best:
            _, cell, degs, why = best
            mid = math.radians(degs[len(degs) // 2])
            x = origin[0] + CLICK_DISTANCE * math.cos(mid)
            y = origin[1] - CLICK_DISTANCE * math.sin(mid)
            return round(x), round(y), f"{why} at cell {cell[:2]} angle {math.degrees(mid):.1f}"
    top_row = min(b["row"] for b in bubbles)
    b = min((b for b in bubbles if b["row"] == top_row), key=lambda b: abs(b["x"] - origin[0]))
    return b["x"], b["y"], "fallback: no simulated shot, aiming at top row"


def find_best_target(shooter_color, bubbles):
    result = choose_target(shooter_color, bubbles)
    return None if result is None else result[:2]
