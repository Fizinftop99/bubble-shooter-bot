"""Numba kernels for the deterministic part of SimGame.shoot: flight, snap, cluster pop and hanging drop.

Randomness stays in engine.py on random.Random, so seeded games are unchanged. The board is passed in as
arrays built from the dict (in dict order) and the kernels return which cells to add or delete; the dict
itself is only ever edited by engine.py, so its insertion order is the same as on the pure Python path.

Collision test. The pure path compares math.hypot(dx, dy) < COLLISION_DIST. CPython's hypot is its own
algorithm (vector_norm in Modules/mathmodule.c) and is not the same function as libm's hypot or
sqrt(dx*dx + dy*dy): near the boundary the last bit can differ and flip the comparison. So the kernel
first rejects or accepts with the squared distance when it is clearly away from the boundary, and only
inside a thin band calls _cpython_hypot, a line-by-line port of vector_norm for two coordinates.
tests/test_engine_fast.py checks the port bit for bit against math.hypot, including at the boundary.
"""
import itertools
import math

import numpy as np
from numba import njit

from .engine import (
    ARRIVAL_OFFSET,
    BOARD_WIDTH,
    BUBBLE_SIZE,
    CEILING_SNAP_Y,
    CEILING_Y,
    COLLISION_DIST,
    MAX_FLIGHT_FRAMES,
    MIN_POP,
    ORIGIN_X,
    ORIGIN_Y,
    ROW_SHIFT,
    SHOOTER_X,
    SHOOTER_Y,
    SNAP_BACKOFF,
    SNAP_X_FUDGE,
    STEP_BACK,
    WALL_LEFT,
    WALL_RIGHT,
)

OK = 0
NEVER_LANDED = 1
UNSUPPORTED = 2  # board outside the grid the kernels handle; the caller uses the pure Python path

MIN_GRID_ROWS = 40
GRID_ROW_MARGIN = 3
COLLISION_DIST2 = COLLISION_DIST * COLLISION_DIST
# Squared distances this far from COLLISION_DIST2 decide the comparison on their own: the rounding error of
# dx*dx + dy*dy and of hypot is around 1e-15 relative, many orders of magnitude below this band.
BAND = 1e-9
SURE_HIT2 = COLLISION_DIST2 * (1 - BAND)
SURE_MISS2 = COLLISION_DIST2 * (1 + BAND)
SPLITTER = 134217729.0  # 2**27 + 1, Veltkamp split for an exact product


@njit(cache=True)
def _two_product(a, b):
    """(hi, lo) with hi = fl(a*b) and hi + lo == a*b exactly (Dekker), what fma(a, b, -hi) gives."""
    p = a * b
    t = SPLITTER * a
    ah = t - (t - a)
    al = a - ah
    t = SPLITTER * b
    bh = t - (t - b)
    bl = b - bh
    return p, ((ah * bh - p) + ah * bl + al * bh) + al * bl


@njit(cache=True)
def _cpython_hypot(a, b):
    """CPython 3.12 math.hypot(a, b) for finite inputs: vector_norm() with n == 2."""
    a, b = abs(a), abs(b)
    m = max(a, b)
    if m == 0.0:
        return m
    _, max_e = math.frexp(m)
    if max_e < -1023:
        return math.hypot(a, b)  # subnormal path, never reached by the simulator
    scale = math.ldexp(1.0, -max_e)
    csum, frac1, frac2 = 1.0, 0.0, 0.0
    for x in (a * scale, b * scale):
        hi, lo = _two_product(x, x)
        s = csum + hi
        frac2 += (csum - s) + hi
        csum = s
        frac1 += lo
    h = math.sqrt(csum - 1.0 + (frac1 + frac2))
    hi, lo = _two_product(-h, h)
    s = csum + hi
    frac2 += (csum - s) + hi
    csum = s
    frac1 += lo
    x = csum - 1.0 + (frac1 + frac2)
    h += x / (2.0 * h)
    return h / scale


@njit(cache=True)
def _collides(dx, dy):
    """Exactly math.hypot(dx, dy) < COLLISION_DIST as CPython evaluates it."""
    if abs(dx) >= COLLISION_DIST or abs(dy) >= COLLISION_DIST:
        return False  # hypot >= max(|dx|, |dy|)
    d2 = dx * dx + dy * dy
    if d2 < SURE_HIT2:
        return True
    if d2 > SURE_MISS2:
        return False
    return _cpython_hypot(dx, dy) < COLLISION_DIST


@njit(cache=True)
def _build_grid(keys, colors):
    """Colour grid [by, bx] (-1 empty). Returns an empty grid when a cell is outside it."""
    n = keys.shape[0]
    max_by = 0
    for i in range(n):
        bx, by = keys[i, 0], keys[i, 1]
        if bx < 0 or bx >= BOARD_WIDTH or by < 0:
            return np.full((0, BOARD_WIDTH), -1, dtype=np.int64)
        if by > max_by:
            max_by = by
    grid = np.full((max(MIN_GRID_ROWS, max_by + GRID_ROW_MARGIN), BOARD_WIDTH), -1, dtype=np.int64)
    for i in range(n):
        grid[keys[i, 1], keys[i, 0]] = colors[i]
    return grid


@njit(cache=True)
def _js_round(value):
    return math.floor(value + 0.5)


@njit(cache=True)
def _snap(grid, x, y, vx, vy):
    base_x, base_y = x - ARRIVAL_OFFSET * vx, y - ARRIVAL_OFFSET * vy
    counter = 0
    while True:
        px = base_x - SNAP_BACKOFF * counter * vx - ORIGIN_X
        py = base_y - SNAP_BACKOFF * counter * vy - ORIGIN_Y
        counter += 1
        by = max(0, int(_js_round(py / BUBBLE_SIZE)))
        px -= (by % 2) * ROW_SHIFT
        bx = int(_js_round(px * SNAP_X_FUDGE / BUBBLE_SIZE))
        bx = min(BOARD_WIDTH - 1, max(0, bx))
        if by >= grid.shape[0] or grid[by, bx] < 0:
            return bx, by


@njit(cache=True)
def _fly(grid, keys, vx, vy):
    """(status, bx, by): the same frame loop as SimGame._fly."""
    n = keys.shape[0]
    cxs = np.empty(n, dtype=np.float64)
    cys = np.empty(n, dtype=np.float64)
    for i in range(n):
        bx, by = keys[i, 0], keys[i, 1]
        cxs[i] = ORIGIN_X + bx * BUBBLE_SIZE + (by % 2) * ROW_SHIFT
        cys[i] = ORIGIN_Y + by * BUBBLE_SIZE
    x, y = float(SHOOTER_X), float(SHOOTER_Y)
    for _ in range(MAX_FLIGHT_FRAMES):
        x += vx
        y += vy
        if x > WALL_RIGHT:
            vx, x = -vx, float(WALL_RIGHT)
        elif x < WALL_LEFT:
            vx, x = -vx, float(WALL_LEFT)
        ahead_x, ahead_y = x + vx, y + vy
        if ahead_y < CEILING_Y:
            bx, by = _snap(grid, ahead_x, CEILING_SNAP_Y, vx, vy)
            return OK, bx, by
        for i in range(n):
            if _collides(ahead_x - cxs[i], ahead_y - cys[i]):
                bx, by = _snap(grid, ahead_x - STEP_BACK * vx, ahead_y - STEP_BACK * vy, vx, vy)
                return OK, bx, by
    return NEVER_LANDED, -1, -1


@njit(cache=True)
def _neighbors(bx, by, out):
    """Fill out[:, :] with SimGame's neighbours of (bx, by); returns how many."""
    k = 0
    for nx in (bx - 1, bx + 1):
        if 0 <= nx < BOARD_WIDTH:
            out[k, 0], out[k, 1] = nx, by
            k += 1
    other = bx - 1 if by % 2 == 0 else bx + 1
    for ny in (by - 1, by + 1):
        if ny < 0:
            continue
        for nx in (bx, other):
            if 0 <= nx < BOARD_WIDTH:
                out[k, 0], out[k, 1] = nx, ny
                k += 1
    return k


@njit(cache=True)
def _occupied(grid, bx, by):
    return by < grid.shape[0] and grid[by, bx] >= 0


@njit(cache=True)
def _pop_cluster(grid, bx, by, color, removed):
    """Same-colour flood fill from (bx, by); clears it from the grid if it pops. Returns its size."""
    rows = grid.shape[0]
    seen = np.zeros((rows, BOARD_WIDTH), dtype=np.bool_)
    nb = np.empty((6, 2), dtype=np.int64)
    seen[by, bx] = True
    removed[0, 0], removed[0, 1] = bx, by
    size = 1
    head = 0
    while head < size:  # the order of the flood fill does not change the set it finds
        cx, cy = removed[head, 0], removed[head, 1]
        head += 1
        for j in range(_neighbors(cx, cy, nb)):
            nx, ny = nb[j, 0], nb[j, 1]
            if ny < rows and not seen[ny, nx] and grid[ny, nx] == color:
                seen[ny, nx] = True
                removed[size, 0], removed[size, 1] = nx, ny
                size += 1
    if size >= MIN_POP:
        for i in range(size):
            grid[removed[i, 1], removed[i, 0]] = -1
    return size


@njit(cache=True)
def _drop_hanging(grid, removed, start):
    """Clear cells not connected to row 0; appends them to removed[start:]. Returns how many."""
    rows = grid.shape[0]
    anchored = np.zeros((rows, BOARD_WIDTH), dtype=np.bool_)
    stack = np.empty((rows * BOARD_WIDTH, 2), dtype=np.int64)
    nb = np.empty((6, 2), dtype=np.int64)
    top = 0
    for bx in range(BOARD_WIDTH):
        if grid[0, bx] >= 0:
            anchored[0, bx] = True
            stack[top, 0], stack[top, 1] = bx, 0
            top += 1
    while top > 0:
        top -= 1
        cx, cy = stack[top, 0], stack[top, 1]
        for j in range(_neighbors(cx, cy, nb)):
            nx, ny = nb[j, 0], nb[j, 1]
            if _occupied(grid, nx, ny) and not anchored[ny, nx]:
                anchored[ny, nx] = True
                stack[top, 0], stack[top, 1] = nx, ny
                top += 1
    k = start
    for by in range(rows):
        for bx in range(BOARD_WIDTH):
            if grid[by, bx] >= 0 and not anchored[by, bx]:
                grid[by, bx] = -1
                removed[k, 0], removed[k, 1] = bx, by
                k += 1
    return k - start


@njit(cache=True)
def shoot_kernel(keys, colors, vx, vy, color):
    """Fly, place, pop, drop. Returns (status, bx, by, n_cluster, n_hanging, removed) where removed holds
    the cluster (only if it popped, n_cluster >= MIN_POP) followed by the hanging cells."""
    empty = np.empty((0, 2), dtype=np.int64)
    grid = _build_grid(keys, colors)
    if grid.shape[0] == 0:
        return UNSUPPORTED, -1, -1, 0, 0, empty
    status, bx, by = _fly(grid, keys, vx, vy)
    if status != OK:
        return status, -1, -1, 0, 0, empty
    if by >= grid.shape[0]:
        return UNSUPPORTED, -1, -1, 0, 0, empty
    grid[by, bx] = color
    removed = np.empty((keys.shape[0] + 1, 2), dtype=np.int64)
    size = _pop_cluster(grid, bx, by, color, removed)
    if size < MIN_POP:
        return OK, bx, by, size, 0, empty
    hanging = _drop_hanging(grid, removed, size)
    return OK, bx, by, size, hanging, removed[: size + hanging]


@njit(cache=True)
def fly_kernel(keys, colors, vx, vy):
    grid = _build_grid(keys, colors)
    if grid.shape[0] == 0:
        return UNSUPPORTED, -1, -1
    return _fly(grid, keys, vx, vy)


@njit(cache=True)
def resolve_kernel(keys, colors, bx, by, color):
    """SimGame._resolve_landing for a cell already on the board: (status, n_cluster, n_hanging, removed)."""
    empty = np.empty((0, 2), dtype=np.int64)
    grid = _build_grid(keys, colors)
    if grid.shape[0] == 0 or by >= grid.shape[0]:
        return UNSUPPORTED, 0, 0, empty
    removed = np.empty((keys.shape[0] + 1, 2), dtype=np.int64)
    size = _pop_cluster(grid, bx, by, color, removed)
    if size < MIN_POP:
        return OK, size, 0, empty
    hanging = _drop_hanging(grid, removed, size)
    return OK, size, hanging, removed[: size + hanging]


@njit(cache=True)
def drop_kernel(keys, colors):
    """SimGame._drop_hanging: (status, removed)."""
    grid = _build_grid(keys, colors)
    if grid.shape[0] == 0:
        return UNSUPPORTED, np.empty((0, 2), dtype=np.int64)
    removed = np.empty((keys.shape[0], 2), dtype=np.int64)
    hanging = _drop_hanging(grid, removed, 0)
    return OK, removed[:hanging]


def board_arrays(board):
    """The dict as (keys (n, 2), colours (n,)) int64 arrays in dict order."""
    n = len(board)
    keys = np.fromiter(itertools.chain.from_iterable(board), dtype=np.int64, count=2 * n).reshape(n, 2)
    colors = np.fromiter(board.values(), dtype=np.int64, count=n)
    return keys, colors
