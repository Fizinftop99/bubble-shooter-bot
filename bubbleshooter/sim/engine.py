import math
import random

import numpy as np
from numba import njit

from .. import layout

BOARD_WIDTH = 17
INITIAL_ROWS = 9
COLOR_COUNT = 6
BUBBLE_SIZE = 32
ORIGIN_X = 40
ORIGIN_Y = 40
ROW_SHIFT = BUBBLE_SIZE / 2
WALL_LEFT = 37
WALL_RIGHT = 561
SHOOTER_X = 296
SHOOTER_Y = 552
LAUNCH_SPEED = 18
MAX_ANGLE_DEG = 75
COLLISION_DIST = 0.75 * BUBBLE_SIZE
STEP_BACK = 0.75
ARRIVAL_OFFSET = 0.05
SNAP_BACKOFF = 0.25
CEILING_Y = BUBBLE_SIZE
CEILING_SNAP_Y = CEILING_Y + 0.01
MAX_FLIGHT_FRAMES = 2000
SNAP_X_FUDGE = 0.9999
START_LIVES = 5
LOSS_ROW = 15
MIN_POP = 3
POP_SCORE_UNIT = 10
POP_SCORE_GROUP = 3
HANG_SCORE = 100
SCREEN_SCALE = 749 / 800
EMPTY = -1
KEPT, POPPED, HUNG = 0, 1, 2
GRID_MIN_ROWS = 24

PALETTE = (
    (0, 188, 31),
    (0, 21, 188),
    (188, 1, 0),
    (187, 187, 0),
    (179, 0, 188),
    (0, 177, 188),
)


@njit(cache=True)
def _js_round(value):
    return int(math.floor(value + 0.5))


@njit(cache=True)
def _snap_cell(grid, x, y, vx, vy):
    base_x, base_y = x - ARRIVAL_OFFSET * vx, y - ARRIVAL_OFFSET * vy
    counter = 0
    while True:
        px = base_x - SNAP_BACKOFF * counter * vx - ORIGIN_X
        py = base_y - SNAP_BACKOFF * counter * vy - ORIGIN_Y
        counter += 1
        by = max(0, _js_round(py / BUBBLE_SIZE))
        px -= (by % 2) * ROW_SHIFT
        bx = _js_round(px * SNAP_X_FUDGE / BUBBLE_SIZE)
        bx = min(BOARD_WIDTH - 1, max(0, bx))
        if by >= grid.shape[0] or grid[by, bx] == EMPTY:
            return bx, by


@njit(cache=True)
def _fly_cell(grid, vx, vy):
    x, y = float(SHOOTER_X), float(SHOOTER_Y)
    last_row = -1
    for by in range(grid.shape[0]):
        for bx in range(BOARD_WIDTH):
            if grid[by, bx] != EMPTY:
                last_row = by
                break
    for _ in range(MAX_FLIGHT_FRAMES):
        x += vx
        y += vy
        if x > WALL_RIGHT:
            vx, x = -vx, float(WALL_RIGHT)
        elif x < WALL_LEFT:
            vx, x = -vx, float(WALL_LEFT)
        ahead_x, ahead_y = x + vx, y + vy
        if ahead_y < CEILING_Y:
            return _snap_cell(grid, ahead_x, CEILING_SNAP_Y, vx, vy)
        for by in range(last_row + 1):
            cy = ORIGIN_Y + by * BUBBLE_SIZE
            if abs(ahead_y - cy) >= COLLISION_DIST:
                continue
            for bx in range(BOARD_WIDTH):
                if grid[by, bx] == EMPTY:
                    continue
                cx = ORIGIN_X + bx * BUBBLE_SIZE + (by % 2) * ROW_SHIFT
                if math.hypot(ahead_x - cx, ahead_y - cy) < COLLISION_DIST:
                    return _snap_cell(grid, ahead_x - STEP_BACK * vx, ahead_y - STEP_BACK * vy, vx, vy)
    return -1, -1


@njit(cache=True)
def _neighbor_cells(bx, by, rows):
    out = np.empty((6, 2), dtype=np.int64)
    count = 0
    side = bx - 1 if by % 2 == 0 else bx + 1
    for nx, ny in ((bx - 1, by), (bx + 1, by), (bx, by - 1), (side, by - 1), (bx, by + 1), (side, by + 1)):
        if 0 <= nx < BOARD_WIDTH and 0 <= ny < rows:
            out[count, 0] = nx
            out[count, 1] = ny
            count += 1
    return out, count


@njit(cache=True)
def _mark_hanging(grid, removed):
    rows = grid.shape[0]
    anchored = np.zeros(grid.shape, dtype=np.bool_)
    stack = np.empty((rows * BOARD_WIDTH, 2), dtype=np.int64)
    top = 0
    for bx in range(BOARD_WIDTH):
        if grid[0, bx] != EMPTY and removed[0, bx] == KEPT:
            anchored[0, bx] = True
            stack[top, 0], stack[top, 1] = bx, 0
            top += 1
    while top > 0:
        top -= 1
        cells, count = _neighbor_cells(stack[top, 0], stack[top, 1], rows)
        for k in range(count):
            nx, ny = cells[k, 0], cells[k, 1]
            if grid[ny, nx] != EMPTY and removed[ny, nx] == KEPT and not anchored[ny, nx]:
                anchored[ny, nx] = True
                stack[top, 0], stack[top, 1] = nx, ny
                top += 1
    hanging = 0
    for by in range(rows):
        for bx in range(BOARD_WIDTH):
            if grid[by, bx] != EMPTY and removed[by, bx] == KEPT and not anchored[by, bx]:
                removed[by, bx] = HUNG
                hanging += 1
    return hanging


@njit(cache=True)
def _resolve_cells(grid, bx, by, color):
    rows = grid.shape[0]
    removed = np.zeros(grid.shape, dtype=np.int8)
    stack = np.empty((rows * BOARD_WIDTH, 2), dtype=np.int64)
    removed[by, bx] = POPPED
    stack[0, 0], stack[0, 1] = bx, by
    top, size = 1, 1
    while top > 0:
        top -= 1
        cells, count = _neighbor_cells(stack[top, 0], stack[top, 1], rows)
        for k in range(count):
            nx, ny = cells[k, 0], cells[k, 1]
            if grid[ny, nx] == color and removed[ny, nx] == KEPT:
                removed[ny, nx] = POPPED
                stack[top, 0], stack[top, 1] = nx, ny
                top += 1
                size += 1
    if size < MIN_POP:
        return removed, size, 0
    return removed, size, _mark_hanging(grid, removed)


class SimGame:
    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.board = {}
        self.score = 0
        self.shots = 0
        self.pushes = 0
        self.last_cell = None
        self.won = False
        self.lost = False
        self.lives = START_LIVES
        self.max_lives = START_LIVES
        for by in range(INITIAL_ROWS):
            for bx in range(BOARD_WIDTH):
                self.board[(bx, by)] = self.rng.randrange(COLOR_COUNT)
        self.queue = self._random_color(set(self.board.values()))
        self._promote()

    @property
    def is_over(self):
        return self.won or self.lost

    @property
    def final_score(self):
        return self.score * 2 if self.won else self.score

    def shooter_color(self):
        return PALETTE[self.ready]

    @staticmethod
    def layout_position(cell):
        bx, by = cell
        shifted = by % 2 == 1
        return (
            layout.COL0_X + bx * layout.COL_PITCH + (layout.ODD_ROW_SHIFT if shifted else 0),
            layout.GRID_ROW0_Y + by * layout.ROW_PITCH,
        )

    def bubbles(self):
        out = []
        for (bx, by), color in self.board.items():
            x, y = self.layout_position((bx, by))
            out.append({"x": x, "y": y, "color": PALETTE[color], "row": by, "shifted": by % 2 == 1})
        return out

    def _random_color(self, present):
        return self.rng.choice(sorted(present))

    def _present_colors(self, include_queue):
        present = set(self.board.values())
        present.add(self.ready)
        if include_queue:
            present.add(self.queue)
        return present

    def _promote(self):
        self.ready = self.queue
        self.queue = self._random_color(self._present_colors(include_queue=False))

    def shoot(self, click):
        if self.is_over:
            raise RuntimeError("game is over")
        self.shots += 1
        color = self.ready
        vx, vy = self._launch_velocity(click[0] / SCREEN_SCALE, click[1] / SCREEN_SCALE)
        cell = self._fly(vx, vy)
        self.last_cell = cell
        self.board[cell] = color
        popped = self._resolve_landing(cell, color)
        self._promote()
        if not popped:
            self._lose_life()
        self._finish_turn()
        return popped

    @staticmethod
    def _launch_velocity(mouse_x, mouse_y):
        dx, dy = SHOOTER_X - mouse_x, SHOOTER_Y - mouse_y
        angle = math.degrees(math.atan2(dy, dx)) - 90
        if angle < -180 or angle > MAX_ANGLE_DEG:
            angle = MAX_ANGLE_DEG
        elif angle < -MAX_ANGLE_DEG:
            angle = -MAX_ANGLE_DEG
        rad = math.radians(angle + 90)
        return -LAUNCH_SPEED * math.cos(rad), -LAUNCH_SPEED * math.sin(rad)

    def _grid(self):
        rows = max(GRID_MIN_ROWS, max((by for _, by in self.board), default=0) + 2)
        grid = np.full((rows, BOARD_WIDTH), EMPTY, dtype=np.int8)
        for (bx, by), color in self.board.items():
            grid[by, bx] = color
        return grid

    def _fly(self, vx, vy):
        bx, by = _fly_cell(self._grid(), float(vx), float(vy))
        if bx < 0:
            raise RuntimeError("shot never landed")
        return int(bx), int(by)

    def _remove(self, removed):
        self.board = {cell: color for cell, color in self.board.items() if removed[cell[1], cell[0]] == KEPT}

    def _resolve_landing(self, cell, color):
        removed, size, hanging = _resolve_cells(self._grid(), cell[0], cell[1], color)
        if size < MIN_POP:
            return False
        for k in range(1, size + 1):
            self.score += POP_SCORE_UNIT * math.ceil(k / POP_SCORE_GROUP)
        self._remove(removed)
        self.score += HANG_SCORE * int(hanging)
        return True

    def _drop_hanging(self):
        grid = self._grid()
        removed = np.zeros(grid.shape, dtype=np.int8)
        hanging = int(_mark_hanging(grid, removed))
        if hanging:
            self._remove(removed)
        return hanging

    def _lose_life(self):
        if self.lives > 0:
            self.lives -= 1
        else:
            self._push_rows()

    def _push_rows(self):
        self.pushes += 1
        present = self._present_colors(include_queue=True)
        rows = 1 + COLOR_COUNT - len(present)
        limit = len(present) - 1
        self.max_lives -= 1
        if self.max_lives <= 0:
            self.max_lives = limit
        self.max_lives = min(self.max_lives, limit)
        self.lives = self.max_lives
        for _ in range(rows):
            self.board = {(bx, by + 1): color for (bx, by), color in self.board.items()}
            for bx in range(BOARD_WIDTH):
                self.board[(bx, 0)] = self._random_color(present)
        self._drop_hanging()

    def _finish_turn(self):
        if not self.board:
            self.won = True
        elif any(by >= LOSS_ROW for _, by in self.board):
            self.lost = True
