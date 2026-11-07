import math
import random

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

PALETTE = (
    (0, 188, 31),
    (0, 21, 188),
    (188, 1, 0),
    (187, 187, 0),
    (179, 0, 188),
    (0, 177, 188),
)


def _js_round(value):
    return math.floor(value + 0.5)


def _cell_center(bx, by):
    return ORIGIN_X + bx * BUBBLE_SIZE + (by % 2) * ROW_SHIFT, ORIGIN_Y + by * BUBBLE_SIZE


def _neighbors(bx, by):
    cells = [(bx - 1, by), (bx + 1, by)]
    side = (bx, bx - 1) if by % 2 == 0 else (bx, bx + 1)
    for ny in (by - 1, by + 1):
        cells.extend((nx, ny) for nx in side)
    return [(x, y) for x, y in cells if 0 <= x < BOARD_WIDTH and y >= 0]


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

    def _fly(self, vx, vy):
        x, y = SHOOTER_X, SHOOTER_Y
        centers = [(_cell_center(*cell), cell) for cell in self.board]
        for _ in range(MAX_FLIGHT_FRAMES):
            x += vx
            y += vy
            if x > WALL_RIGHT:
                vx, x = -vx, WALL_RIGHT
            elif x < WALL_LEFT:
                vx, x = -vx, WALL_LEFT
            ahead_x, ahead_y = x + vx, y + vy
            if ahead_y < CEILING_Y:
                return self._snap(ahead_x, CEILING_SNAP_Y, vx, vy)
            for (cx, cy), _ in centers:
                if math.hypot(ahead_x - cx, ahead_y - cy) < COLLISION_DIST:
                    return self._snap(ahead_x - STEP_BACK * vx, ahead_y - STEP_BACK * vy, vx, vy)
        raise RuntimeError("shot never landed")

    def _snap(self, x, y, vx, vy):
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
            if (bx, by) not in self.board:
                return bx, by

    def _cluster(self, cell, color):
        seen, frontier = {cell}, [cell]
        while frontier:
            cur = frontier.pop()
            for n in _neighbors(*cur):
                if n not in seen and self.board.get(n) == color:
                    seen.add(n)
                    frontier.append(n)
        return seen

    def _resolve_landing(self, cell, color):
        cluster = self._cluster(cell, color)
        if len(cluster) < MIN_POP:
            return False
        for k in range(1, len(cluster) + 1):
            self.score += POP_SCORE_UNIT * math.ceil(k / POP_SCORE_GROUP)
        for c in cluster:
            del self.board[c]
        self.score += HANG_SCORE * self._drop_hanging()
        return True

    def _drop_hanging(self):
        anchored = {c for c in self.board if c[1] == 0}
        frontier = list(anchored)
        while frontier:
            cur = frontier.pop()
            for n in _neighbors(*cur):
                if n in self.board and n not in anchored:
                    anchored.add(n)
                    frontier.append(n)
        hanging = [c for c in self.board if c not in anchored]
        for c in hanging:
            del self.board[c]
        return len(hanging)

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
