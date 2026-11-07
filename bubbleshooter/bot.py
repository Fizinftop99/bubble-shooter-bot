import math
import random
import time

from . import layout

JITTER_PX = 5
AIM_DURATION_VARIATION = 0.25
AIM_STEP_S = 0.016
AIM_ARC_PX = 14
AIM_TREMOR_PX = 0.6
AIM_SETTLE_RANGE_S = (0.12, 0.35)
CLICK_DELAY_RANGE_S = (0.04, 0.12)
WAYPOINT_ANGLE_OFFSET_DEG = (8, 45)
WAYPOINT_MIN_DEG = 12
WAYPOINT_MAX_DEG = 168
WAYPOINT_DISTANCE_PX = 250
WAYPOINT_PAUSE_RANGE_S = (0.08, 0.28)


def jittered_point(canvas_bbox, target, jitter=JITTER_PX, rng=random):
    x = canvas_bbox["x"] + target[0] + rng.uniform(-jitter, jitter)
    y = canvas_bbox["y"] + target[1] + rng.uniform(-jitter, jitter)
    return x, y


def _smoothstep(t):
    return t * t * (3 - 2 * t)


def aim_path(start, end, pacing, rng=random):
    dist = math.hypot(end[0] - start[0], end[1] - start[1])
    duration = min(pacing.aim_max_s, max(pacing.aim_min_s, dist / pacing.aim_speed_px_s))
    duration *= 1 + rng.uniform(-AIM_DURATION_VARIATION, AIM_DURATION_VARIATION)
    steps = max(2, int(duration / AIM_STEP_S))
    nx, ny = (-(end[1] - start[1]), end[0] - start[0]) if dist else (0, 0)
    norm = math.hypot(nx, ny) or 1
    arc = rng.uniform(-AIM_ARC_PX, AIM_ARC_PX) * min(1, dist / 200)
    points = []
    for i in range(1, steps + 1):
        t = i / steps
        e = _smoothstep(t)
        bend = math.sin(math.pi * e) * arc
        tremor = AIM_TREMOR_PX * (1 - t)
        points.append((
            start[0] + (end[0] - start[0]) * e + nx / norm * bend + rng.uniform(-tremor, tremor),
            start[1] + (end[1] - start[1]) * e + ny / norm * bend + rng.uniform(-tremor, tremor),
        ))
    points[-1] = end
    return points, duration / steps


def considered_points(origin, end, pacing, rng=random):
    count = rng.choices(range(len(pacing.waypoint_count_weights)), pacing.waypoint_count_weights)[0]
    target_deg = math.degrees(math.atan2(origin[1] - end[1], end[0] - origin[0]))
    points = []
    for _ in range(count):
        offset = rng.uniform(*WAYPOINT_ANGLE_OFFSET_DEG) * rng.choice((-1, 1))
        deg = min(WAYPOINT_MAX_DEG, max(WAYPOINT_MIN_DEG, target_deg + offset))
        rad = math.radians(deg)
        points.append((origin[0] + WAYPOINT_DISTANCE_PX * math.cos(rad), origin[1] - WAYPOINT_DISTANCE_PX * math.sin(rad)))
    return points


class HumanAimer:
    def __init__(self, pacing):
        self._pacing = pacing
        self._pos = None

    def _sweep(self, page, start, end):
        points, delay = aim_path(start, end, self._pacing)
        for x, y in points:
            page.mouse.move(x, y)
            time.sleep(delay)

    def shoot(self, page, canvas_bbox, target):
        if not self._pacing.human_aim:
            end = (canvas_bbox["x"] + target[0], canvas_bbox["y"] + target[1])
            page.mouse.move(*end)
            page.mouse.click(*end)
            return end
        end = jittered_point(canvas_bbox, target)
        origin = (canvas_bbox["x"] + layout.SHOOTER_POS[0], canvas_bbox["y"] + layout.SHOOTER_POS[1])
        pos = self._pos or origin
        for waypoint in considered_points(origin, end, self._pacing):
            self._sweep(page, pos, waypoint)
            time.sleep(random.uniform(*WAYPOINT_PAUSE_RANGE_S))
            pos = waypoint
        self._sweep(page, pos, end)
        time.sleep(random.uniform(*AIM_SETTLE_RANGE_S))
        page.mouse.click(*end)
        time.sleep(random.uniform(*CLICK_DELAY_RANGE_S))
        self._pos = end
        return end
