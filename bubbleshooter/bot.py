import math
import random
import time

from . import layout

JITTER_PX = 5
AIM_MIN_S = 0.35
AIM_MAX_S = 1.4
AIM_SPEED_PX_S = 450
AIM_DURATION_VARIATION = 0.25
AIM_STEP_S = 0.016
AIM_ARC_PX = 14
AIM_TREMOR_PX = 0.6
AIM_SETTLE_RANGE_S = (0.12, 0.35)
CLICK_DELAY_RANGE_S = (0.04, 0.12)


def jittered_point(canvas_bbox, target, jitter=JITTER_PX, rng=random):
    x = canvas_bbox["x"] + target[0] + rng.uniform(-jitter, jitter)
    y = canvas_bbox["y"] + target[1] + rng.uniform(-jitter, jitter)
    return x, y


def _smoothstep(t):
    return t * t * (3 - 2 * t)


def aim_path(start, end, rng=random):
    dist = math.hypot(end[0] - start[0], end[1] - start[1])
    duration = min(AIM_MAX_S, max(AIM_MIN_S, dist / AIM_SPEED_PX_S))
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


class HumanAimer:
    def __init__(self):
        self._pos = None

    def shoot(self, page, canvas_bbox, target):
        end = jittered_point(canvas_bbox, target)
        start = self._pos or (canvas_bbox["x"] + layout.SHOOTER_POS[0], canvas_bbox["y"] + layout.SHOOTER_POS[1])
        points, delay = aim_path(start, end)
        for x, y in points:
            page.mouse.move(x, y)
            time.sleep(delay)
        time.sleep(random.uniform(*AIM_SETTLE_RANGE_S))
        page.mouse.click(*end)
        time.sleep(random.uniform(*CLICK_DELAY_RANGE_S))
        self._pos = end
        return end
