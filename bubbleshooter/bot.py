import random
import time

JITTER_PX = 5
CLICK_DELAY_RANGE_S = (0.05, 0.15)


def jittered_point(canvas_bbox, target, jitter=JITTER_PX, rng=random):
    x = canvas_bbox["x"] + target[0] + rng.uniform(-jitter, jitter)
    y = canvas_bbox["y"] + target[1] + rng.uniform(-jitter, jitter)
    return x, y


def shoot(page, canvas_bbox, target):
    x, y = jittered_point(canvas_bbox, target)
    page.mouse.move(x, y)
    time.sleep(random.uniform(*CLICK_DELAY_RANGE_S))
    page.mouse.click(x, y)
    return x, y
