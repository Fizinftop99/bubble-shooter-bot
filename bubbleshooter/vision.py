import math

from . import layout

COLOR_TOLERANCE = 40
SAMPLE_RING_RADIUS = 8
SAMPLE_RING_POINTS = 8
BUBBLE_MAX_MIN_CHANNEL = 160
SOLID_RING_MIN_HITS = 7
GAME_OVER_LINE_Y = 240
GAME_OVER_LINE_XS = (230, 300, 360, 430, 460)
GAME_OVER_LINE_MIN_RG = 215
GAME_OVER_LINE_MAX_RG = 245
GAME_OVER_LINE_RG_DIFF = 6
GAME_OVER_LINE_MIN_B = 245


def color_distance(a, b):
    return math.sqrt(sum((p - q) ** 2 for p, q in zip(a, b)))


def colors_match(a, b):
    return color_distance(a, b) < COLOR_TOLERANCE


def is_bubble_color(rgb):
    return min(rgb) < BUBBLE_MAX_MIN_CHANNEL


def sample_color(img, cx, cy):
    # a ring avoids the specular highlight and dark rim at the bubble centre
    w, h = img.size
    samples = []
    for i in range(SAMPLE_RING_POINTS):
        a = 2 * math.pi * i / SAMPLE_RING_POINTS
        x = round(cx + SAMPLE_RING_RADIUS * math.cos(a))
        y = round(cy + SAMPLE_RING_RADIUS * math.sin(a))
        if 0 <= x < w and 0 <= y < h:
            samples.append(img.getpixel((x, y))[:3])
    if not samples:
        return (0, 0, 0)
    return tuple(sorted(s[c] for s in samples)[len(samples) // 2] for c in range(3))


def get_shooter_color(img):
    color = sample_color(img, *layout.SHOOTER_POS)
    return color if is_bubble_color(color) else None


def _is_solid_bubble(img, cx, cy):
    w, h = img.size
    hits = 0
    for i in range(SAMPLE_RING_POINTS):
        a = 2 * math.pi * i / SAMPLE_RING_POINTS
        x = round(cx + SAMPLE_RING_RADIUS * math.cos(a))
        y = round(cy + SAMPLE_RING_RADIUS * math.sin(a))
        if 0 <= x < w and 0 <= y < h and is_bubble_color(img.getpixel((x, y))[:3]):
            hits += 1
    return hits >= SOLID_RING_MIN_HITS


def _row_is_shifted(img, y, prev_shifted):
    left = img.getpixel((layout.FIELD_LEFT_X + 6, y))[:3]
    right = img.getpixel((layout.FIELD_RIGHT_X - 6, y))[:3]
    if is_bubble_color(left):
        return False
    if is_bubble_color(right):
        return True
    solid = {
        shifted: sum(
            _is_solid_bubble(img, layout.COL0_X + (layout.ODD_ROW_SHIFT if shifted else 0) + col * layout.COL_PITCH, y)
            for col in range(layout.COLS)
        )
        for shifted in (False, True)
    }
    if solid[False] != solid[True]:
        return solid[True] > solid[False]
    return not prev_shifted


def find_grid_bubbles(img):
    bubbles = []
    shifted = False
    row = 0
    y = layout.GRID_ROW0_Y
    while y <= layout.FIELD_BOTTOM_Y - layout.BUBBLE_RADIUS:
        shifted = _row_is_shifted(img, y, shifted)
        x0 = layout.COL0_X + (layout.ODD_ROW_SHIFT if shifted else 0)
        for col in range(layout.COLS):
            x = x0 + col * layout.COL_PITCH
            color = sample_color(img, x, y)
            if is_bubble_color(color):
                bubbles.append({"x": x, "y": y, "color": color, "row": row, "shifted": shifted})
        row += 1
        y += layout.ROW_PITCH
    return bubbles


def is_game_over(img):
    for x in GAME_OVER_LINE_XS:
        r, g, b = img.getpixel((x, GAME_OVER_LINE_Y))
        if not GAME_OVER_LINE_MIN_RG <= r <= GAME_OVER_LINE_MAX_RG or abs(r - g) > GAME_OVER_LINE_RG_DIFF or b < GAME_OVER_LINE_MIN_B:
            return False
    return True
