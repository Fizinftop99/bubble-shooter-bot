from PIL import Image, ImageDraw

from bubbleshooter import layout, vision

BACKGROUND = vision.EMPTY_SLOT_COLOR
GREY = (157, 157, 157)
RED = (188, 1, 0)


def frame(lives, next_color=RED):
    img = Image.new("RGB", (749, 562), BACKGROUND)
    draw = ImageDraw.Draw(img)
    r = layout.BUBBLE_RADIUS - 1
    x, y = layout.NEXT_BUBBLE_POS
    if next_color:
        draw.ellipse((x - r, y - r, x + r, y + r), fill=next_color)
    for x in layout.LIFE_SLOT_XS[:lives]:
        draw.ellipse((x - r, layout.LIFE_SLOT_Y - r, x + r, layout.LIFE_SLOT_Y + r), fill=GREY)
    return img


def test_lives_are_the_filled_magazine_slots():
    for lives in range(len(layout.LIFE_SLOT_XS) + 1):
        assert vision.count_lives(frame(lives)) == lives


def test_next_colour_is_read_and_missing_one_is_none():
    assert vision.get_next_color(frame(3)) == RED
    assert vision.get_next_color(frame(3, next_color=None)) is None
