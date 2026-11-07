import sys

from PIL import Image

from bubbleshooter import layout, strategy, vision


def _debug_dump(img):
    bubbles = vision.find_grid_bubbles(img)
    shooter = vision.get_shooter_color(img)
    reps = []

    def label(c):
        for i, r in enumerate(reps):
            if vision.colors_match(r, c):
                return "RYBMGCOPQ"[i % 9]
        reps.append(c)
        return "RYBMGCOPQ"[(len(reps) - 1) % 9]

    print("shooter", shooter, label(shooter) if shooter else "-")
    rows = {}
    for b in bubbles:
        rows.setdefault(b["row"], []).append(b)
    for r in sorted(rows):
        line = [" "] * (2 * layout.COLS + 2)
        for b in rows[r]:
            line[(b["x"] - layout.FIELD_LEFT_X) // 15] = label(b["color"])
        print(f"{r:2d} " + " ".join(line))
    print("palette:", reps)
    print("target:", strategy.choose_target(shooter, bubbles))



if __name__ == "__main__":
    _debug_dump(Image.open(sys.argv[1]).convert("RGB"))
