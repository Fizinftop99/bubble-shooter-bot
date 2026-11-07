import time
from pathlib import Path

from . import bot, strategy, vision

SHOT_WAIT_S = 1.2
RETRY_WAIT_S = 0.5
LOW_BUBBLE_COUNT = 3
LOW_COUNT_LIMIT = 4
STALL_LIMIT = 8
EMPTY_WAIT_LIMIT = 20
ERROR_REOPEN_LIMIT = 3
GAME_OVER_SHOT_DIR = Path("screens")


class GameLoop:
    def __init__(self, browser):
        self._browser = browser
        self._shots = 0
        self._low_streak = 0
        self._stall_streak = 0
        self._empty_streak = 0
        self._error_streak = 0
        self._last_sig = None
        self._last_bbox = None
        self._started = time.monotonic()

    def run(self):
        try:
            while not self._finished():
                self._step()
        except KeyboardInterrupt:
            self._print_status("stopped by user", None)
            return
        self._print_status("game over", self._save_final_screenshot())

    def _finished(self):
        return self._low_streak >= LOW_COUNT_LIMIT or self._stall_streak >= STALL_LIMIT or self._empty_streak >= EMPTY_WAIT_LIMIT

    def _step(self):
        try:
            img, bbox = self._browser.grab_frame()
        except Exception as e:
            self._on_error(e)
            return
        self._error_streak = 0
        self._last_bbox = bbox
        shooter = vision.get_shooter_color(img)
        bubbles = vision.find_grid_bubbles(img)
        result = strategy.choose_target(shooter, bubbles)
        if shooter is None or result is None:
            self._empty_streak += 1
            print(f"[wait] shooter={shooter} bubbles={len(bubbles)}")
            time.sleep(RETRY_WAIT_S)
            return
        self._empty_streak = 0
        self._track_stuck(shooter, bubbles)
        if self._finished():
            return
        tx, ty, reason = result
        x, y = bot.shoot(self._browser.page, bbox, (tx, ty))
        self._shots += 1
        print(f"#{self._shots} shooter={shooter} bubbles={len(bubbles)} target=({tx},{ty}) click=({x:.0f},{y:.0f}) reason={reason}")
        time.sleep(SHOT_WAIT_S)

    def _on_error(self, error):
        print(f"[error] frame grab failed: {str(error).splitlines()[0]}")
        self._error_streak += 1
        if self._error_streak >= ERROR_REOPEN_LIMIT:
            print("[recover] reopening page")
            self._error_streak = 0
            self._browser.reopen()
        time.sleep(RETRY_WAIT_S)

    def _track_stuck(self, shooter, bubbles):
        self._low_streak = self._low_streak + 1 if len(bubbles) <= LOW_BUBBLE_COUNT else 0
        sig = (len(bubbles), shooter, tuple((b["x"], b["y"]) for b in bubbles[:5]))
        self._stall_streak = self._stall_streak + 1 if sig == self._last_sig else 0
        self._last_sig = sig

    def _save_final_screenshot(self):
        if self._last_bbox is None:
            return None
        GAME_OVER_SHOT_DIR.mkdir(exist_ok=True)
        path = GAME_OVER_SHOT_DIR / f"game_over_{time.strftime('%Y%m%d_%H%M%S')}.png"
        self._browser.save_screenshot(path, self._last_bbox)
        return path

    def _print_status(self, outcome, screenshot):
        elapsed = time.monotonic() - self._started
        print("=" * 50)
        print(f"STATUS: {outcome}")
        print(f"shots fired: {self._shots}")
        print(f"play time:   {int(elapsed // 60)}m {int(elapsed % 60)}s")
        if screenshot:
            print(f"final screen (shows score): {screenshot}")
        print("=" * 50)
