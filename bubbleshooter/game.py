import time
from pathlib import Path

from . import bot, strategy, vision

RETRY_WAIT_S = 0.5
LOW_BUBBLE_COUNT = 3
LOW_COUNT_LIMIT = 4
STALL_LIMIT = 8
EMPTY_WAIT_LIMIT = 20
ERROR_REOPEN_LIMIT = 3
SETTLE_POLL_S = 0.04
SETTLE_TIMEOUT_S = 3.0
SETTLE_STABLE_FRAMES = 2


class GameLoop:
    def __init__(self, browser, settings):
        self._browser = browser
        self._settings = settings
        self._aimer = bot.HumanAimer(settings.pacing)
        self._shots = 0
        self._low_streak = 0
        self._stall_streak = 0
        self._empty_streak = 0
        self._error_streak = 0
        self._last_sig = None
        self._last_bbox = None
        self._window_closed = False
        self._game_over = False
        self._started = time.monotonic()

    def run(self):
        try:
            while self._stop_reason() is None:
                self._step()
        except KeyboardInterrupt:
            self._print_status("stopped by user", None)
            return
        self._print_status(self._stop_reason(), self._save_final_screenshot())

    def _stop_reason(self):
        run = self._settings.run
        if self._window_closed:
            return "browser window closed"
        if self._game_over or self._low_streak >= LOW_COUNT_LIMIT or self._stall_streak >= STALL_LIMIT or self._empty_streak >= EMPTY_WAIT_LIMIT:
            return "game over"
        if run.max_shots and self._shots >= run.max_shots:
            return "shot limit reached"
        if run.max_play_minutes and time.monotonic() - self._started >= run.max_play_minutes * 60:
            return "time limit reached"
        return None

    def _step(self):
        try:
            img, bbox = self._browser.grab_frame()
        except Exception as e:
            self._on_error(e)
            return
        self._error_streak = 0
        self._last_bbox = bbox
        if vision.is_game_over(img):
            self._game_over = True
            return
        shooter = vision.get_shooter_color(img)
        bubbles = vision.find_grid_bubbles(img)
        result = strategy.choose_target(shooter, bubbles, weights=self._settings.weights)
        if shooter is None or result is None:
            self._empty_streak += 1
            print(f"[wait] shooter={shooter} bubbles={len(bubbles)}")
            time.sleep(RETRY_WAIT_S)
            return
        self._empty_streak = 0
        self._track_stuck(shooter, bubbles)
        if self._stop_reason() is not None:
            return
        tx, ty, reason = result
        try:
            x, y = self._aimer.shoot(self._browser.page, bbox, (tx, ty))
        except Exception as e:
            self._on_error(e)
            return
        self._shots += 1
        print(f"#{self._shots} shooter={shooter} bubbles={len(bubbles)} rows={max(b['row'] for b in bubbles)} target=({tx},{ty}) click=({x:.0f},{y:.0f}) reason={reason}")
        if self._settings.pacing.adaptive_wait:
            self._wait_settled(bubbles)
        else:
            time.sleep(self._settings.pacing.shot_wait_s)

    def _wait_settled(self, before):
        before_sig = self._board_signature(before)
        deadline = time.monotonic() + SETTLE_TIMEOUT_S
        last, stable, changed = None, 0, False
        while time.monotonic() < deadline:
            try:
                img, _ = self._browser.grab_frame()
            except Exception:
                return
            if vision.is_game_over(img):
                return
            sig = self._board_signature(vision.find_grid_bubbles(img))
            changed = changed or sig != before_sig
            stable = stable + 1 if sig == last else 0
            last = sig
            if changed and stable >= SETTLE_STABLE_FRAMES:
                return
            time.sleep(SETTLE_POLL_S)

    @staticmethod
    def _board_signature(bubbles):
        return tuple(sorted((b["x"], b["y"]) for b in bubbles))

    def _on_error(self, error):
        print(f"[error] frame grab failed: {str(error).splitlines()[0]}")
        if not self._browser.is_open():
            self._window_closed = True
            return
        self._error_streak += 1
        if self._error_streak >= ERROR_REOPEN_LIMIT:
            print("[recover] reopening page")
            self._error_streak = 0
            try:
                self._browser.reopen()
            except Exception as e:
                print(f"[error] reopen failed: {str(e).splitlines()[0]}")
        time.sleep(RETRY_WAIT_S)

    def _track_stuck(self, shooter, bubbles):
        self._low_streak = self._low_streak + 1 if len(bubbles) <= LOW_BUBBLE_COUNT else 0
        sig = (len(bubbles), shooter, tuple((b["x"], b["y"]) for b in bubbles[:5]))
        self._stall_streak = self._stall_streak + 1 if sig == self._last_sig else 0
        self._last_sig = sig

    def _save_final_screenshot(self):
        if self._last_bbox is None or self._window_closed:
            return None
        shot_dir = Path(self._settings.run.screenshot_dir)
        shot_dir.mkdir(exist_ok=True)
        path = shot_dir / f"game_over_{time.strftime('%Y%m%d_%H%M%S')}.png"
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
