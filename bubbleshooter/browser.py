import io
import time

from PIL import Image
from playwright.sync_api import sync_playwright

from . import adblock

GAME_FRAME_SELECTOR = "iframe[src*='bubbleshooter-game']"
SETTLE_WAIT_S = 1.2


class GameBrowser:
    def __init__(self, settings):
        self._settings = settings
        self._playwright = None
        self._browser = None
        self._context = None
        self.page = None
        self._canvas = None

    def __enter__(self):
        self._playwright = sync_playwright().start()
        self._launch()
        return self

    def _launch(self):
        self._browser = self._playwright.chromium.launch(headless=False, executable_path=self._settings.browser.browser_path)
        self.page = self._browser.new_page(viewport=self._settings.browser.viewport)

    def __exit__(self, *exc):
        for closer in (self._browser and self._browser.close, self._playwright and self._playwright.stop):
            try:
                closer and closer()
            except Exception:
                pass

    def reopen(self):
        try:
            self.page.close()
        except Exception:
            pass
        try:
            self.page = self._browser.new_page(viewport=self._settings.browser.viewport)
        except Exception:
            self._launch()
        self.open_game()

    def open_game(self):
        self.page.goto(self._settings.browser.game_url)
        self._canvas = self.page.frame_locator(GAME_FRAME_SELECTOR).locator("canvas")
        self._canvas.wait_for(state="visible", timeout=self._settings.browser.canvas_timeout_ms)
        self._canvas.scroll_into_view_if_needed()
        time.sleep(SETTLE_WAIT_S)

    def grab_frame(self):
        bbox = self._canvas.bounding_box()
        png = self.page.screenshot(clip=bbox)
        return Image.open(io.BytesIO(png)).convert("RGB"), bbox

    def click_canvas(self, bbox, pos):
        self.page.mouse.click(bbox["x"] + pos[0], bbox["y"] + pos[1])

    def save_screenshot(self, path, bbox):
        self.page.screenshot(path=str(path), clip=bbox)
