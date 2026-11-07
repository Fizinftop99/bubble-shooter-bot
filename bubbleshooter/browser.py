import base64
import io
import time

from PIL import Image
from playwright.sync_api import sync_playwright

from . import adblock

GAME_FRAME_SELECTOR = "iframe[src*='bubbleshooter-game']"
SETTLE_WAIT_S = 1.2
SCREENSHOT_FORMAT = "png"
WINDOW_MARGIN_PX = (20, 140)


class GameBrowser:
    def __init__(self, settings):
        self._settings = settings
        self._playwright = None
        self._context = None
        self.page = None
        self._canvas = None
        self._cdp = None

    def __enter__(self):
        self._playwright = sync_playwright().start()
        self._launch()
        return self

    def _launch(self):
        cfg = self._settings.browser
        self._context = self._playwright.chromium.launch_persistent_context(
            str(cfg.profile_path),
            headless=False,
            executable_path=cfg.browser_path,
            viewport=cfg.viewport,
            args=[f"--window-size={cfg.viewport_width + WINDOW_MARGIN_PX[0]},{cfg.viewport_height + WINDOW_MARGIN_PX[1]}"],
            ignore_default_args=["--disable-extensions"],
        )
        if cfg.block_ads:
            adblock.install(self._context)
        pages = self._context.pages
        self.page = pages[0] if pages else self._context.new_page()
        for extra in pages[1:]:
            extra.close()

    def __exit__(self, *exc):
        for closer in (self._context and self._context.close, self._playwright and self._playwright.stop):
            try:
                closer and closer()
            except Exception:
                pass

    def is_open(self):
        try:
            return not self.page.is_closed()
        except Exception:
            return False

    def reopen(self):
        try:
            self.page.close()
        except Exception:
            pass
        try:
            self.page = self._context.new_page()
        except Exception:
            self._launch()
        self.open_game()

    def open_game(self):
        self.page.goto(self._settings.browser.game_url)
        self._canvas = self.page.frame_locator(GAME_FRAME_SELECTOR).locator("canvas")
        self._canvas.wait_for(state="visible", timeout=self._settings.browser.canvas_timeout_ms)
        self._canvas.scroll_into_view_if_needed()
        self._cdp = self._context.new_cdp_session(self.page)
        time.sleep(SETTLE_WAIT_S)

    def _capture_page(self):
        # fromSurface=False renders off-screen; the default capture makes the visible page flicker
        reply = self._cdp.send("Page.captureScreenshot", {"format": SCREENSHOT_FORMAT, "fromSurface": False})
        return Image.open(io.BytesIO(base64.b64decode(reply["data"]))).convert("RGB")

    def grab_frame(self):
        bbox = self._canvas.bounding_box()
        page_image = self._capture_page()
        scale = page_image.width / self.page.evaluate("window.innerWidth")
        box = tuple(round(v * scale) for v in (bbox["x"], bbox["y"], bbox["x"] + bbox["width"], bbox["y"] + bbox["height"]))
        size = (round(bbox["width"]), round(bbox["height"]))
        return page_image.crop(box).resize(size, Image.LANCZOS), bbox

    def click_canvas(self, bbox, pos):
        self.page.mouse.click(bbox["x"] + pos[0], bbox["y"] + pos[1])

    def save_screenshot(self, path, bbox):
        image, _ = self.grab_frame()
        image.save(path)
