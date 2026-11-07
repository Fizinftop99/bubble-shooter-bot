from bubbleshooter.browser import GameBrowser
from bubbleshooter.config import load_settings
from bubbleshooter.game import GameLoop


def main():
    settings = load_settings()
    with GameBrowser(settings) as browser:
        browser.open_game()
        GameLoop(browser, settings).run()
        if settings.browser.keep_open and browser.is_open():
            print("Browser left open. Close the window (or press Ctrl+C) to exit.")
            try:
                browser.wait_until_closed()
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()
