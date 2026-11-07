from bubbleshooter.browser import GameBrowser
from bubbleshooter.config import load_settings
from bubbleshooter.game import GameLoop


def main():
    settings = load_settings()
    with GameBrowser(settings) as browser:
        browser.open_game()
        GameLoop(browser, settings).run()


if __name__ == "__main__":
    main()
