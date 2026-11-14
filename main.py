import argparse
from dataclasses import replace

from bubbleshooter.browser import GameBrowser
from bubbleshooter.config import load_settings
from bubbleshooter.decider import make_decider
from bubbleshooter.game import GameLoop


def parse_args():
    parser = argparse.ArgumentParser(description="Bubble Shooter bot")
    parser.add_argument("--decider", choices=("rule", "model"), help="who picks each shot (default: DECIDER from .env, else rule)")
    return parser.parse_args()


def main():
    args = parse_args()
    settings = load_settings()
    if args.decider:
        settings = replace(settings, decider=replace(settings.decider, decider=args.decider))
    decider = make_decider(settings)
    if decider.name == "model":
        print(f"Loading model {settings.decider.model_name} at {settings.decider.model_url} ...")
        problem = decider.warm_up()
        if problem:
            print(f"[warn] model not reachable ({problem}); every shot will fall back to the rule")
    with GameBrowser(settings) as browser:
        browser.open_game()
        GameLoop(browser, settings, decider).run()
        if settings.browser.keep_open and browser.is_open():
            print("Browser left open. Close the window (or press Ctrl+C) to exit.")
            try:
                browser.wait_until_closed()
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()
