import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_GAME_URL = "https://www.bubbleshooter.com/"
DEFAULT_BROWSER_PATH = r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe"
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


@dataclass(frozen=True)
class Settings:
    browser_path: str
    game_url: str


def load_settings():
    load_dotenv(ENV_FILE)
    return Settings(
        browser_path=os.getenv("BROWSER_PATH", DEFAULT_BROWSER_PATH),
        game_url=os.getenv("GAME_URL", DEFAULT_GAME_URL),
    )
