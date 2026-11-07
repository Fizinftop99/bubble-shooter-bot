import os
from dataclasses import dataclass, field, fields
from pathlib import Path

from dotenv import load_dotenv

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


@dataclass(frozen=True)
class BrowserSettings:
    browser_path: str = r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe"
    game_url: str = "https://www.bubbleshooter.com/"
    viewport_width: int = 1280
    viewport_height: int = 900
    canvas_timeout_ms: int = 30_000
    block_ads: bool = True
    user_data_dir: str = ".brave-profile"

    @property
    def profile_path(self):
        path = Path(self.user_data_dir)
        return path if path.is_absolute() else ENV_FILE.parent / path

    @property
    def viewport(self):
        return {"width": self.viewport_width, "height": self.viewport_height}


@dataclass(frozen=True)
class PacingSettings:
    aim_speed_px_s: float = 700
    aim_min_s: float = 0.25
    aim_max_s: float = 1.0
    shot_wait_s: float = 1.2
    waypoint_count_weights: tuple = (0.3, 0.5, 0.2)


@dataclass(frozen=True)
class StrategyWeights:
    drop_weight: float = 10.0
    depth_weight: float = 0.15
    miss_base: float = -3.0
    friend_weight: float = 0.4
    growth_penalty: float = 1.5
    pressure_row: int = 8
    front_weight: float = 4.0


@dataclass(frozen=True)
class RunSettings:
    screenshot_dir: str = "screens"
    max_play_minutes: float = 0
    max_shots: int = 0


@dataclass(frozen=True)
class Settings:
    browser: BrowserSettings = field(default_factory=BrowserSettings)
    pacing: PacingSettings = field(default_factory=PacingSettings)
    weights: StrategyWeights = field(default_factory=StrategyWeights)
    run: RunSettings = field(default_factory=RunSettings)


def _from_env(cls):
    overrides = {}
    for f in fields(cls):
        raw = os.getenv(f.name.upper(), "").strip()
        if not raw:
            continue
        default = getattr(cls(), f.name)
        if isinstance(default, bool):
            overrides[f.name] = raw.lower() in ("1", "true", "yes", "on")
        elif isinstance(default, tuple):
            overrides[f.name] = tuple(float(part) for part in raw.split(","))
        else:
            overrides[f.name] = type(default)(raw)
    return cls(**overrides)


def load_settings():
    load_dotenv(ENV_FILE)
    return Settings(
        browser=_from_env(BrowserSettings),
        pacing=_from_env(PacingSettings),
        weights=_from_env(StrategyWeights),
        run=_from_env(RunSettings),
    )
