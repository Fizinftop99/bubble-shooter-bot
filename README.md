# Bubble Shooter Bot

Plays https://www.bubbleshooter.com/ automatically using Playwright and Pillow. It reads the game canvas from screenshots, simulates each possible shot (including wall bounces), and clicks the best one. When the game ends it stops and prints a status report.

## Requirements

- Python 3.12+
- Brave browser (or another Chromium-based browser, see [Configuration](#configuration))
- [uv](https://docs.astral.sh/uv/) package manager

## 1. Install the package manager (uv)

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

macOS / Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Alternative, with pip:

```bash
python -m pip install uv
```

Check the install (use `python -m uv --version` if you installed with pip):

```bash
uv --version
```

## 2. Create the virtual environment and install dependencies

From the project folder:

```bash
uv venv
uv sync
```

`uv venv` creates `.venv/`. `uv sync` installs `playwright`, `pillow` and `python-dotenv` from `pyproject.toml` / `uv.lock`.

Activate it if you want to run commands without `uv run`:

```powershell
.venv\Scripts\activate
```

```bash
source .venv/bin/activate
```

## 3. Configure

Copy the example settings file:

```bash
cp .env.example .env
```

On Windows PowerShell: `Copy-Item .env.example .env`.

## Configuration

All settings are optional; anything missing from `.env` falls back to the defaults listed in `.env.example`.

| Group | Variables |
| --- | --- |
| Browser | `BROWSER_PATH`, `GAME_URL`, `VIEWPORT_WIDTH`, `VIEWPORT_HEIGHT`, `CANVAS_TIMEOUT_MS`, `BLOCK_ADS` |
| Pacing (how human the aiming looks) | `AIM_SPEED_PX_S`, `AIM_MIN_S`, `AIM_MAX_S`, `SHOT_WAIT_S`, `WAYPOINT_COUNT_WEIGHTS` |
| Strategy weights | `DROP_WEIGHT`, `DEPTH_WEIGHT`, `MISS_BASE`, `FRIEND_WEIGHT`, `GROWTH_PENALTY` |
| Run limits and output | `MAX_PLAY_MINUTES`, `MAX_SHOTS` (0 = unlimited), `SCREENSHOT_DIR` |

The bot does not log in to the site.

## 4. Run

```bash
uv run python main.py
```

A browser window opens and the bot starts shooting. Press `Ctrl+C` to stop.

The bot also stops by itself at game over, then prints shots fired, play time and the path of the final screenshot (saved in `screens/`, which shows the score).

## Inspect what the bot sees

Print the detected grid, shooter color and chosen shot for a saved canvas screenshot:

```bash
uv run python -m tools.inspect_frame path/to/screenshot.png
```

## Project layout

```
main.py                  entry point
bubbleshooter/
  config.py              settings from .env
  layout.py              pixel geometry of the game canvas
  vision.py              screenshot to shooter color and grid bubbles
  strategy.py            shot simulation and target choice
  bot.py                 jittered click
  browser.py             browser launch, screenshots, clicks, page recovery
  game.py                game loop, game-over detection, status report
tools/inspect_frame.py   debugging helper
```

## Notes

- The pixel coordinates in `layout.py` are calibrated to this game's 750x562 canvas.
- The page can crash after several minutes; the bot reopens it, which starts a new game.
- Playwright's bundled Chromium failed to start on the development machine, which is why the bot launches an installed browser via `BROWSER_PATH`.
