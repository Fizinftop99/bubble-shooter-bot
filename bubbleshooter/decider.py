import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from . import layout, strategy
from .vision import colors_match

COLOUR_LETTERS = "RGBYMCOPQ"
MAX_ANSWER_TOKENS = 24
CANDIDATES_SHOWN = 4  # must equal the teacher's top_k used for the training data (dataset.DATASET_CONFIG)
REQUEST_WORKERS = 4
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"row": {"type": "integer"}, "col": {"type": "integer"}},
    "required": ["row", "col"],
    "additionalProperties": False,
}
SYSTEM_PROMPT = (
    "You choose shots in a Bubble Shooter game. The board is a hex grid: row 0 is the top, "
    "rows marked 'shifted' sit half a cell to the right. '.' is an empty cell.\n"
    "Rules: landing next to same-colour bubbles pops the group if it reaches 3 or more. "
    "Bubbles left hanging without a path to row 0 drop, and each dropped bubble is worth far more than a popped one. "
    "A shot that does not pop costs a life; with no lives left, new rows are pushed in from the top. "
    "A bubble reaching row 15 loses the game. "
    "Removing the last bubble of a colour makes every later push bigger, so avoid it unless the board is nearly clear. "
    "The next colour is the bubble you will shoot after this one.\n"
    "You are given a few promising cells the shot can reach and what happens there. Pick exactly one of those cells. "
    'Answer with JSON only: {"row": <row>, "col": <col>}.'
)


@dataclass(frozen=True)
class Decision:
    x: int
    y: int
    reason: str
    source: str = "rule"
    latency_ms: float = 0.0
    fallback: str | None = None
    agrees_with_rule: bool | None = None


class Decider(Protocol):
    name: str

    def decide(self, shooter_color, bubbles, next_color=None, lives=None) -> Decision | None: ...


class RuleDecider:
    name = "rule"

    def __init__(self, weights):
        self._weights = weights

    def decide(self, shooter_color, bubbles, next_color=None, lives=None):
        started = time.perf_counter()
        result = strategy.choose_target(shooter_color, bubbles, weights=self._weights)
        if result is None:
            return None
        return Decision(*result, latency_ms=(time.perf_counter() - started) * 1000)


def cell_address(cell):
    x, y, row = cell
    shifted = (x - layout.COL0_X) % layout.COL_PITCH != 0
    col = (x - layout.COL0_X - (layout.ODD_ROW_SHIFT if shifted else 0)) // layout.COL_PITCH
    return row, col


def _palette(shooter_color, bubbles):
    reps = []

    def letter(color):
        for i, rep in enumerate(reps):
            if colors_match(rep, color):
                return COLOUR_LETTERS[i % len(COLOUR_LETTERS)]
        reps.append(color)
        return COLOUR_LETTERS[(len(reps) - 1) % len(COLOUR_LETTERS)]

    # Letters follow the first appearance in reading order, so the same board gets the same letters
    # whatever order the caller lists the bubbles in (live vision, simulator, or a dataset record).
    for b in sorted(bubbles, key=lambda b: (b["row"], b["x"])):
        letter(b["color"])
    return letter


def build_prompt(shooter_color, bubbles, candidates, next_color=None, lives=None):
    """`next_color` and `lives` get their own lines when known; None leaves the line out."""
    letter = _palette(shooter_color, bubbles)
    rows, counts = {}, {}
    for b in bubbles:
        row, col = cell_address((b["x"], b["y"], b["row"]))
        line = rows.setdefault(row, {"shifted": b["shifted"], "cells": ["."] * layout.COLS})
        if 0 <= col < layout.COLS:
            line["cells"][col] = letter(b["color"])
        counts[letter(b["color"])] = counts.get(letter(b["color"]), 0) + 1
    lines = ["Board (columns 0-16 left to right):"]
    for row in range(max(rows) + 1):
        line = rows.get(row)
        cells = "".join(line["cells"]) if line else "." * layout.COLS
        lines.append(f"row {row:2d}: {cells}{' shifted' if line and line['shifted'] else ''}")
    lines.append("Bubbles per colour: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    lines.append(f"Lowest occupied row: {max(rows)} (row 15 loses)")
    lines.append(f"Shooter colour: {letter(shooter_color)}")
    if next_color is not None:
        lines.append(f"Next colour: {letter(next_color)}")
    if lives is not None:
        lines.append(f"Lives left: {lives}" + (" (a shot that does not pop pushes new rows)" if lives == 0 else ""))
    lines.append("Reachable cells:")
    for candidate in sorted(candidates, key=lambda c: cell_address(c.cell)):
        row, col = cell_address(candidate.cell)
        wipe = f", removes the last bubble of {candidate.wiped} colour(s)" if candidate.wiped else ""
        lines.append(f"- row {row} col {col}: {candidate.why}{wipe}")
    return "\n".join(lines)


def parse_answer(text, by_address):
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return None, "invalid json"
    if not isinstance(data, dict):
        return None, "invalid json"
    row, col = data.get("row"), data.get("col")
    if type(row) is not int or type(col) is not int:
        return None, "invalid fields"
    candidate = by_address.get((row, col))
    if candidate is None:
        return None, "unreachable cell"
    return candidate, None


class ChatClient:
    def __init__(self, base_url, model):
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._model = model

    def complete(self, system, user, timeout_s):
        body = {
            "model": self._model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
            "max_tokens": MAX_ANSWER_TOKENS,
            "stream": False,
            "response_format": {"type": "json_schema", "json_schema": {"name": "shot", "strict": True, "schema": ANSWER_SCHEMA}},
        }
        request = urllib.request.Request(self._url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            reply = json.load(response)
        return reply["choices"][0]["message"]["content"]


class DecisionLog:
    def __init__(self, directory, label="game"):
        folder = Path(directory)
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / f"decisions_{label}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"

    def write(self, decision, bubble_count):
        record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "bubbles": bubble_count, **asdict(decision)}
        record["latency_ms"] = round(decision.latency_ms, 1)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")


class ModelDecider:
    name = "model"

    def __init__(self, weights, settings, client=None, log=None, shown=CANDIDATES_SHOWN):
        self._weights = weights
        self._shown = shown
        self._timeout_s = settings.model_timeout_s
        self._warmup_s = settings.model_warmup_s
        self._client = client or ChatClient(settings.model_url, settings.model_name)
        self._rule = RuleDecider(weights)
        self._pool = ThreadPoolExecutor(max_workers=REQUEST_WORKERS, thread_name_prefix="decider")
        self._log = log
        self.latencies_ms = []
        self.fallbacks = {}
        self.model_shots = 0
        self.disagreements = 0

    def warm_up(self):
        try:
            self._client.complete(SYSTEM_PROMPT, "Reachable cells:\n- row 0 col 0: pop 3 drop 0", self._warmup_s)
            return None
        except Exception as error:
            return str(error).splitlines()[0] if str(error) else type(error).__name__

    def decide(self, shooter_color, bubbles, next_color=None, lives=None):
        candidates = strategy.list_candidates(shooter_color, bubbles, weights=self._weights)
        if len(candidates) < 2:
            return self._record(self._rule.decide(shooter_color, bubbles), bubbles)
        # Same shortlist the rollout teacher labelled: the best few cells by the rule's score (stable sort, so the
        # rule's own pick stays first on ties). Cells outside it are not offered, so the model cannot pick them.
        shortlist = sorted(candidates, key=lambda c: -c.score)[: self._shown]
        started = time.perf_counter()
        candidate, problem = self._ask(build_prompt(shooter_color, bubbles, shortlist, next_color, lives), shortlist)
        latency_ms = (time.perf_counter() - started) * 1000
        self.latencies_ms.append(latency_ms)
        best = strategy.best_candidate(candidates)
        if candidate is None:
            self.fallbacks[problem] = self.fallbacks.get(problem, 0) + 1
            decision = Decision(*best.aim(), source="rule", latency_ms=latency_ms, fallback=problem)
        else:
            self.model_shots += 1
            agrees = candidate.cell == best.cell
            self.disagreements += not agrees
            decision = Decision(*candidate.aim(), source="model", latency_ms=latency_ms, agrees_with_rule=agrees)
        return self._record(decision, bubbles)

    def _ask(self, prompt, candidates):
        by_address = {cell_address(c.cell): c for c in candidates}
        future = self._pool.submit(self._client.complete, SYSTEM_PROMPT, prompt, self._timeout_s)
        try:
            text = future.result(timeout=self._timeout_s)
        except (FutureTimeout, TimeoutError):
            future.cancel()
            return None, "timeout"
        except Exception as error:
            if "timed out" in str(error).lower():
                return None, "timeout"
            status = getattr(error, "code", None)
            return None, f"request failed: {type(error).__name__}{f' {status}' if status else ''}"
        return parse_answer(text, by_address)

    def _record(self, decision, bubbles):
        if self._log and decision is not None:
            self._log.write(decision, len(bubbles))
        return decision

    def summary(self):
        asked = len(self.latencies_ms)
        fallback_total = sum(self.fallbacks.values())
        if not asked:
            return "model decider: no model calls"
        ordered = sorted(self.latencies_ms)
        p95 = ordered[min(asked - 1, int(asked * 0.95))]
        reasons = ", ".join(f"{k}: {v}" for k, v in sorted(self.fallbacks.items())) or "none"
        return (
            f"model decider: {asked} calls, {self.model_shots} used, "
            f"fallback rate {fallback_total / asked:.1%} ({reasons}), "
            f"latency mean {statistics.mean(ordered):.0f} ms / p95 {p95:.0f} ms, "
            f"differs from rule in {self.disagreements} of {self.model_shots}"
        )


def make_decider(settings, weights=None, with_log=True):
    weights = weights or settings.weights
    cfg = settings.decider
    if cfg.decider == "rule":
        return RuleDecider(weights)
    if cfg.decider != "model":
        raise ValueError(f"unknown decider {cfg.decider!r}; use 'rule' or 'model'")
    log = DecisionLog(cfg.decision_log_dir) if with_log and cfg.decision_log_dir else None
    return ModelDecider(weights, cfg, log=log)
