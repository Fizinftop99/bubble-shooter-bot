"""Rebuild the model prompt from a dataset record.

Goes through decider.build_prompt, so training prompts and live prompts share one code path.
"""
from types import SimpleNamespace

from .. import layout
from ..decider import SYSTEM_PROMPT, build_prompt
from .engine import PALETTE


def cell_pixel(row, col):
    shift = layout.ODD_ROW_SHIFT if row % 2 == 1 else 0
    return layout.COL0_X + col * layout.COL_PITCH + shift, layout.GRID_ROW0_Y + row * layout.ROW_PITCH


def prompt_from_record(record):
    bubbles = []
    for row, col, colour in record["board"]:
        x, y = cell_pixel(row, col)
        bubbles.append({"x": x, "y": y, "color": PALETTE[colour], "row": row, "shifted": row % 2 == 1})
    candidates = []
    for c in record["candidates"]:
        x, y = cell_pixel(c["row"], c["col"])
        candidates.append(SimpleNamespace(cell=(x, y, c["row"]), why=c["why"], wiped=c["wiped"]))
    return build_prompt(PALETTE[record["shooter"]], bubbles, candidates)


def answer_from_record(record, index=None):
    """Target JSON text for the candidate at `index` (default: the teacher's pick)."""
    c = record["candidates"][record["chosen"] if index is None else index]
    return f'{{"row": {c["row"]}, "col": {c["col"]}}}'


__all__ = ["SYSTEM_PROMPT", "prompt_from_record", "answer_from_record", "cell_pixel"]
