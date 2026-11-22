"""Recompute teacher labels from the raw rollouts stored in schema-2 dataset records.

The values the teacher acted on are in each candidate's "teacher_value". Here the same numbers (or those of any
other outcome function) are rebuilt from "rollouts" without the simulator, and the spread of the rollouts gives
the standard error of every label.

    from bubbleshooter.sim import relabel
    cfg = relabel.config_for("noscore")          # or relabel.config_from_meta(meta)
    values = relabel.candidate_values(record, cfg)
"""
import json
import math
import statistics
from dataclasses import replace
from pathlib import Path

from .teacher import OUTCOMES, TeacherConfig, value_of

SCHEMA = 2


def config_for(outcome, base=TeacherConfig()):
    """TeacherConfig with the named outcome preset (teacher.OUTCOMES) applied."""
    return replace(base, **OUTCOMES[outcome])


def config_from_meta(meta):
    """The exact TeacherConfig a file was generated with, from its .meta.json (a dict or a path)."""
    if not isinstance(meta, dict):
        meta = json.loads(Path(meta).read_text(encoding="utf-8"))
    return TeacherConfig(**meta["teacher_config"])


def _check(record):
    if record.get("schema") != SCHEMA:
        raise ValueError(
            f"record has schema {record.get('schema')!r}, relabel needs schema {SCHEMA} (raw rollouts); "
            "records from older datasets only carry the averaged teacher_value"
        )


def rollout_values(record, cfg):
    """values[i][j]: candidate i, rollout j, scored with cfg."""
    _check(record)
    return [[value_of(tuple(run), cfg) for run in c["rollouts"]] for c in record["candidates"]]


def candidate_values(record, cfg):
    """Mean value per candidate. With the generating cfg this reproduces teacher_value exactly."""
    return [statistics.fmean(runs) for runs in rollout_values(record, cfg)]


def standard_errors(record, cfg):
    """Standard error of each candidate's mean value (0 when all its rollouts agree)."""
    return [statistics.stdev(runs) / math.sqrt(len(runs)) if len(runs) > 1 else math.inf for runs in rollout_values(record, cfg)]


def first_argmax(values):
    best = 0
    for i in range(1, len(values)):
        if values[i] > values[best]:
            best = i
    return best
