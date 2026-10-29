"""The summary and the checks of a confirmatory (sealed) batch of the scenario S1.

A sealed batch has the phase tables of the development pool (``nmi.scenario.protocol``) and a different life cycle: nothing in it is fitted. "Registered" items are the item keys listed for the batch (``registered_keys``). This module provides the following:

* ``answer_summary``: the summary of the answer phase of a batch. ``z`` is the FINAL full-pool fit of the development answer run applied to the batch's raw confidences (``nmi.scenario.analysis.batch_z``); the
  calibration (``nmi.scenario.calibration.calibrate``) is never called, so there are no folds, no AUROC and no calibration gate here (those are the development pool's, and ``development`` repeats them from the
  development summary). Only the QUALITY GATE of the batch is judged: the digit quality (digit mass at least 0.9 for at least 99 % of ALL registered items; an item without a result or without a digit
  distribution does not pass and stays in the denominator) and the answer-phase missing share (at most 1 %). A batch that fails it, or whose development calibration gate was not passed, has an INVALID ``z``.
* ``expert_summary``: the expert summary of the batch. It suggests no percent: the ``a_pct`` that the decide prompts state is the development pool's and does not follow the expert's result on the sealed pool.
* ``complete_table``: a phase table against the REGISTERED item keys. An item (or an item under a condition) that the table lacks is expressed as a missing row, never dropped; a row of an item that is not
  registered, or a key listed twice, is an error. Without this, a table that lost the same items under every condition would pass the pairing checks on a shrunken registered set.
* ``development_gates``: the gate values the report keeps next to the batch's own, so that it never shows a bare ``valid``.

Nothing here fits a model, reads a question, an answer or a generated text, or looks at the correctness column to make ``z``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import pyarrow as pa

from nmi.scenario import analysis, summary
from nmi.scenario.protocol import MISSING

ANSWER_KIND = "sealed_answer_summary"
Z_COLUMN = "z_frozen"  # the column of the sealed answer table that holds z from the frozen development fit (null where an item has none)
ABSENT_REASON = "absent_from_run"  # the ``missing_reason`` of a row that ``complete_table`` adds
A_PCT_NOTE = (
    "the percent that the decide prompts state is the expert's accuracy on the DEVELOPMENT pool, frozen in the run configuration; the expert's result on the sealed pool does not change it, "
    "so no percent is suggested here"
)


# ---------------------------------------------------------------------------------------
# the registered items
# ---------------------------------------------------------------------------------------

def _missing_row(schema: pa.Schema, key: str, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {name: None for name in schema.names}
    row.update(item_key=key, status=MISSING, missing=True, missing_reason=ABSENT_REASON)
    row.update(extra or {})
    return row


def complete_table(
    table: pa.Table, schema: pa.Schema, registered_keys: Sequence[str], *, conditions: Sequence[Any] | None = None,
) -> tuple[pa.Table, dict[str, Any]]:
    """The table of a phase checked against the registered item keys and completed with missing rows.

    ``conditions`` is None for the expert and answer tables (one row per registered item) and the conditions of the design for the decide table (one row per registered item AND condition; each condition needs
    ``label``, ``group``, ``level`` and ``points``, and a condition that the table lacks altogether is expressed as a missing row for every registered item). A registered item (under a condition) without a
    row becomes a row with ``status = missing``, ``missing = True`` and ``missing_reason = absent_from_run``. The rows come back in the order of the registered keys (for decide: condition by condition in
    the order given), the rows that were there with their values unchanged.

    ``ValueError`` for a column that the schema needs and the table lacks, a row without a key, a key that is not registered (a row of another pool), a key listed twice (under a condition), a registered key
    listed twice, or a condition label that is not one of ``conditions``. The report counts what was added (``n_rows_added``, and ``added_by_condition`` for decide)."""
    keys = list(registered_keys)
    if len(set(keys)) != len(keys):
        raise ValueError("a registered item key is listed twice")
    index = {k: i for i, k in enumerate(keys)}
    lacking = [name for name in schema.names if name not in table.schema.names]
    if lacking:
        raise ValueError(f"the table has no column {lacking}")
    cols = {name: table.column(name).to_pylist() for name in schema.names}
    n_rows = table.num_rows
    cond_list = list(conditions) if conditions is not None else None
    if cond_list is not None and "condition" not in cols:
        raise ValueError("the decide table has no condition column")
    cond_index = {c.label: j for j, c in enumerate(cond_list)} if cond_list is not None else {None: 0}
    present: set[tuple[Any, str]] = set()
    sort_keys: list[tuple[int, int]] = []
    rows: list[dict[str, Any]] = []
    for r in range(n_rows):
        key = cols["item_key"][r]
        if key is None:
            raise ValueError("a row has no item key")
        if key not in index:
            raise ValueError("a row belongs to an item that is not registered (a row of another pool?)")
        label = cols["condition"][r] if cond_list is not None else None
        if label not in cond_index:
            raise ValueError(f"unknown condition {label!r}; the design has {[c.label for c in cond_list or []]}")
        if (label, key) in present:
            raise ValueError(f"an item key is listed twice{'' if label is None else f' under {label}'}")
        present.add((label, key))
        rows.append({name: cols[name][r] for name in schema.names})
        sort_keys.append((cond_index[label], index[key]))
    added: dict[str, int] = {}
    for cond in cond_list if cond_list is not None else [None]:
        label = None if cond is None else cond.label
        extra = None if cond is None else {
            "condition": cond.label, "group": cond.group, "level": int(cond.level), "delay_s": None if cond.group == "S-loc" else int(cond.level), "points": float(cond.points),
        }
        n_added = 0
        for key in keys:
            if (label, key) not in present:
                rows.append(_missing_row(schema, key, extra))
                sort_keys.append((cond_index[label], index[key]))
                n_added += 1
        if label is not None:
            added[label] = n_added
    order = sorted(range(len(rows)), key=lambda i: sort_keys[i])
    ordered = [rows[i] for i in order]
    completed = pa.table({f.name: pa.array([row[f.name] for row in ordered], type=f.type) for f in schema}, schema=schema)
    n_added_total = len(rows) - n_rows
    report: dict[str, Any] = {"n_registered": len(keys), "n_rows_in": n_rows, "n_rows_out": completed.num_rows, "n_rows_added": n_added_total}
    if cond_list is not None:
        report["added_by_condition"] = {label: n for label, n in added.items() if n}
    return completed, report


# ---------------------------------------------------------------------------------------
# the development side of the frozen fit
# ---------------------------------------------------------------------------------------

def development_gates(development: Mapping[str, Any]) -> dict[str, Any]:
    """The three gates of the development pool and the frozen fit, as a development answer summary (``answer_summary.json``) or a recomputed ``{"z_valid", "calibration"}`` states them. Missing parts are None;
    nothing is judged here."""
    cal = development.get("calibration") if isinstance(development.get("calibration"), Mapping) else {}
    gates = cal.get("gates") or {}
    dq, au, ca = gates.get("digit_quality") or {}, gates.get("auroc") or {}, gates.get("calibration") or {}
    return {
        "z_valid": development.get("z_valid"), "z_invalid_reasons": list(development.get("z_invalid_reasons") or []), "answer_accuracy": cal.get("accuracy"), "answer_missing_rate": development.get("missing_rate"),
        "digit_quality": {"value": dq.get("value"), "threshold": dq.get("threshold"), "pass": dq.get("pass")},
        "auroc": {"value": au.get("value"), "lower": au.get("lower"), "upper": au.get("upper"), "threshold": au.get("threshold"), "pass": au.get("pass")},
        "calibration": {"logloss_oof": ca.get("logloss_oof"), "logloss_constant": ca.get("logloss_constant"), "ece_oof": ca.get("ece_oof"), "ece_max": ca.get("ece_max"), "pass": ca.get("pass")},
        "final_fit": cal.get("final"), "failures": list(cal.get("failures") or []),
    }


# ---------------------------------------------------------------------------------------
# the summaries of a sealed batch
# ---------------------------------------------------------------------------------------

def expert_summary(table: pa.Table, a_pct: int) -> dict[str, Any]:
    """``nmi.scenario.summary.expert_summary`` of the batch, without a suggested percent: ``a_pct_suggested`` is null and ``a_pct_frozen`` is the percent of the run configuration."""
    out = summary.expert_summary(table)
    out["a_pct_suggested"] = None
    out["a_pct_frozen"] = int(a_pct)
    out["a_pct_note"] = A_PCT_NOTE
    return out


def answer_summary(
    table: pa.Table, development: Mapping[str, Any], batch: analysis.BatchZ, *, frozen_source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The summary of the answer phase of a sealed batch. ``table`` is the answer table (one row per registered item), ``development`` the development answer summary whose final fit is applied, ``batch`` the
    result of ``nmi.scenario.analysis.batch_z`` on that table with the registered keys, ``frozen_source`` where the fit comes from (path and content hash of the development summary).

    The descriptive part (the missing share, the accuracies, the digit mass, the generation statistics) is that of ``nmi.scenario.summary.answer_summary``. ``z_valid`` is the batch's validity: the development
    calibration gate passed AND the quality gate of this batch passed. There is no ``calibration`` block, and ``refit`` is false."""
    out = summary.answer_summary(table, None)
    out.pop("calibration_error", None)
    out.update({
        "kind": ANSWER_KIND,
        "scope": ("confirmatory batch: z from the frozen final fit of the development answer run, nothing is refitted; the digit-quality gate (all registered items) and the answer-phase missing gate are "
                  "judged here, the AUROC and the calibration gates belong to the development pool and are repeated under development"),
        "z_valid": bool(batch.valid), "z_invalid_reasons": list(batch.reasons), "n_with_z": batch.n_with_z, "refit": False,
        "quality_gate": batch.quality, "frozen_fit": batch.fit, "development": development_gates(development), "frozen_source": None if frozen_source is None else dict(frozen_source),
    })
    return out


def add_z_column(table: pa.Table, batch: analysis.BatchZ) -> pa.Table:
    """The answer table with the column ``z_frozen`` appended: z of every row from the frozen fit, null where the row has none."""
    values = [float(v) if math.isfinite(float(v)) else None for v in batch.z]
    return table.append_column(Z_COLUMN, pa.array(values, type=pa.float64()))


# ---------------------------------------------------------------------------------------
# the check of the stored z
# ---------------------------------------------------------------------------------------

__all__ = [
    "ABSENT_REASON", "ANSWER_KIND", "A_PCT_NOTE", "Z_COLUMN", "add_z_column", "answer_summary", "complete_table", "development_gates", "expert_summary",
]
