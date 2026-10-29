"""What the tables of ANY batch of the scenario S1 allow without fitting anything.

The development pool and the confirmatory (sealed) batch have the same phase tables (``nmi.scenario.protocol``). The development entry ``scripts/run_scenario.py`` reads the development pool only; the functions
here take a table and know nothing of the pool, so the same code serves both. Nothing here fits a model, reads a label, a question or a generated text.

* ``missing_flags``: the boolean ``missing`` of every row of a phase table, checked against ``status``. The expert and answer tables have one row per registered item, the decide table one row per registered
  item AND condition.
* ``quality_gate``: the QUALITY GATE of a batch, for the confirmatory batch of a model: the digit quality, over ALL REGISTERED items (an item without a usable digit
  distribution does not pass), must be at least 99 %, and the answer-phase missing share at most 1 %. It is not the CALIBRATION GATE (``nmi.scenario.calibration.calibrate``: the three gates of the development
  pool, with the fits). The confirmatory batch is not refitted and its labels are not used.
* ``frozen_fit`` and ``batch_z``: the only way a confirmatory batch gets ``z``: the final full-pool fit of the development summary, applied to the batch's confidences. A batch whose quality gate fails, or
  whose development calibration gate was not passed, has an INVALID z (it blocks the claims that depend on z; the claim about the highest cost level does not).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyarrow as pa

from nmi.scenario import calibration
from nmi.scenario.protocol import MISSING, OK

MISSING_MAX = 0.01  # more than 1 % missing: z is invalid (answer phase) or the condition is untestable (decide phase)
ANSWER_MISSING_MAX = MISSING_MAX


# ---------------------------------------------------------------------------------------
# the missing flags
# ---------------------------------------------------------------------------------------

def missing_flags(table: pa.Table) -> np.ndarray:
    """The per-row missing flag of a phase table as a boolean array (True: the row has no result).

    The table's ``status`` must be ``ok`` or ``missing`` in every row. The boolean column ``missing`` must be there with no null and agree with ``status`` (``ValueError`` otherwise); a table written before
    the column existed (older runs) has none, and the flag then follows from ``status``."""
    names = table.schema.names
    if "status" not in names:
        raise ValueError("the table has no status column")
    status = table.column("status").to_pylist()
    bad = sorted({repr(s) for s in status if s not in (OK, MISSING)})
    if bad:
        raise ValueError(f"status values {bad} are neither {OK!r} nor {MISSING!r}")
    from_status = np.array([s == MISSING for s in status], dtype=bool)
    if "missing" not in names:
        return from_status
    raw = table.column("missing").to_pylist()
    if any(v is None for v in raw):
        raise ValueError("the missing flag is null in some rows")
    flags = np.array(raw, dtype=bool)
    if not np.array_equal(flags, from_status):
        raise ValueError("the missing flag and status disagree in some rows")
    return flags


# ---------------------------------------------------------------------------------------
# the quality gate of a batch
# ---------------------------------------------------------------------------------------

def answer_missing(n_missing: int, n_registered: int) -> dict[str, Any]:
    """The missing share of the answer phase must be at most 1 % of the registered items."""
    rate = n_missing / n_registered if n_registered else None
    return {"value": rate, "n_missing": n_missing, "n_registered": n_registered, "threshold": ANSWER_MISSING_MAX, "pass": bool(rate is not None and rate <= ANSWER_MISSING_MAX)}


def quality_gate(table: pa.Table, *, registered_keys: Sequence[str] | None = None) -> dict[str, Any]:
    """The QUALITY GATE of one model's answer-phase table, for any batch (a confirmatory batch in particular; nothing is refitted, no label is read):

    * ``digit_quality``: the share of all registered items whose digit mass is at least 0.9 is at least 99 %; an item that is missing or has no usable digit distribution does not pass and stays in the
      denominator (``nmi.scenario.calibration.digit_quality``, the function that is also gate 1 of the development pool);
    * ``answer_missing``: at most 1 % of the registered items have no answer-phase result.

    ``registered_keys`` are the keys of all registered items of the batch. Without them the rows of the table are the registered items; with them every row must belong to a registered item, and a registered
    item without any row counts as missing (a table that lost rows cannot raise its own pass rate). Plain data comes back; ``pass`` needs both checks."""
    keys = table.column("item_key").to_pylist()
    if any(k is None for k in keys):
        raise ValueError("a row has no item key")
    if len(set(keys)) != len(keys):
        raise ValueError("an item key is listed twice")
    flags = missing_flags(table)
    n_rows = len(keys)
    if registered_keys is None:
        n_registered, n_absent = n_rows, 0
    else:
        registered = list(registered_keys)
        if len(set(registered)) != len(registered):
            raise ValueError("a registered item key is listed twice")
        unknown = set(keys) - set(registered)
        if unknown:
            raise ValueError(f"{len(unknown)} row(s) belong to items that are not registered")
        n_registered, n_absent = len(registered), len(registered) - n_rows
    masses = [m for m, miss in zip(table.column("digit_mass").to_pylist(), flags, strict=True) if not miss]
    digit = calibration.digit_quality(masses, n_registered)
    missing = answer_missing(n_registered - len(masses), n_registered)
    return {
        "kind": "quality_gate",
        "scope": "any batch of answer-phase results (development or confirmatory): digit quality over all registered items and the answer-phase missing share; nothing is refitted and no label is read",
        "n_registered": n_registered, "n_rows": n_rows, "n_without_row": n_absent, "digit_quality": digit, "answer_missing": missing, "pass": bool(digit["pass"] and missing["pass"]), "refit": False,
    }


# ---------------------------------------------------------------------------------------
# z of a confirmatory batch from the frozen development fit
# ---------------------------------------------------------------------------------------

def frozen_fit(development: Mapping[str, Any]) -> calibration.Fit:
    """The final full-pool fit of a development answer summary (``answer_summary.json`` as a dict, or what ``nmi.scenario.summary.answer_summary`` returns) as a ``calibration.Fit``.

    ``ValueError`` when the summary has no calibration or no usable final fit (it did not converge, a parameter is not finite or b is negative)."""
    try:
        final = development["calibration"]["final"]
        a, b, n, ok = final["a"], final["b"], final["n"], final["ok"]
    except (KeyError, TypeError) as error:
        raise ValueError("not a development answer summary: it has no calibration.final") from error
    if not ok or a is None or b is None or not (math.isfinite(float(a)) and math.isfinite(float(b))) or float(b) < 0.0:
        raise ValueError("the development calibration has no usable final fit")
    return calibration.Fit(float(a), float(b), True, int(n), None)


@dataclass(frozen=True)
class BatchZ:
    """z of the rows of a batch's answer table from the frozen development fit, with its validity."""

    item_keys: tuple[str, ...]
    z: np.ndarray  # per row, NaN where the answer-phase result is missing or has no confidence
    valid: bool  # the development calibration gate is passed AND the quality gate of this batch is passed
    reasons: tuple[str, ...]  # why not, if not valid
    quality: dict[str, Any]
    fit: dict[str, Any] | None  # a, b, n of the frozen development fit (None if there was none)

    @property
    def n_with_z(self) -> int:
        return int(np.isfinite(self.z).sum())


def batch_z(development: Mapping[str, Any], table: pa.Table, *, registered_keys: Sequence[str] | None = None) -> BatchZ:
    """z for an answer table of a batch (a confirmatory batch in particular) from the FROZEN final fit of the development summary: ``sigmoid(a + b logit(c))``. Nothing is refitted, the correctness column is
    never read, and the parameters are those of the development pool whatever the batch looks like.

    ``valid`` is false, and ``reasons`` says why, if the development summary is not ``z_valid`` (the calibration gate), if there is no usable final fit, or if the quality gate of this batch is not passed (digit
    quality over all registered items below 99 %, or more than 1 % of the answers missing)."""
    quality = quality_gate(table, registered_keys=registered_keys)
    keys = tuple(table.column("item_key").to_pylist())
    flags = missing_flags(table)
    reasons: list[str] = []
    if development.get("z_valid") is not True:
        reasons.append("the calibration gate of the development pool is not passed (z_valid is not true)")
    if not quality["digit_quality"]["pass"]:
        reasons.append("quality gate: the digit quality is below 99 % of the registered items")
    if not quality["answer_missing"]["pass"]:
        reasons.append("quality gate: more than 1 % of the answer-phase results are missing")
    z = np.full(len(keys), np.nan)
    fit_info: dict[str, Any] | None = None
    try:
        fit = frozen_fit(development)
    except ValueError as error:
        fit = None
        reasons.append(str(error))
    if fit is not None:
        c = np.array([math.nan if (v is None or miss) else v for v, miss in zip(table.column("c").to_pylist(), flags, strict=True)], dtype=np.float64)
        have = np.isfinite(c)
        z[have] = fit.z(c[have])
        fit_info = {"a": fit.a, "b": fit.b, "n": fit.n, "source": "final fit on the development pool, frozen"}
    return BatchZ(item_keys=keys, z=z, valid=not reasons, reasons=tuple(reasons), quality=quality, fit=fit_info)


__all__ = [
    "ANSWER_MISSING_MAX", "BatchZ", "MISSING_MAX", "answer_missing", "batch_z", "frozen_fit", "missing_flags", "quality_gate",
]
