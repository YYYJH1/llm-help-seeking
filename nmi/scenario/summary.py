"""Summaries of the phase tables of the scenario S1: counts and rates only, no text.

* ``expert_summary``: the expert's accuracy (unrounded, primary score and the three sensitivity variants), the integer percent it suggests for the prompts (half up), missing share.
* ``answer_summary``: missing share, accuracies, the digit quality, and two gate blocks that are kept apart: ``calibration`` is the CALIBRATION GATE of the development pool (the three gates and the fits,
  ``nmi.scenario.calibration``), ``quality_gate`` the QUALITY GATE that any batch gets (digit quality over all registered items and the answer-phase missing share, ``nmi.scenario.analysis``; on the
  development pool it is a rehearsal of the check a confirmatory batch gets, and does not enter ``z_valid``). ``z`` is VALID only if the calibration is valid AND the share of missing answer-phase items is at
  most 1 %; otherwise ``z_valid`` is false and ``z_invalid_reasons`` says why. ``digit_gate_pass_rate`` is a description over the ANSWERED items only; the gate itself is judged over all
  registered items (``quality_gate.digit_quality`` and ``calibration.gates.digit_quality`` give the same numbers).
* ``decide_summary``: per condition, the denominators (registered items, missing, answered), the three outcomes (they sum to the answered items; a missing decision is none of them and in particular not a
  format failure), the format gate (format failure <= 1 %: ok; up to 5 %: deviation; above 5 %: untestable), the missing gate (a condition with more than 1 % missing is untestable), syntaxes, the multi-call
  share, whether the generated question equals the original one, the native-writing reading, truncation, the accuracy of the self-answers, and content-free diagnostics (structure signatures, shapes).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

import numpy as np
import pyarrow as pa

from nmi.scenario import analysis
from nmi.scenario.analysis import MISSING_MAX
from nmi.scenario.calibration import CalibrationResult

FORMAT_OK_MAX = 0.01
FORMAT_DEVIATION_MAX = 0.05
SCORE_COLUMNS = ("correct", "correct_full_alias", "correct_negation", "correct_short_boundary")
TOP = 8


def a_pct_of(accuracy: float) -> int:
    """The integer percent that a measured accuracy is stated as in the prompts: rounded half up on its decimal form, kept within 1 to 99."""
    pct = int((Decimal(repr(float(accuracy))) * 100).quantize(Decimal(1), rounding=ROUND_HALF_UP))
    return min(99, max(1, pct))


def _col(table: pa.Table, name: str) -> list[Any]:
    return table.column(name).to_pylist()


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if len(values) else None


def _gen_stats(table: pa.Table, ok: np.ndarray) -> dict[str, Any]:
    tokens = np.array([t for t, k in zip(_col(table, "gen_tokens"), ok, strict=True) if k and t is not None], dtype=np.float64)
    truncated = [t for t, k in zip(_col(table, "truncated"), ok, strict=True) if k and t is not None]
    empty = [t for t, k in zip(_col(table, "empty"), ok, strict=True) if k and t is not None]
    service = np.array([t for t, k in zip(_col(table, "service_s"), ok, strict=True) if k and t is not None], dtype=np.float64)
    return {
        "gen_tokens_mean": _mean(tokens), "gen_tokens_median": float(np.median(tokens)) if tokens.size else None, "gen_tokens_p95": float(np.percentile(tokens, 95)) if tokens.size else None,
        "truncated_rate": _rate(sum(truncated), len(truncated)), "empty_rate": _rate(sum(empty), len(empty)), "service_s_mean": _mean(service), "n_service_s": int(service.size),
    }


def _status(table: pa.Table) -> tuple[np.ndarray, int, int, int]:
    """(ok mask, rows, rows that are ok, rows that are missing); the boolean ``missing`` and ``status`` must agree (``analysis.missing_flags``)."""
    missing = analysis.missing_flags(table)
    n_missing = int(missing.sum())
    return ~missing, int(missing.size), int(missing.size) - n_missing, n_missing


def expert_summary(table: pa.Table) -> dict[str, Any]:
    ok, n, n_ok, n_missing = _status(table)
    out: dict[str, Any] = {"n_registered": n, "n_ok": n_ok, "n_missing": n_missing, "missing_rate": _rate(n_missing, n)}
    for name in SCORE_COLUMNS:
        values = [v for v, k in zip(_col(table, name), ok, strict=True) if k]
        out["accuracy" if name == "correct" else f"accuracy_{name[len('correct_'):]}"] = _mean([float(v) for v in values])
    acc = out["accuracy"]
    out["a_pct_suggested"] = a_pct_of(acc) if acc is not None else None
    out.update(_gen_stats(table, ok))
    return out


def answer_summary(table: pa.Table, calibration: CalibrationResult | None, *, calibration_error: str | None = None) -> dict[str, Any]:
    ok, n, n_ok, n_missing = _status(table)
    out: dict[str, Any] = {"n_registered": n, "n_ok": n_ok, "n_missing": n_missing, "missing_rate": _rate(n_missing, n)}
    for name in SCORE_COLUMNS:
        values = [v for v, k in zip(_col(table, name), ok, strict=True) if k]
        out["accuracy" if name == "correct" else f"accuracy_{name[len('correct_'):]}"] = _mean([float(v) for v in values])
    mass = np.array([m for m, k in zip(_col(table, "digit_mass"), ok, strict=True) if k and m is not None], dtype=np.float64)
    gate = [g for g, k in zip(_col(table, "digit_gate"), ok, strict=True) if k and g is not None]
    c = np.array([x for x, k in zip(_col(table, "c"), ok, strict=True) if k and x is not None], dtype=np.float64)
    out.update({"digit_mass_mean": _mean(mass), "digit_gate_pass_rate": _rate(sum(gate), len(gate)), "c_mean": _mean(c), "n_without_c": int(n_ok - c.size)})
    out.update(_gen_stats(table, ok))
    reasons: list[str] = []
    if n and n_missing / n > MISSING_MAX:
        reasons.append(f"missing share {n_missing / n:.4f} > {MISSING_MAX}")
    if calibration is None:
        reasons.append("calibration not run" if calibration_error is None else f"calibration raised {calibration_error}")
    else:
        out["calibration"] = calibration.as_dict()
        if not calibration.z_valid:
            reasons.append("calibration gates or fits failed")
    out["calibration_error"] = calibration_error
    out["quality_gate"] = analysis.quality_gate(table)
    out["z_invalid_reasons"] = reasons
    out["z_valid"] = not reasons
    return out


def format_gate(rate: float | None) -> str | None:
    """ok (<= 1 %), deviation (up to 5 %) or untestable (above 5 %)."""
    if rate is None:
        return None
    return "ok" if rate <= FORMAT_OK_MAX else "deviation" if rate <= FORMAT_DEVIATION_MAX else "untestable"


def decide_summary(table: pa.Table, conditions: Sequence[str]) -> dict[str, Any]:
    """Per condition of the table (and for the whole table) the counts and rates described above; ``conditions`` are the condition labels, in order."""
    cond_col = _col(table, "condition")
    out: dict[str, Any] = {"conditions": {}, "n_rows": len(cond_col)}
    for label in conditions:
        mask = np.array([c == label for c in cond_col])
        sub = table.filter(pa.array(mask))
        ok, n, n_ok, n_missing = _status(sub)
        outcome = Counter(o for o, k in zip(_col(sub, "outcome"), ok, strict=True) if k)
        if sum(outcome.values()) != n_ok:
            raise ValueError(f"{label}: the outcomes do not sum to the answered items")
        fail_rate = _rate(outcome["format_failure"], n_ok)
        offload_rows = [k for k, o in enumerate(_col(sub, "outcome")) if o == "offload"]
        n_off = len(offload_rows)

        def sel(name: str, rows: Sequence[int] = offload_rows, sub: pa.Table = sub) -> list[Any]:  # the table of this condition is bound at definition
            values = _col(sub, name)
            return [values[k] for k in rows]

        self_rows = [k for k, o in enumerate(_col(sub, "outcome")) if o == "self_answer"]
        missing_rate = _rate(n_missing, n)
        entry: dict[str, Any] = {
            "n_registered": n, "n_missing": n_missing, "missing_rate": missing_rate, "n_ok": n_ok, "missing_untestable": bool(missing_rate is not None and missing_rate > MISSING_MAX),
            "outcomes": {k: outcome.get(k, 0) for k in ("offload", "self_answer", "format_failure")},
            "rates": {k: _rate(outcome.get(k, 0), n_ok) for k in ("offload", "self_answer", "format_failure")},
            "format_failure_rate": fail_rate, "format_gate": format_gate(fail_rate),
            "reply_classes": dict(sorted(Counter(c for c, k in zip(_col(sub, "reply_class"), ok, strict=True) if k).items())),
            "syntax_counts": dict(sorted(Counter(sel("syntax")).items())),
            "multi_call_rate": _rate(sum(bool(v) for v in sel("multi_call")), n_off),
            "arg_equals_question_rate": _rate(sum(bool(v) for v in sel("arg_equals_question")), n_off),
            "first_valid_tagged_rate": _rate(sum(bool(v) for v in sel("first_valid_tagged")), n_off),
            "executed_native_rate": _rate(sum(bool(v) for v in sel("executed_native")), n_off),
            "text_before_rate": _rate(sum(bool(v) for v in sel("text_before")), n_off), "text_after_rate": _rate(sum(bool(v) for v in sel("text_after")), n_off),
            "native_offload_rate": _rate(sum(bool(v) for v, k in zip(_col(sub, "native_offload"), ok, strict=True) if k), n_ok),
            "self_answer_accuracy": _mean([float(v) for v in sel("self_correct", self_rows)]),
            "structure_top": Counter(s for s, k in zip(_col(sub, "structure"), ok, strict=True) if k).most_common(TOP),
            "shape_top": Counter(s for s in _col(sub, "shape") if s is not None).most_common(TOP),
        }
        entry.update(_gen_stats(sub, ok))
        entry["untestable"] = bool(entry["missing_untestable"] or entry["format_gate"] == "untestable")
        entry["untestable_reasons"] = [r for r, bad in (("missing share above 1 %", entry["missing_untestable"]), ("format failure above 5 %", entry["format_gate"] == "untestable")) if bad]
        out["conditions"][label] = entry
    out["n_untestable"] = sum(bool(v["untestable"]) for v in out["conditions"].values())
    out["n_missing"] = sum(v["n_missing"] for v in out["conditions"].values())
    out["n_registered"] = sum(v["n_registered"] for v in out["conditions"].values())
    out["missing_rate"] = _rate(out["n_missing"], out["n_registered"])
    return out


def jsonable(obj: Any) -> Any:
    """NaN and infinities as None, tuples as lists (the summaries are written as strict JSON)."""
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj
