"""The grid and the quality gate of the C1 confirmation run.

``scripts/run_c1_confirm.py`` reads the 54-cell psychometric grid (a_H x c0 x w) of the P-ask prompt, paraphrase ``a00``, framing F2, standard layout, and the P-answer prompt once per item.
This module holds the constants of that grid (``CELLS`` in the order of ``run_decisions.grid_conditions``), the key and label of a cell, and the quality gate computed from counts: in P-answer and in
each of the 54 cells of P-ask, the share of registered trials whose digit mass is at least the gate (``gate_pass``) is at least 0.99. The denominator is every EXPECTED trial (one per item, and one
per item and cell), so a missing row counts against the share. A trial with ``gate_pass`` False is a valid readout: it stays in the analysis, its ask decision is ``margin > 0`` as the readout
gave it, and it is not a mask for dropping items. It reads no question, no label text and no generated text.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any

# ---------------------------------------------------------------------------------------
# fixed settings
# ---------------------------------------------------------------------------------------

GATE_MIN_RATE = 0.99
PARAPHRASE = "a00"
FRAMING = "f2"
LAYOUT = "standard"
GRID_TAG = "psychometric"

A_H_LEVELS = (0.60, 0.80, 0.95)
C0_LEVELS = (0.0, 0.1, 0.2)
W_LEVELS = (0.0, 0.1, 0.2, 0.45, 0.8, 1.5)
CELLS: tuple[tuple[float, float, float], ...] = tuple(itertools.product(A_H_LEVELS, C0_LEVELS, W_LEVELS))  # the 54 conditions, in the order of ``run_decisions.grid_conditions``


def cell_key(a_h: float, c0: float, w: float) -> tuple[float, float, float]:
    return (round(float(a_h), 9), round(float(c0), 9), round(float(w), 9))


CELL_INDEX: dict[tuple[float, float, float], int] = {cell_key(*cell): k for k, cell in enumerate(CELLS)}


def cell_label(cell: tuple[float, float, float]) -> str:
    return f"a_H={cell[0]:g},c0={cell[1]:g},w={cell[2]:g}"


# ---------------------------------------------------------------------------------------
# the quality gate
# ---------------------------------------------------------------------------------------

def quality_gate_report(
    *, n_registered: int, answer_rows: int, answer_pass: int, ask_rows: Sequence[int], ask_pass: Sequence[int], threshold: float = GATE_MIN_RATE,
    cells: Sequence[tuple[float, float, float]] = CELLS,
) -> dict[str, Any]:
    """The gate from counts: P-answer and each of the 54 cells need ``n_pass / n_registered >= threshold``, where ``n_registered`` is the number of expected trials (one per item), so a row that is not there counts as a failure."""
    if len(ask_rows) != len(cells) or len(ask_pass) != len(cells):
        raise ValueError("one count of rows and one of passes are needed for every cell")
    if n_registered < 1:
        raise ValueError("there is no registered trial")
    answer_rate = answer_pass / n_registered
    per_cell = []
    for cell, rows, passed in zip(cells, ask_rows, ask_pass, strict=True):
        rate = passed / n_registered
        per_cell.append({"a_H": cell[0], "c0": cell[1], "w": cell[2], "n_registered": int(n_registered), "n_rows": int(rows), "n_pass": int(passed), "rate": rate, "passes": bool(rate >= threshold)})
    worst = min(range(len(per_cell)), key=lambda k: per_cell[k]["rate"])
    return {
        "threshold": threshold,
        "denominator": "every registered trial: one P-answer trial per item and one P-ask trial per item and cell",
        "p_answer": {"n_registered": int(n_registered), "n_rows": int(answer_rows), "n_pass": int(answer_pass), "rate": answer_rate, "passes": bool(answer_rate >= threshold)},
        "p_ask": {
            "cells": per_cell, "min_rate": per_cell[worst]["rate"], "min_cell": cell_label(tuple(cells[worst])), "n_cells_failing": int(sum(not c["passes"] for c in per_cell)),
            "passes": bool(all(c["passes"] for c in per_cell)),
        },
        "passed": bool(answer_rate >= threshold and all(c["passes"] for c in per_cell)),
    }
