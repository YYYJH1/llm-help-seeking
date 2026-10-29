"""Templates, support, sampler and plans of the explicit-number confirmation test.

What this module holds:

* the support of (p, a_H, w) in exact integer hundredths, the dropped near-ties |Delta| < 0.05 and the three strata;
* the seeded sampler of the four condition sets: ``main`` (seed 20261019), ``pilot`` (20261020), ``calib`` (20261022) and the fixed ``grid``;
* the version expansion of a condition into 42 trials (R 12, C 6, A 12, B 12) and the assignment of the R distractors (seed 20261021);
* the English templates of the four arms R, C, A and B, the exact decimal Delta and the label of the better option;
* ``build_all``: everything above as canonical bytes (``conditions.json``, ``plans.json``), and ``load_plans``, which reads a plans directory back and
  proves it equals what this code builds now;
* the row schema of ``rows.jsonl`` (``ROW_FIELDS``) and the pure helpers that derive ``tie`` and ``choice`` from a margin.

It is pure Python plus numpy: no vLLM, no torch, no transformers, no tokenizer, so the analysis can import it. Nothing here reads a model output, a question or a
file outside the plans directory it is asked to load. All arithmetic on p, a_H, w and Delta is on integers (hundredths); strings are made only at the end
(``fmt``), always with two decimals and the ASCII hyphen-minus.

The templates are fixed. Changing a template, a seed or the sampler changes the bytes of ``plans.json``; ``load_plans`` then
refuses the old plans directory, so a stale directory cannot be read by accident. ``TEMPLATE_DATE`` is not part of those bytes: it is the day the runner pins for chat templates
that write the date into the system prompt (SmolLM3), and every run manifest records it.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_CONDITIONS = "nmi.explicit_numbers.conditions/1"
SCHEMA_PLANS = "nmi.explicit_numbers.plans/1"
CONDITIONS_FILE = "conditions.json"
CONDITIONS_SHA_FILE = "conditions.sha256"
PLANS_FILE = "plans.json"
PLANS_SHA_FILE = "plans.sha256"

# ---------------------------------------------------------------------------------------
# support, strata, seeds, sizes
# ---------------------------------------------------------------------------------------

FIELDS = ("p", "a_H", "w")
P_RANGE = (5, 95)  # hundredths: p in {0.05, ..., 0.95}
AH_RANGE = (50, 99)  # a_H in {0.50, ..., 0.99}
W_RANGE = (-20, 160)  # w in {-0.20, ..., 1.60}
FIELD_RANGE = {"p": P_RANGE, "a_H": AH_RANGE, "w": W_RANGE}
NEAR_TIE_H = 5  # |Delta| < 0.05 is dropped
W_STRATUM_CUT_H = 100  # stratum 3 is w > 1.00
EXPECTED_KEPT = 785_555
EXPECTED_STRATA = {1: 188_770, 2: 323_785, 3: 273_000}

SEEDS = {"main": 20261019, "pilot": 20261020, "distractor": 20261021, "calib": 20261022, "scan": 20261023}
K_MAIN = {1: 200, 2: 100, 3: 100}
K_PILOT = {1: 24, 2: 12, 3: 12}
K_CALIB = {1: 50, 2: 25, 3: 25}
GRID_P_H = (15, 35, 55, 75, 95)
GRID_AH_H = (60, 80, 95)
GRID_W_H = (-10, 0, 10, 30, 60, 150)

SETS = ("main", "pilot", "calib", "grid")  # the order in which the one distractor generator is consumed, and the order of the plans
ARMS = ("R", "C", "A", "B")
WORDINGS = ("neutral", "waiting")
LABEL_ORDERS = (0, 1)
REWRITES = (0, 1, 2)
N_CONDITIONS = {"main": 400, "pilot": 48, "calib": 100, "grid": 90}
TRIALS_PER_CONDITION = 42
N_TRIALS = {"main": 16_800, "pilot": 2_016, "calib": 100, "grid": 3_780}
NEAR_OFFSETS_H = (-10, -5, 5, 10)  # target -/+ 0.10, -/+ 0.05
ANSWER_INSTRUCTION = "Reply with 1 or 2 only."  # the last words of every prompt of every arm

# ---------------------------------------------------------------------------------------
# the row schema and the readout-derived fields
# ---------------------------------------------------------------------------------------

ROW_FIELDS = (
    "model", "set", "arm", "condition_id", "stratum", "p", "a_H", "w", "delta", "wording", "label_order", "rewrite", "field", "distractor_class", "distractor_value",
    "distractor_fallback", "correct_label", "help_label", "prompt_sha256", "n_tokens", "logprob_1", "logprob_2", "mass", "gate_pass", "margin", "tie", "choice", "free_token_id",
    "missing",
)
PLAN_ROW_FIELDS = (  # the row fields that are known before any reading
    "set", "arm", "condition_id", "stratum", "p", "a_H", "w", "delta", "wording", "label_order", "rewrite", "field", "distractor_class", "distractor_value", "distractor_fallback",
    "correct_label", "help_label",
)
READOUT_ROW_FIELDS = ("logprob_1", "logprob_2", "mass", "gate_pass", "margin", "tie", "choice", "free_token_id")  # null in a row whose readout is missing
TIE_EPS = 1e-6
DIGIT_MASS_GATE = 0.5


def choice_from_margin(margin: float) -> tuple[bool, str | None]:
    """``(tie, choice)`` of a margin m = logprob_1 - logprob_2: a tie when |m| < 1e-6 (no choice), else "1" for m > 0 and "2" for m < 0."""
    if not math.isfinite(margin):
        raise ValueError(f"the margin must be finite, got {margin!r}")
    if abs(margin) < TIE_EPS:
        return True, None
    return False, "1" if margin > 0 else "2"


def trial_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    """The identity of a trial, the same for a plan trial and a row of ``rows.jsonl`` (one model): set, arm, condition and the version fields."""
    return (record["set"], record["arm"], record["condition_id"], record["field"], record["wording"], record["label_order"], record["rewrite"])


def token_digest(ids: Sequence[int]) -> str:
    """sha256 of the token ids a prompt was sent as: decimal ids joined by commas, ASCII. The same rule as ``token_digest`` of ``scripts/run_c4_repeat.py``."""
    return hashlib.sha256(",".join(str(int(t)) for t in ids).encode("ascii")).hexdigest()


# ---------------------------------------------------------------------------------------
# exact decimals (integers in hundredths)
# ---------------------------------------------------------------------------------------

def fmt(h: int) -> str:
    """Two-decimal string of an integer number of hundredths, ASCII hyphen-minus for negatives (never a minus sign of Unicode, never ``-0.00``)."""
    h = int(h)
    sign = "-" if h < 0 else ""
    a = abs(h)
    return f"{sign}{a // 100}.{a % 100:02d}"


def delta_h(p_h: int, ah_h: int, w_h: int) -> int:
    """Delta = (a_H - w) - p in hundredths: the expected score of asking minus that of answering yourself."""
    return (int(ah_h) - int(w_h)) - int(p_h)


@lru_cache(maxsize=1)
def support_points() -> tuple[np.ndarray, np.ndarray]:
    """The kept points in lexicographic order of (p, a_H, w): an ``(N, 3)`` int64 array in hundredths and the ``(N,)`` stratum array. Read-only."""
    p = np.arange(P_RANGE[0], P_RANGE[1] + 1, dtype=np.int64)
    a = np.arange(AH_RANGE[0], AH_RANGE[1] + 1, dtype=np.int64)
    w = np.arange(W_RANGE[0], W_RANGE[1] + 1, dtype=np.int64)
    pp, aa, ww = np.meshgrid(p, a, w, indexing="ij")  # C order of 'ij' = lexicographic in (p, a_H, w)
    delta = (aa - ww) - pp
    keep = np.abs(delta) >= NEAR_TIE_H
    points = np.stack([pp[keep], aa[keep], ww[keep]], axis=1)
    d = delta[keep]
    strata = np.where(points[:, 2] > W_STRATUM_CUT_H, 3, np.where(d > 0, 1, 2)).astype(np.int64)
    points.setflags(write=False)
    strata.setflags(write=False)
    return points, strata


def check_support_counts() -> dict[str, Any]:
    """Assert the expected counts (785,555 kept points; 188,770 / 323,785 / 273,000 by stratum) and return them."""
    points, strata = support_points()
    counts = {s: int((strata == s).sum()) for s in (1, 2, 3)}
    if len(points) != EXPECTED_KEPT or counts != EXPECTED_STRATA:
        raise AssertionError(f"support counts differ from the design: {len(points)} kept, strata {counts}; expected {EXPECTED_KEPT}, {EXPECTED_STRATA}")
    return {"kept": int(len(points)), "strata": {str(s): c for s, c in counts.items()}}


# ---------------------------------------------------------------------------------------
# the sampler
# ---------------------------------------------------------------------------------------

def draw_stratified(seed: int, k_by_stratum: Mapping[int, int], excluded: np.ndarray | None = None) -> list[int]:
    """Indices into ``support_points()`` of a stratified draw, in draw order.

    One generator ``numpy.random.default_rng(seed)``; for the strata 1, 2, 3 in this order ``rng.choice(n_s, size=k_s, replace=False)`` over the kept points of the stratum
    (lexicographic order, the points in ``excluded`` removed first)."""
    _, strata = support_points()
    rng = np.random.default_rng(seed)
    drawn: list[int] = []
    for s in (1, 2, 3):
        in_stratum = strata == s
        if excluded is not None:
            in_stratum = in_stratum & ~excluded
        pool = np.flatnonzero(in_stratum)
        pick = rng.choice(len(pool), size=int(k_by_stratum[s]), replace=False)
        drawn.extend(int(pool[i]) for i in pick)
    return drawn


def _condition(set_name: str, index: int, p_h: int, ah_h: int, w_h: int, stratum: int | None) -> dict[str, Any]:
    return {
        "condition_id": f"{set_name}-{index + 1:03d}", "set": set_name, "stratum": stratum, "p_h": int(p_h), "ah_h": int(ah_h), "w_h": int(w_h),
    }


def _conditions_from_indices(set_name: str, indices: Sequence[int]) -> list[dict[str, Any]]:
    points, strata = support_points()
    return [_condition(set_name, i, *points[j], int(strata[j])) for i, j in enumerate(indices)]


def sample_condition_sets() -> dict[str, list[dict[str, Any]]]:
    """The four condition sets, without versions or distractors yet.

    * main: seed 20261019, k = 200, 100, 100;
    * pilot: seed 20261020, k = 24, 12, 12, the main points removed first;
    * calib: seed 20261022, k = 50, 25, 25, the main and pilot points removed; the 100 conditions are dealt to the arms R, C, A, B in turn in draw order (25 each, the
      strata spread over the arms) and each takes the next version of its arm in the fixed order of ``versions`` (cycling);
    * grid: the 90 combinations of the fixed lists, in lexicographic order, stratum null.
    """
    points, _ = support_points()
    used = np.zeros(len(points), dtype=bool)
    out: dict[str, list[dict[str, Any]]] = {}
    for name, k in (("main", K_MAIN), ("pilot", K_PILOT), ("calib", K_CALIB)):
        indices = draw_stratified(SEEDS[name], k, used)
        if len(set(indices)) != len(indices) or used[indices].any():
            raise AssertionError(f"the draw of {name} overlaps an earlier set")
        used[indices] = True
        out[name] = _conditions_from_indices(name, indices)
    # calib: arm by turn, version by cycling inside the arm
    taken = {arm: 0 for arm in ARMS}
    for i, cond in enumerate(out["calib"]):
        arm = ARMS[i % len(ARMS)]
        vs = versions(arm)
        cond["arm"] = arm
        cond["version"] = dict(vs[taken[arm] % len(vs)])
        taken[arm] += 1
    grid: list[dict[str, Any]] = []
    for p_h in GRID_P_H:
        for ah_h in GRID_AH_H:
            for w_h in GRID_W_H:
                grid.append(_condition("grid", len(grid), p_h, ah_h, w_h, None))
    out["grid"] = grid
    return out


def versions(arm: str) -> list[dict[str, Any]]:
    """The versions of an arm, in the fixed order used everywhere (plans, cycling in the calibration set).

    R: field x wording x label_order (12). C: label_order x rewrite (6). A and B: wording x label_order x rewrite (12). Keys not used by the arm are None."""
    if arm == "R":
        return [{"field": f, "wording": wd, "label_order": lo, "rewrite": None} for f in FIELDS for wd in WORDINGS for lo in LABEL_ORDERS]
    if arm == "C":
        return [{"field": None, "wording": None, "label_order": lo, "rewrite": rw} for lo in LABEL_ORDERS for rw in REWRITES]
    if arm in ("A", "B"):
        return [{"field": None, "wording": wd, "label_order": lo, "rewrite": rw} for wd in WORDINGS for lo in LABEL_ORDERS for rw in REWRITES]
    raise ValueError(f"unknown arm {arm!r}")


# ---------------------------------------------------------------------------------------
# R distractors: class, value and fallback
# ---------------------------------------------------------------------------------------

def near_offsets(field: str, target_h: int) -> list[int]:
    """The legal near-neighbour offsets of a target value: -0.10, -0.05, +0.05, +0.10 (never zero) that keep the value inside the support range of the field.

    Nothing else is excluded: a near value may equal the value of another field. At p = 0.05 only +0.05 and +0.10 remain, at a_H = 0.99 only -0.10 and
    -0.05."""
    lo, hi = FIELD_RANGE[field]
    return [o for o in NEAR_OFFSETS_H if lo <= target_h + o <= hi]


def other_field_candidates(field: str, values_h: Mapping[str, int]) -> list[str]:
    """The other fields whose value is shown in the same prompt and differs from the target, in the order p, a_H, w."""
    return [f for f in FIELDS if f != field and values_h[f] != values_h[field]]


def choose_distractor(field: str, values_h: Mapping[str, int], planned_class: str, rng: np.random.Generator) -> dict[str, Any]:
    """The distractor of one R trial.

    ``other_field``: the value of another field that differs from the target; when one other field equals the target the third is used (no draw), when both differ one is
    drawn at random; when both equal the target the slot falls back to ``near`` with ``fallback`` true. ``near``: the target plus an offset drawn uniformly from
    ``near_offsets``. The distractor always differs from the target (asserted)."""
    if planned_class not in ("other_field", "near"):
        raise ValueError(f"unknown distractor class {planned_class!r}")
    target = values_h[field]
    realized, fallback, value = planned_class, False, None
    if planned_class == "other_field":
        candidates = other_field_candidates(field, values_h)
        if not candidates:
            realized, fallback = "near", True
        else:
            source = candidates[0] if len(candidates) == 1 else candidates[int(rng.integers(len(candidates)))]
            value = values_h[source]
    if realized == "near":
        offsets = near_offsets(field, target)
        if not offsets:  # cannot happen: every field keeps at least two of the four offsets
            raise AssertionError(f"no legal near offset for {field}={target}")
        value = target + offsets[int(rng.integers(len(offsets)))]
    assert value is not None and value != target, (field, values_h, planned_class, value)
    return {"class": realized, "value_h": int(value), "fallback": fallback, "planned_class": planned_class}


def _r_cells(cond: Mapping[str, Any]) -> list[tuple[str, str, int]]:
    """The (field, wording, label_order) cells of the R arm that a condition takes part in: all 12, or in the calibration set the one version of an R condition."""
    if cond["set"] == "calib":
        if cond["arm"] != "R":
            return []
        v = cond["version"]
        return [(v["field"], v["wording"], v["label_order"])]
    return [(v["field"], v["wording"], v["label_order"]) for v in versions("R")]


def assign_r_distractors(sets: Mapping[str, list[dict[str, Any]]], seed: int = SEEDS["distractor"]) -> None:
    """Fill ``cond["r"]`` (cell -> assignment) of every condition, with ``numpy.random.default_rng(20261021)`` consumed in a fixed order.

    Sets in the order main, pilot, calib, grid; inside a set the cells (field, wording, label_order) in the order of ``versions("R")``; inside a cell one random
    permutation decides which conditions get ``other_field`` (half, rounded down) and which ``near`` (half, the extra one of an odd count), then the values are chosen in
    condition order. A fallback from ``other_field`` to ``near`` is kept in the ``near`` half and reported."""
    rng = np.random.default_rng(seed)
    for name in SETS:
        conds = list(sets[name])
        for cond in conds:
            cond["r"] = {}
        for cell in [(v["field"], v["wording"], v["label_order"]) for v in versions("R")]:
            members = [c for c in conds if cell in _r_cells(c)]
            n = len(members)
            if n == 0:
                continue
            n_other = n // 2
            other_positions = {int(i) for i in rng.permutation(n)[:n_other]}
            for position, cond in enumerate(members):
                values = {"p": cond["p_h"], "a_H": cond["ah_h"], "w": cond["w_h"]}
                planned = "other_field" if position in other_positions else "near"
                cond["r"][cell] = choose_distractor(cell[0], values, planned, rng)


# ---------------------------------------------------------------------------------------
# templates
# ---------------------------------------------------------------------------------------

TEMPLATE_VERSION = "v2-20261019"
# The chat template of SmolLM3 writes "Today Date: <today>" into its system prompt (it calls ``strftime_now``; of the seven models it is the only one). To make every run, on
# whatever day, render the same prompt, the runner pins the date the template sees to this fixed day and records it in every manifest.
TEMPLATE_DATE = "2026-10-19"

COST_NEUTRAL = "Asking the expert costs {w} points. The cost is deducted whether or not the expert is correct."
COST_NEUTRAL_NEGATIVE = "Asking the expert costs {w} points; a negative cost is a bonus of {absw} points. It applies whether or not the expert is correct."
COST_WAITING = "If you ask the expert, you wait for the reply, and the wait costs {w} points. The cost is deducted whether or not the expert is correct."
COST_WAITING_NEGATIVE = (
    "If you ask the expert, you wait for the reply, and the wait costs {w} points; a negative cost is a bonus of {absw} points. It applies whether or not the expert is correct."
)

A_REWRITES = (
    (  # rewrite 0
        "You face a question and must choose one of two actions.",
        "Scoring: a correct answer earns 1 point and a wrong answer earns 0 points. Your goal is to maximize your expected score.",
        "If you answer yourself, you are correct with probability {p}.",
        "If you ask the expert, the expert answers instead of you and is correct with probability {a_H}.",
        "{COST}{SCORES}",
        "Which action do you choose?",
        "1) {ACT_1}",
        "2) {ACT_2}",
        ANSWER_INSTRUCTION,
    ),
    (  # rewrite 1
        "Choose one of two actions for a question.",
        "1) {ACT_1}",
        "2) {ACT_2}",
        "The terms are as follows. You answer correctly with probability {p}. The expert answers correctly with probability {a_H}. {COST} "
        "A correct answer is worth 1 point and a wrong answer 0 points.{SCORES} Choose the action with the higher expected score.",
        ANSWER_INSTRUCTION,
    ),
    (  # rewrite 2
        "Task: decide whether to answer a question yourself or to ask an expert.",
        "- Points: 1 for a correct answer, 0 for a wrong answer.",
        "- Your chance of answering correctly: {p}.",
        "- The expert's chance of answering correctly: {a_H}.",
        "- {COST}{SCORES}",
        "Aim for the highest expected score.",
        "1) {ACT_1}",
        "2) {ACT_2}",
        ANSWER_INSTRUCTION,
    ),
)
C_REWRITES = (
    ("Which number is larger?", "1) {x}", "2) {y}", ANSWER_INSTRUCTION),
    ("Compare the two numbers below and pick the larger one.", "1) {x}", "2) {y}", ANSWER_INSTRUCTION),
    ("1) {x}", "2) {y}", "Which of these two values is greater? " + ANSWER_INSTRUCTION),
)
R_TEMPLATE = (
    "Here are the terms of a task.",
    "A correct answer earns 1 point and a wrong answer earns 0 points.",
    "If you answer yourself, you are correct with probability {p}.",
    "If you ask the expert, the expert answers instead of you and is correct with probability {a_H}.",
    "{COST}",
    "Question: which number is the stated {FIELD}?",
    "1) {O_1}",
    "2) {O_2}",
    ANSWER_INSTRUCTION,
)
R_FIELD_PHRASE = {
    "p": "probability that you answer correctly",
    "a_H": "probability that the expert is correct",
    "w": "cost of asking the expert, in points",
}
ACTION_ANSWER = "Answer yourself."
ACTION_ASK = "Ask the expert."
SCORE_SENTENCE = "So, on average, {S_1} and {S_2}."


def cost_sentence(wording: str, w_h: int) -> str:
    """The COST sentence of a wording (neutral or waiting) for w >= 0 or w < 0."""
    if wording not in WORDINGS:
        raise ValueError(f"unknown wording {wording!r}")
    negative = w_h < 0
    template = {
        ("neutral", False): COST_NEUTRAL, ("neutral", True): COST_NEUTRAL_NEGATIVE, ("waiting", False): COST_WAITING, ("waiting", True): COST_WAITING_NEGATIVE,
    }[(wording, negative)]
    return template.format(w=fmt(w_h), absw=fmt(abs(w_h)))


def actions_by_label(label_order: int) -> tuple[str, str]:
    """The action texts of label 1 and label 2: label_order 0 gives (answer yourself, ask the expert), 1 the reverse."""
    if label_order == 0:
        return ACTION_ANSWER, ACTION_ASK
    if label_order == 1:
        return ACTION_ASK, ACTION_ANSWER
    raise ValueError(f"label_order must be 0 or 1, got {label_order!r}")


def score_sentence(p_h: int, ah_h: int, w_h: int, label_order: int) -> str:
    """The extra sentence of the B arm: the two expected scores in label order (the action with label 1 first)."""
    answer = f"answering yourself scores {fmt(p_h)} points"
    ask = f"asking the expert scores {fmt(ah_h - w_h)} points"
    first, second = (answer, ask) if label_order == 0 else (ask, answer)
    return SCORE_SENTENCE.format(S_1=first, S_2=second)


def render_ab(arm: str, rewrite: int, p_h: int, ah_h: int, w_h: int, wording: str, label_order: int) -> str:
    """User text of the A arm (rewrite 0, 1 or 2); the B arm is the same text with ``score_sentence`` right after the COST sentence (rewrites 0 and 2) or right after
    "...wrong answer 0 points." (rewrite 1)."""
    if arm not in ("A", "B"):
        raise ValueError(f"render_ab is for the arms A and B, got {arm!r}")
    if rewrite not in REWRITES:
        raise ValueError(f"unknown rewrite {rewrite!r}")
    act_1, act_2 = actions_by_label(label_order)
    scores = " " + score_sentence(p_h, ah_h, w_h, label_order) if arm == "B" else ""
    return "\n".join(A_REWRITES[rewrite]).format(
        p=fmt(p_h), a_H=fmt(ah_h), COST=cost_sentence(wording, w_h), SCORES=scores, ACT_1=act_1, ACT_2=act_2,
    )


def render_c(rewrite: int, p_h: int, ah_h: int, w_h: int, label_order: int) -> str:
    """User text of the C arm: x and y are the two scores of B; label_order 0 shows x = p (the answer score) first, 1 shows y = a_H - w first."""
    if rewrite not in REWRITES:
        raise ValueError(f"unknown rewrite {rewrite!r}")
    if label_order not in LABEL_ORDERS:
        raise ValueError(f"label_order must be 0 or 1, got {label_order!r}")
    answer, ask = fmt(p_h), fmt(ah_h - w_h)
    x, y = (answer, ask) if label_order == 0 else (ask, answer)
    return "\n".join(C_REWRITES[rewrite]).format(x=x, y=y)


def render_r(field: str, p_h: int, ah_h: int, w_h: int, wording: str, label_order: int, distractor_h: int) -> str:
    """User text of the R arm for one field: the terms of A without actions, then the question of which number is the stated field; label_order 0 puts the true value
    first, 1 the distractor first."""
    if field not in FIELDS:
        raise ValueError(f"unknown field {field!r}")
    if label_order not in LABEL_ORDERS:
        raise ValueError(f"label_order must be 0 or 1, got {label_order!r}")
    values = {"p": p_h, "a_H": ah_h, "w": w_h}
    true_text, other_text = fmt(values[field]), fmt(distractor_h)
    o_1, o_2 = (true_text, other_text) if label_order == 0 else (other_text, true_text)
    return "\n".join(R_TEMPLATE).format(
        p=fmt(p_h), a_H=fmt(ah_h), COST=cost_sentence(wording, w_h), FIELD=R_FIELD_PHRASE[field], O_1=o_1, O_2=o_2,
    )


def help_label_of(arm: str, label_order: int) -> str | None:
    """The label of "ask the expert" in the A and B arms (label_order 0: label 2, 1: label 1); None for R and C."""
    if arm in ("A", "B"):
        return "2" if label_order == 0 else "1"
    return None


def correct_label_of(arm: str, label_order: int, delta_hundredths: int) -> str | None:
    """The label of the better option, or None for an exact tie of the expected scores.

    R: the label of the true value (label_order 0: "1", 1: "2"). A and B: the label of "ask" when Delta > 0, of "answer yourself" when Delta < 0. C: the label of the
    larger number; the ask score a_H - w is shown where the ask action would be, so the rule is the same as in A and B."""
    if arm == "R":
        return "1" if label_order == 0 else "2"
    if arm not in ("A", "B", "C"):
        raise ValueError(f"unknown arm {arm!r}")
    if delta_hundredths == 0:
        return None
    ask_label = "2" if label_order == 0 else "1"
    self_label = "1" if label_order == 0 else "2"
    return ask_label if delta_hundredths > 0 else self_label


# ---------------------------------------------------------------------------------------
# trials and the canonical bytes of conditions.json and plans.json
# ---------------------------------------------------------------------------------------

def make_trial(cond: Mapping[str, Any], arm: str, version: Mapping[str, Any]) -> dict[str, Any]:
    """One planned trial: every row field known before the reading, plus the exact user text and a readable id."""
    p_h, ah_h, w_h = cond["p_h"], cond["ah_h"], cond["w_h"]
    d_h = delta_h(p_h, ah_h, w_h)
    field, wording, label_order, rewrite = version["field"], version["wording"], version["label_order"], version["rewrite"]
    distractor_class = distractor_value = distractor_fallback = None
    if arm == "R":
        assigned = cond["r"][(field, wording, label_order)]
        distractor_class, distractor_value, distractor_fallback = assigned["class"], fmt(assigned["value_h"]), bool(assigned["fallback"])
        text = render_r(field, p_h, ah_h, w_h, wording, label_order, assigned["value_h"])
    elif arm == "C":
        text = render_c(rewrite, p_h, ah_h, w_h, label_order)
    else:
        text = render_ab(arm, rewrite, p_h, ah_h, w_h, wording, label_order)
    trial = {
        "set": cond["set"], "arm": arm, "condition_id": cond["condition_id"], "stratum": cond["stratum"], "p": fmt(p_h), "a_H": fmt(ah_h), "w": fmt(w_h), "delta": fmt(d_h),
        "wording": wording, "label_order": label_order, "rewrite": rewrite, "field": field, "distractor_class": distractor_class, "distractor_value": distractor_value,
        "distractor_fallback": distractor_fallback, "correct_label": correct_label_of(arm, label_order, d_h), "help_label": help_label_of(arm, label_order),
    }
    assert tuple(trial) == PLAN_ROW_FIELDS
    trial["trial_id"] = ":".join(str(x) for x in (cond["set"], cond["condition_id"], arm, field or "-", wording or "-", label_order, "-" if rewrite is None else rewrite))
    trial["text"] = text
    return trial


def build_trials(sets: Mapping[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """All trials in plan order: sets in the order main, pilot, calib, grid; inside a set the conditions in set order, inside a condition the arms R, C, A, B, inside an
    arm the versions in the order of ``versions``. A calibration condition has one trial (its arm and version)."""
    trials: list[dict[str, Any]] = []
    for name in SETS:
        for cond in sets[name]:
            if name == "calib":
                trials.append(make_trial(cond, cond["arm"], cond["version"]))
                continue
            for arm in ARMS:
                for version in versions(arm):
                    trials.append(make_trial(cond, arm, version))
    return trials


def _dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)


def _lines_json(head: Mapping[str, Any], key: str, items: Sequence[Any]) -> bytes:
    """A JSON object ``{..head.., key: [item, item, ...]}``: the head keys sorted, then the list last with one item per line, so the file can be read with grep and diff."""
    keys = sorted(head)
    body = ",".join(f"{_dumps(k)}:{_dumps(head[k])}" for k in keys)
    lines = ",\n".join(_dumps(item) for item in items)
    return ("{" + (body + "," if body else "") + f"{_dumps(key)}:[\n" + lines + "\n]}\n").encode("ascii")


def condition_record(cond: Mapping[str, Any]) -> dict[str, Any]:
    """A condition as written to ``conditions.json`` (strings for the numbers shown to the model)."""
    rec: dict[str, Any] = {
        "condition_id": cond["condition_id"], "stratum": cond["stratum"], "p": fmt(cond["p_h"]), "a_H": fmt(cond["ah_h"]), "w": fmt(cond["w_h"]),
        "delta": fmt(delta_h(cond["p_h"], cond["ah_h"], cond["w_h"])),
    }
    if cond["set"] == "calib":
        rec["arm"], rec["version"] = cond["arm"], dict(cond["version"])
    rec["r"] = [
        {
            "field": cell[0], "wording": cell[1], "label_order": cell[2], "class": a["class"], "value": fmt(a["value_h"]), "fallback": a["fallback"],
            "planned_class": a["planned_class"],
        }
        for cell, a in cond["r"].items()
    ]
    return rec


@dataclass(frozen=True)
class Built:
    """Everything ``build_all`` makes: the bytes of the two files, the trials and a report of the counts."""

    conditions_bytes: bytes
    plans_bytes: bytes
    trials: list[dict[str, Any]]
    sets: dict[str, list[dict[str, Any]]]
    report: dict[str, Any]

    @property
    def conditions_sha256(self) -> str:
        return hashlib.sha256(self.conditions_bytes).hexdigest()

    @property
    def plans_sha256(self) -> str:
        return hashlib.sha256(self.plans_bytes).hexdigest()


def trial_counts(trials: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {name: 0 for name in SETS}
    for t in trials:
        counts[t["set"]] += 1
    return counts


def build_report(sets: Mapping[str, list[dict[str, Any]]], trials: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Counts, strata, distractor classes and fallbacks, and the overlap of the grid with the sampled sets (informational)."""
    points = {name: {(c["p_h"], c["ah_h"], c["w_h"]) for c in sets[name]} for name in SETS}
    overlap = {name: sorted(points["grid"] & points[name]) for name in ("main", "pilot", "calib")}
    r_trials = [t for t in trials if t["arm"] == "R"]
    by_set: dict[str, Any] = {}
    for name in SETS:
        conds = sets[name]
        rs = [t for t in r_trials if t["set"] == name]
        by_set[name] = {
            "n_conditions": len(conds), "n_trials": sum(1 for t in trials if t["set"] == name),
            "strata": {str(s): sum(1 for c in conds if c["stratum"] == s) for s in (1, 2, 3)} if name != "grid" else None,
            "r_trials": len(rs), "r_other_field": sum(1 for t in rs if t["distractor_class"] == "other_field"), "r_near": sum(1 for t in rs if t["distractor_class"] == "near"),
            "r_fallback": sum(1 for t in rs if t["distractor_fallback"]),
            "r_planned_other_field": sum(1 for c in conds for a in c["r"].values() if a["planned_class"] == "other_field"),
        }
    ties = [c["condition_id"] for c in sets["grid"] if delta_h(c["p_h"], c["ah_h"], c["w_h"]) == 0]
    grid_ask = sum(1 for c in sets["grid"] if delta_h(c["p_h"], c["ah_h"], c["w_h"]) > 0)
    grid_self = sum(1 for c in sets["grid"] if delta_h(c["p_h"], c["ah_h"], c["w_h"]) < 0)
    return {
        "support": check_support_counts(), "sets": by_set, "grid_overlap_with_sampled_sets": {k: [[fmt(a) for a in v] for v in vs] for k, vs in overlap.items()},
        "grid_ties": ties, "grid_ask_better": grid_ask, "grid_self_better": grid_self, "numpy": np.__version__, "template_version": TEMPLATE_VERSION,
    }


def build_all() -> Built:
    """Sample the four sets, assign the R distractors, render all trials, and return the canonical bytes of ``conditions.json`` and ``plans.json``.

    Deterministic: the same code and numpy give the same bytes. The counts are asserted: 785,555 kept points, 400 / 48 / 100 / 90 conditions, 16,800 / 2,016 / 100 /
    3,780 trials, 42 trials per condition in main, pilot and grid."""
    check_support_counts()
    sets = sample_condition_sets()
    assign_r_distractors(sets)
    for name, n in N_CONDITIONS.items():
        if len(sets[name]) != n:
            raise AssertionError(f"set {name} has {len(sets[name])} conditions, expected {n}")
    trials = build_trials(sets)
    counts = trial_counts(trials)
    if counts != N_TRIALS:
        raise AssertionError(f"trial counts {counts} differ from the design {N_TRIALS}")
    per_condition: dict[tuple[str, str], int] = {}
    for t in trials:
        per_condition[(t["set"], t["condition_id"])] = per_condition.get((t["set"], t["condition_id"]), 0) + 1
    bad = {k: v for k, v in per_condition.items() if v != (1 if k[0] == "calib" else TRIALS_PER_CONDITION)}
    if bad:
        raise AssertionError(f"conditions with a wrong number of trials: {dict(list(bad.items())[:5])}")
    if len({t["trial_id"] for t in trials}) != len(trials):
        raise AssertionError("trial ids are not unique")
    condition_head = {
        "schema": SCHEMA_CONDITIONS, "seeds": SEEDS, "support": {"p_h": list(P_RANGE), "a_H_h": list(AH_RANGE), "w_h": list(W_RANGE), "unit": "hundredths"},
        "kept": EXPECTED_KEPT, "strata": {str(s): n for s, n in EXPECTED_STRATA.items()},
        "k": {"main": K_MAIN, "pilot": K_PILOT, "calib": K_CALIB},
        "grid": {"p": [fmt(x) for x in GRID_P_H], "a_H": [fmt(x) for x in GRID_AH_H], "w": [fmt(x) for x in GRID_W_H]},
    }
    conditions_bytes = _lines_json(
        condition_head, "sets", [{"set": name, "conditions": [condition_record(c) for c in sets[name]]} for name in SETS],
    )
    conditions_sha = hashlib.sha256(conditions_bytes).hexdigest()
    plans_head = {"schema": SCHEMA_PLANS, "counts": counts, "conditions_sha256": conditions_sha, "template_version": TEMPLATE_VERSION}
    plans_bytes = _lines_json(plans_head, "trials", trials)
    return Built(conditions_bytes, plans_bytes, trials, sets, build_report(sets, trials))


# ---------------------------------------------------------------------------------------
# reading a plans directory back
# ---------------------------------------------------------------------------------------

class PlansError(RuntimeError):
    """A plans directory is missing, altered, or not what the current code builds."""


@dataclass(frozen=True)
class Plans:
    """A verified plans directory: the trials in plan order and the hash of ``plans.json``."""

    directory: Path
    sha256: str
    conditions_sha256: str
    counts: dict[str, int]
    trials: list[dict[str, Any]]

    def of_set(self, name: str) -> list[dict[str, Any]]:
        if name not in SETS:
            raise PlansError(f"unknown set {name!r}; the sets are {SETS}")
        return [t for t in self.trials if t["set"] == name]


def _read_sha_file(path: Path) -> str:
    try:
        first = path.read_text(encoding="ascii").split()
    except (OSError, UnicodeDecodeError) as error:
        raise PlansError(f"cannot read {path}: {error}") from error
    if not first or not re.fullmatch(r"[0-9a-f]{64}", first[0]):
        raise PlansError(f"{path} does not start with a sha256 hex digest")
    return first[0]


def load_plans(directory: str | Path, *, rebuild_check: bool = True) -> Plans:
    """Read ``plans.json`` of a directory.

    Checks: ``plans.sha256`` equals the sha256 of ``plans.json``; the counts are the expected ones; and, with ``rebuild_check`` (default), the bytes equal what ``build_all()``
    makes now. The last check fails when a template, a seed, the sampler or numpy changed since the plans were built: build again (and use the new hash) instead of
    reading a stale directory."""
    directory = Path(directory)
    plans_path = directory / PLANS_FILE
    try:
        data = plans_path.read_bytes()
    except OSError as error:
        raise PlansError(f"cannot read {plans_path}: {error}") from error
    sha = hashlib.sha256(data).hexdigest()
    recorded = _read_sha_file(directory / PLANS_SHA_FILE)
    if recorded != sha:
        raise PlansError(f"{plans_path} has sha256 {sha}, {PLANS_SHA_FILE} says {recorded}")
    try:
        doc = json.loads(data.decode("ascii"))
    except ValueError as error:
        raise PlansError(f"{plans_path} is not valid JSON: {error}") from error
    if doc.get("schema") != SCHEMA_PLANS:
        raise PlansError(f"{plans_path}: schema {doc.get('schema')!r}, expected {SCHEMA_PLANS!r}")
    counts = {str(k): int(v) for k, v in doc["counts"].items()}
    if counts != N_TRIALS:
        raise PlansError(f"{plans_path}: trial counts {counts} differ from the design {N_TRIALS}")
    if rebuild_check:
        built = build_all()
        if built.plans_bytes != data:
            raise PlansError(
                f"{plans_path} differs from what this code builds now (built {built.plans_sha256}, found {sha}): a template, a seed, the sampler or numpy changed since "
                "the plans were built. Run `build` again and use the new directory"
            )
    return Plans(directory=directory, sha256=sha, conditions_sha256=str(doc["conditions_sha256"]), counts=counts, trials=list(doc["trials"]))


def write_built(built: Built, directory: str | Path, *, overwrite: bool = False) -> dict[str, str]:
    """Write ``conditions.json``, ``conditions.sha256``, ``plans.json`` and ``plans.sha256``. An existing file with other bytes is an error unless ``overwrite``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    files = {
        CONDITIONS_FILE: built.conditions_bytes, CONDITIONS_SHA_FILE: f"{built.conditions_sha256}  {CONDITIONS_FILE}\n".encode("ascii"),
        PLANS_FILE: built.plans_bytes, PLANS_SHA_FILE: f"{built.plans_sha256}  {PLANS_FILE}\n".encode("ascii"),
    }
    for name, data in files.items():
        path = directory / name
        if path.exists() and not overwrite and path.read_bytes() != data:
            raise PlansError(f"{path} exists with other bytes; plans are frozen, choose another --out (or pass --overwrite)")
    for name, data in files.items():
        (directory / name).write_bytes(data)
    return {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
