"""Prompts, items, z0, condition samplers, payoffs and frozen plans of the "repair composable conditional responses" experiment.

CPU only: no GPU, no network. ``nmi`` is imported read only.

No question text is printed or stored by anything here. Prompts exist in memory; plans hold item keys, numbers and labels only.

PUBLIC INTERFACE (stable; the training script imports it)

* ``decision_prompt(item, cond) -> dict``  keys ``text`` (the USER text), ``option_labels`` (the 4 answer digits as strings, in the order of the item's options), ``help_label`` (a digit
  string), ``spans`` (character spans in ``text``: ``cost_number``, ``rule_block``, ``decision`` and others) and ``meta``. ``cond`` is a ``Cond``, an ``nmi.mechanism.prompts.Condition`` or a dict
  (a plan row works). Variants: ``base`` (``render_prompt`` of the mechanism experiments, frames ``neutral``, ``waiting``, ``wait_mention``), ``net_score``, ``deduct_self``, ``deduct_wrong``.
* ``panswer_prompt(item, domain) -> str``  the USER text of the P-answer (Kumaran's template). Chat template and the assistant prefix are applied by ``render_for_model`` only, once.
* ``render_for_model(tokenizer, user_text, model_id, assistant_prefix=ASSISTANT_PREFIX) -> list[int]``  the token ids exactly as the readout sees them (chat template with
  ``enable_thinking: false``, then the prefix ``"Answer: "``, ``add_special_tokens=False``). The population set (the P-ask family of the operating-curve decision run) passes ``assistant_prefix=""``, as that run did.

Helpers for the training script and the evaluation: ``label_help(item_key, mapping_id)``, ``answer_labels(help_label)``, ``route_margin(logprobs5, help_label)``,
``top_answer_label(logprobs5, help_label)``, ``panswer_readout(logprobs5, temperature, gt_position)``, ``payoffs(cond, z0)``, ``load_items()``, ``load_z0(model_id, pool)``,
``load_z0_arc(model_id)``, ``sample_set(name, items, ...)``, ``build_model_plans(model_id)``,
``price_extrap_feasibility(...)``, ``plan_text_and_prefix(item, row)``, ``write_plan_file(...)``, ``read_plan_file(...)``.

Choices that the experiment specification leaves open are marked "CHOICE" in the comments.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from fractions import Fraction
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for _p in (str(REPO), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

from nmi.mechanism.prompts import (  # noqa: E402
    CALIBRATION_TEMPERATURES,
    Condition,
    calibrated_answer_chat_text,
    label_mapping,
    render_prompt,
)
from nmi.prompts import PromptBook, render_chat  # noqa: E402

DESIGN_PATH = REPO / "protocol/fine_tuning_design.md"
DESIGN_MD5 = "f0ecd3885685678aa8b3486dea7ecd75"
CONFIG_PATH = REPO / "configs/v6_confirm.yaml"
PROMPT_BOOK_PATH = REPO / "configs/prompts/ask_v1_1.yaml"
RESULTS = REPO / "results"
LEDGER_PATH = RESULTS / "holdout_ledger.jsonl"
PART2_RESULTS = RESULTS / "mechanism_20261016"

ASSISTANT_PREFIX = "Answer: "  # the prefix of the answer readout (``--answer-prefix 'Answer: '`` of the mechanism scripts); the last character is U+0020
# The chat template of SmolLM3-3B writes "Today Date: <today>" into the system header (``strftime_now("%d %B %Y")``). Left alone, the prompts of the base reading, the training and the
# later readings would carry different dates and the reference z0 of the calibration runs would not be reproduced. CHOICE: the date is pinned to the day on which the SmolLM3-3B calibration
# runs were made (see W5_SOURCES), by passing ``strftime_now`` as a template variable, as the mechanism experiments did to replay their collection day.
TEMPLATE_DATE_TEXT = "11 October 2026"
MODEL_IDS = ("qwen3-4b", "gemma-4-e4b-it", "smollm3-3b")
PART2_SHORT = {"qwen3-4b": "qwen", "gemma-4-e4b-it": "gemma", "smollm3-3b": "smol"}  # the prefixes of the <short>_test_ood run directories under PART2_RESULTS

# ---------------------------------------------------------------------------------------
# sets, seeds, grids
# ---------------------------------------------------------------------------------------

SEEDS = {
    "train": 20261031, "val": 20261032, "in_dist": 20261033, "unseen_combo": 20261034, "equal_net": 20261035, "sweep": 20261036, "deduct_position": 20261037,
    "price_extrap": 20261038, "waiting": 20261039, "stress": 20261040, "net_score": 20261041, "arc_in_dist": 20261042, "arc_waiting": 20261043,
}
PER_ITEM = {"train": 16, "val": 8, "in_dist": 4, "unseen_combo": 4, "price_extrap": 4, "waiting": 4, "stress": 2, "arc_in_dist": 4, "arc_waiting": 4}
DECISION_SETS = (
    "train", "val", "in_dist", "unseen_combo", "equal_net", "sweep", "deduct_position", "price_extrap", "waiting", "stress", "net_score", "wait_mention", "arc_in_dist",
    "arc_waiting", "prereg_grid", "population",
)
PANSWER_SETS = ("val_panswer", "eval_panswer", "arc_panswer")
ALL_SETS = DECISION_SETS + PANSWER_SETS
SET_POOL = {  # the item pool of every set
    "train": "cal_train", "val": "cal_val", "val_panswer": "cal_val", "arc_in_dist": "arc", "arc_waiting": "arc", "arc_panswer": "arc", "eval_panswer": "eval",
}
for _name in ("in_dist", "unseen_combo", "equal_net", "sweep", "deduct_position", "price_extrap", "waiting", "stress", "net_score", "wait_mention", "prereg_grid", "population"):
    SET_POOL[_name] = "eval"
# the sets that enter the success definition of the analysis (steps 3-5); the others are descriptive
DECISION_BEARING = ("in_dist", "unseen_combo", "deduct_position", "equal_net", "sweep", "price_extrap", "waiting")
DESCRIPTIVE = ("stress", "net_score", "wait_mention", "arc_in_dist", "arc_waiting", "prereg_grid", "population")
SELECTION_SETS = ("val", "val_panswer")  # the only readouts that checkpoint selection may use
# names for ``read --sets`` and ``inspect --sets``: a group expands to its sets (comma lists may mix groups and single sets)
SET_GROUPS = {
    "selection": SELECTION_SETS,
    "required": DECISION_BEARING + ("eval_panswer",),  # steps 1-5 of that success definition (price_extrap is a description for a model whose price extrapolation is not testable)
    "descriptive": DESCRIPTIVE + ("arc_panswer",),
}
SET_GROUPS["final"] = SET_GROUPS["required"] + SET_GROUPS["descriptive"]
SET_GROUPS["all"] = ALL_SETS


def expand_sets(text: str) -> list[str]:
    """``"selection,in_dist,descriptive"`` -> the sets in order of first mention, each once. An unknown name raises ``ValueError``."""
    out: list[str] = []
    for name in (t.strip() for t in text.split(",") if t.strip()):
        group = SET_GROUPS.get(name, (name,))
        for s in group:
            if s not in ALL_SETS:
                raise ValueError(f"unknown set {s!r}; sets: {list(ALL_SETS)}; groups: {list(SET_GROUPS)}")
            if s not in out:
                out.append(s)
    if not out:
        raise ValueError("no set given")
    return out


A_H_TRAIN_CENTS = tuple(range(55, 75)) + tuple(range(86, 100))  # {0.55..0.74} U {0.86..0.99}: {0.55..0.99} minus the hole [0.75, 0.85]
A_H_HOLE_CENTS = tuple(range(75, 86))
W_TRAIN_CENTS = tuple(range(-20, 41))  # {-0.20..0.40}
W_TRAIN_C010_CENTS = tuple(range(-20, 21))  # c0 = 0.10 only with w <= 0.20
C0_TRAIN_CENTS = (0, 10, 20)
W_UNSEEN_CENTS = tuple(range(21, 41))  # (0.20, 0.40]
X_DEDUCT_CENTS = tuple(range(5, 41))  # {0.05..0.40}
W_STRESS_CENTS = tuple(range(80, 161))  # [0.80, 1.60]
A_H_EXTRAP_CENTS = tuple(range(85, 100))  # [0.85, 0.99]
PRICE_EXTRAP_W_MAX = (0.70, 0.60, 0.50)  # the upper end of w, narrowed in this order while the help-optimal share is too small
PRICE_EXTRAP_MIN_SHARE = 0.20
UNSEEN_SINGLE_ACTION = (0.10, 0.90)  # a help-optimal share below 10% or above 90% marks unseen_combo "single-action"
SWEEP_OFFSETS_CENTS = (-15, -10, -5, 0, 1, 6, 11, 16)  # net values r + offset, r = floor(100 z0) / 100
SWEEP_C0_CENTS = (0, 20)  # c0 values of the sweep set
EQUAL_NET_OFFSETS_CENTS = (-10, 11)  # A_low = r - 0.10, A_high = r + 0.11
WAIT_MENTION_SECONDS = (0, 10)  # as in ``conditions('id')`` of nmi.mechanism.prompts: seconds = 10
WAIT_MENTION_A_H = 0.80  # as in ``conditions('id')`` of nmi.mechanism.prompts: lo = 0.80
PREREG_A_H = (0.60, 0.80, 0.95)
PREREG_C0 = (0.0, 0.10, 0.20)
PREREG_W = (0.0, 0.10, 0.20, 0.45, 0.80, 1.50)
PREREG_N_ITEMS = 500
SPLIT_SALT = "part3-b-split|"
PREREG_SALT = "part3-b-prereg|"  # CHOICE: "the first 500 items by key hash" = ascending sha256(salt + item_key)
WAIT_SALT = "part3-b-wait|"  # CHOICE: the mapping of the wait_mention trials is the parity of this hash (no seed is specified for that set)
VAL_N = 200
VARIANTS = ("base", "net_score", "deduct_self", "deduct_wrong")
FRAMES = ("neutral", "waiting", "wait_mention")

# the operating curve (grid.operating_curve of configs/v6_confirm.yaml): the P-ask prompt a00, F2, standard layout, a_H = 0.85, c0 = 0, 15 levels of w
POPULATION_PARAPHRASE = "a00"
POPULATION_A_H = 0.85
POPULATION_C0 = 0.0
POPULATION_W = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.45, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0)  # configs/v6_confirm.yaml grid.operating_curve.w


def cents(k: int) -> float:
    """The two-decimal float of an integer number of hundredths (exact decimal text, so ``f"{v:.2f}"`` and ``Fraction(str(v))`` agree)."""
    return float(f"{int(k) / 100:.2f}")


def to_cents(v: float) -> int:
    return int(round(float(v) * 100))


def exact(v: float) -> Fraction:
    """A two-decimal term as an exact fraction (``Fraction(str(v))``)."""
    return Fraction(f"{float(v):.2f}")


# ---------------------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------------------

_CONFIG_CACHE: dict[str, Any] = {}


def _config() -> dict[str, Any]:
    if "cfg" not in _CONFIG_CACHE:
        import yaml

        _CONFIG_CACHE["cfg"] = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return _CONFIG_CACHE["cfg"]


def model_entry(model_id: str) -> dict[str, Any]:
    if model_id not in MODEL_IDS:
        raise ValueError(f"unknown model {model_id!r}; the three models of this experiment are {MODEL_IDS}")
    for d in _config()["models"]["devices"]:
        if d["id"] == model_id:
            return dict(d)
    raise KeyError(model_id)


def model_path(model_id: str) -> str:
    return str(model_entry(model_id)["path"])


def chat_kwargs(model_id: str) -> dict[str, Any]:
    kwargs = dict(model_entry(model_id)["chat_template_kwargs"])
    if kwargs != {"enable_thinking": False}:
        raise ValueError(f"{model_id}: chat_template_kwargs {kwargs} differ from the frozen enable_thinking false")
    return kwargs


def temperature(model_id: str) -> float:
    return float(CALIBRATION_TEMPERATURES[model_id])


# ---------------------------------------------------------------------------------------
# the condition of a decision prompt
# ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Cond:
    """One decision condition. ``a_h``, ``c0``, ``w`` and ``x`` are two-decimal floats.

    ``variant``: ``base`` (``render_prompt`` of the mechanism experiments; ``frame`` neutral, waiting or wait_mention), ``net_score`` (neutral prompt plus the net-score sentence), ``deduct_self`` and
    ``deduct_wrong`` (full rule blocks; ``c0 = w = 0`` and the deduction ``x`` is stated instead)."""

    a_h: float
    c0: float
    w: float
    frame: str = "neutral"
    mapping_id: int = 0
    domain: str = "id"
    variant: str = "base"
    x: float | None = None
    wait_seconds: int | None = None

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {self.variant!r}")
        if self.frame not in FRAMES:
            raise ValueError(f"frame must be one of {FRAMES}, got {self.frame!r}")
        if self.domain not in ("id", "ood"):
            raise ValueError("domain must be id or ood")
        if self.mapping_id not in (0, 1):
            raise ValueError("mapping_id must be 0 or 1")
        for name in ("a_h", "c0", "w") + (("x",) if self.x is not None else ()):
            v = float(getattr(self, name))
            if abs(round(v * 100) - v * 100) > 1e-6:
                raise ValueError(f"{name}={v!r} is not a two-decimal value")
        if self.variant in ("deduct_self", "deduct_wrong"):
            if self.x is None or self.c0 != 0 or self.w != 0:
                raise ValueError("the deduction variants need x and c0 = w = 0")
            if self.frame != "neutral":
                raise ValueError("the deduction variants are written for the neutral frame")
        elif self.x is not None:
            raise ValueError("x belongs to the deduction variants only")
        if self.variant == "net_score" and self.frame != "neutral":
            raise ValueError("net_score is the neutral prompt plus one sentence")
        if (self.frame == "wait_mention") != (self.wait_seconds is not None):
            raise ValueError("wait_seconds is given exactly for the wait_mention frame")

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


_COND_FIELDS = tuple(f.name for f in fields(Cond))


def as_cond(obj: Any) -> Cond:
    """A ``Cond`` from a ``Cond``, an ``nmi.mechanism.prompts.Condition`` or a mapping (a plan row works; extra keys are ignored; ``a_H`` is accepted for ``a_h``)."""
    if isinstance(obj, Cond):
        return obj
    if isinstance(obj, Condition):
        return Cond(a_h=obj.a_h, c0=obj.c0, w=obj.w, frame=obj.frame, mapping_id=obj.mapping_id, domain=obj.domain, wait_seconds=obj.wait_seconds)
    if isinstance(obj, Mapping):
        d = dict(obj)
        if "a_h" not in d and "a_H" in d:
            d["a_h"] = d["a_H"]
        return Cond(**{k: d[k] for k in _COND_FIELDS if k in d})
    raise TypeError(f"cannot make a Cond from {type(obj).__name__}")


# ---------------------------------------------------------------------------------------
# labels and readout arithmetic (digit space: logprobs[0..4] belong to the digits "1".."5")
# ---------------------------------------------------------------------------------------

DIGITS = ("1", "2", "3", "4", "5")


def label_help(item_key: str, mapping_id: int) -> tuple[tuple[str, ...], str]:
    """The 4 answer digits (in the order of the item's options) and the help digit: the hash-fixed rotation of the mechanism experiments (``nmi.mechanism.prompts.label_mapping``)."""
    return label_mapping(item_key, mapping_id)


def answer_labels(help_label: str) -> tuple[str, ...]:
    """The four answer digits of a prompt whose help digit is ``help_label``, ascending."""
    if help_label not in DIGITS:
        raise ValueError(f"help label {help_label!r} is not one of 1..5")
    return tuple(d for d in DIGITS if d != help_label)


def route_margin(logprobs5: Sequence[float], help_label: str) -> float:
    """m = l_help - max over the 4 answer digits of l (the item's own labels, never a fixed column). NaN when no answer digit and not the help digit has probability."""
    lp = [float(v) for v in logprobs5]
    if len(lp) != 5:
        raise ValueError("five digit logprobs required")
    h = int(help_label) - 1
    best = max(lp[int(d) - 1] for d in answer_labels(help_label))
    if lp[h] == -math.inf and best == -math.inf:
        return math.nan
    return lp[h] - best


def top_answer_label(logprobs5: Sequence[float], help_label: str) -> str:
    """The answer digit of the largest logprob among the 4 answer digits; a tie goes to the LOWEST label."""
    lp = [float(v) for v in logprobs5]
    best = max(answer_labels(help_label), key=lambda d: (lp[int(d) - 1], -int(d)))
    return best


def option_gap(logprobs5: Sequence[float], help_label: str) -> float:
    """The gap between the two largest logprobs among the 4 answer digits (the quantity of delta_opt)."""
    top = sorted((float(logprobs5[int(d) - 1]) for d in answer_labels(help_label)), reverse=True)
    return float(top[0] - top[1])


def digit_mass(logprobs5: Sequence[float]) -> float:
    return float(sum(math.exp(float(v)) for v in logprobs5))


def panswer_readout(logprobs5: Sequence[float], temp: float, gt_position: int) -> dict[str, Any]:
    """P-answer: the 5 digits give the mass and the gate; ONLY the 4 answer digits "1".."4" give ``pred`` (ties to the lowest), ``z`` (max of softmax(logprobs / T)) and the correctness."""
    from nmi.measure.calibration import confidence

    lp = np.asarray([float(v) for v in logprobs5], dtype=np.float64)
    if lp.shape != (5,):
        raise ValueError("five digit logprobs required")
    mass = digit_mass(lp)
    pred = int(np.argmax(lp[:4])) + 1  # numpy argmax: the first (lowest) index on a tie
    z = float(confidence(lp[None, :4], float(temp))[0])
    return {"mass": mass, "gate_pass": bool(mass >= 0.5), "pred": pred, "z": z, "correct": bool(pred == int(gt_position))}


# ---------------------------------------------------------------------------------------
# payoffs (exact: the displayed two-decimal terms as exact fractions, the continuous z0 as its exact binary value)
# ---------------------------------------------------------------------------------------

def payoffs(cond: Any, z0: float) -> tuple[float, float, str]:
    """``(payoff_help, payoff_self, optimal)`` of one condition and the frozen z0.

    base and net_score: help = a_H - c0 - w, self = z0.  deduct_self: help = a_H, self = z0 - x.  deduct_wrong: help = a_H, self = z0 - x (1 - z0) (c0 = w = 0 there).
    ``optimal`` is ``help`` or ``self`` by exact comparison, ``tie`` when they are equal. z0 is never rounded into a payoff."""
    c = as_cond(cond)
    z = Fraction(float(z0))
    if c.variant in ("base", "net_score"):
        help_ = exact(c.a_h) - exact(c.c0) - exact(c.w)
        self_ = z
    elif c.variant == "deduct_self":
        help_ = exact(c.a_h) - exact(c.c0)
        self_ = z - exact(c.x)
    else:  # deduct_wrong
        help_ = exact(c.a_h) - exact(c.c0)
        self_ = z - exact(c.x) * (1 - z)
    optimal = "help" if help_ > self_ else ("self" if help_ < self_ else "tie")
    return float(help_), float(self_), optimal


def r_cents(z0: float) -> int:
    """r = floor(100 z0) / 100 in hundredths, exactly: r <= z0 < r + 0.01."""
    return int(math.floor(Fraction(float(z0)) * 100))


def z0_equals_r(z0: float) -> bool:
    return Fraction(float(z0)) * 100 == r_cents(z0)


# ---------------------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------------------

def _shift(spans: Mapping[str, tuple[int, int]], at: int, delta: int) -> dict[str, tuple[int, int]]:
    """Spans after a text edit at position ``at`` that adds ``delta`` characters (a span that starts at or after ``at`` moves; one that contains ``at`` grows)."""
    out = {}
    for name, (a, b) in spans.items():
        if a >= at:
            out[name] = (a + delta, b + delta)
        elif b > at:
            out[name] = (a, b + delta)
        else:
            out[name] = (a, b)
    return out


_NO_FEE_SELF = "No fee or deduction applies if you answer yourself. "
_DO_NOT_INFER = "Do not infer any additional benefit, harm, or cost beyond this scoring rule.\n"


def decision_prompt(item: Any, cond: Any) -> dict[str, Any]:
    """The decision prompt (USER text) of one item under one condition, with the labels and the spans.

    ``base`` is ``render_prompt`` of the mechanism experiments, unchanged. ``net_score`` inserts the sentence "So the expected net score of asking is {a_H - c0 - w} points. " directly after the deduction
    sentence (and the subsidy note when w < 0), before the sentence "No fee or deduction applies if you answer yourself." (CHOICE of the place: after the deduction sentence).
    ``deduct_self`` and ``deduct_wrong`` replace the whole rule block: the first line of the block of ``render_prompt`` (the domain's helper-accuracy statement, unchanged) is kept,
    a space follows, then the new sentences and the closing sentence. Return keys: text, option_labels (4 digit strings), help_label, spans, meta."""
    c = as_cond(cond)
    base = render_prompt(item, Condition("p3b", c.domain, c.a_h, c.c0, c.w, c.frame, c.mapping_id, c.wait_seconds))
    text, spans = base.text, dict(base.spans)
    if c.variant == "net_score":
        s, e = spans["rule_block"]
        block = text[s:e]
        k = block.index(_NO_FEE_SELF)
        if block.count(_NO_FEE_SELF) != 1:
            raise AssertionError("the sentence about answering yourself must occur once")
        net = exact(c.a_h) - exact(c.c0) - exact(c.w)
        sentence = f"So the expected net score of asking is {float(net):.2f} points. "
        at = s + k
        text = text[:at] + sentence + text[at:]
        spans = _shift(spans, at, len(sentence))
    elif c.variant in ("deduct_self", "deduct_wrong"):
        s, e = spans["rule_block"]
        block = text[s:e]
        first_line = block.split("\n", 1)[0]  # "The helper answers correctly with probability 0.80. This probability ... label."
        x_text = f"{float(c.x):.2f}"
        if c.variant == "deduct_self":
            tail = f" Asking the helper has no fee and no deduction. If you answer yourself, a score deduction of {x_text} points applies, charged regardless of whether your answer is correct. "
        else:
            tail = (f" Asking the helper has no fee and no deduction. If you answer yourself and your answer is wrong, a score deduction of {x_text} points applies; "
                    "nothing is deducted if your answer is correct. ")
        new_block = first_line + tail + _DO_NOT_INFER
        text = text[:s] + new_block + text[e:]
        x_at = s + len(first_line) + tail.index(x_text + " points")
        delta = len(new_block) - (e - s)
        keep = {k: v for k, v in _shift({k: v for k, v in spans.items() if k not in ("rule_block", "cost_number", "c0", "a_h")}, e, delta).items()}
        keep["a_h"] = spans["a_h"]  # inside the unchanged first line
        keep["rule_block"] = (s, s + len(new_block))
        keep["cost_number"] = (x_at, x_at + len(x_text))
        spans = keep
    meta = {
        "item_key": str(item.item_key), "domain": c.domain, "variant": c.variant, "frame": c.frame, "mapping_id": c.mapping_id, "a_h": c.a_h, "c0": c.c0, "w": c.w, "x": c.x,
        "wait_seconds": c.wait_seconds, "user_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }
    return {"text": text, "option_labels": list(base.option_labels), "help_label": base.help_label, "spans": spans, "meta": meta}


_BOOK: dict[str, PromptBook] = {}


def prompt_book() -> PromptBook:
    if "book" not in _BOOK:
        _BOOK["book"] = PromptBook.load(PROMPT_BOOK_PATH)
    return _BOOK["book"]


def panswer_prompt(item: Any, domain: str = "id") -> str:
    """The USER text of the P-answer: Kumaran's Phase 0/1 template of ``configs/prompts/ask_v1_1.yaml``, the original option order. The calibration runs and the mechanism experiments (MMLU-Redux and ARC alike,
    ``calibrated_answer_chat_text``) used this one template for both domains, so ``domain`` only validates the argument. No chat template and no prefix here."""
    if domain not in ("id", "ood"):
        raise ValueError("domain must be id or ood")
    return prompt_book().answer_prompt(item.question, item.choices)


def population_prompt(item: Any, w: float) -> str:
    """The USER text of the operating-curve prompt (P-ask a00, F2, standard layout, a_H = 0.85, c0 = 0, the displayed cost ``w``), exactly as ``scripts/_c2_common.ask_texts`` builds it
    before the chat template. The help digit is "5" and the options are "1".."4" (no rotation)."""
    return prompt_book().ask_prompt(POPULATION_PARAPHRASE, item.question, item.choices, framing="f2", a_H=POPULATION_A_H, c0=POPULATION_C0, w=float(w), layout="standard")


def template_kwargs(tokenizer: Any, model_id: str) -> dict[str, Any]:
    """The chat-template variables of a model: ``enable_thinking: false`` and, for a template that reads the clock (SmolLM3: ``strftime_now``), the pinned date ``TEMPLATE_DATE_TEXT``."""
    kwargs = chat_kwargs(model_id)
    template = getattr(tokenizer, "chat_template", None)
    if isinstance(template, str) and "strftime_now" in template:
        def frozen_now(fmt: str) -> str:
            if fmt != "%d %B %Y":
                raise ValueError(f"unexpected strftime format {fmt!r} in the chat template")
            return TEMPLATE_DATE_TEXT

        kwargs = {**kwargs, "strftime_now": frozen_now}
    return kwargs


def render_for_model(tokenizer: Any, user_text: str, model_id: str, assistant_prefix: str = ASSISTANT_PREFIX) -> list[int]:
    """The token ids of one user message exactly as the readout sees them: the chat template of the model (``enable_thinking: false``; SmolLM3's date pinned), then ``assistant_prefix``
    appended to the text, tokenised with ``add_special_tokens=False`` (the rule of ``nmi.readout.hf_backend.encode_prompts``). The next token is read: one of the digits "1".."5"."""
    text = render_chat(tokenizer, user_text, template_kwargs(tokenizer, model_id), assistant_prefix=assistant_prefix)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError("a prompt produced no tokens")
    return [int(t) for t in ids]


def render_text_for_model(tokenizer: Any, user_text: str, model_id: str, assistant_prefix: str = ASSISTANT_PREFIX) -> str:
    return render_chat(tokenizer, user_text, template_kwargs(tokenizer, model_id), assistant_prefix=assistant_prefix)


def ids_digest(ids: Sequence[int]) -> str:
    """sha256 of a token id list (int32 little endian): what the outputs store instead of any prompt text."""
    return hashlib.sha256(np.asarray(list(ids), dtype="<i4").tobytes()).hexdigest()


def calibrated_answer_ids(tokenizer: Any, item: Any, model_id: str) -> list[int]:
    """The token ids of ``nmi.mechanism.prompts.calibrated_answer_chat_text`` (the rendering of the P-answer in the mechanism experiments), the reference of the check below (same template variables, date pinned)."""
    text = calibrated_answer_chat_text(item, tokenizer, template_kwargs(tokenizer, model_id))
    return [int(t) for t in tokenizer(text, add_special_tokens=False)["input_ids"]]


def assert_panswer_matches_calibrated(tokenizer: Any, item: Any, model_id: str, domain: str = "id") -> None:
    """Check that the token ids of ``render_for_model(panswer_prompt(...))`` equal those of ``calibrated_answer_chat_text``."""
    mine = render_for_model(tokenizer, panswer_prompt(item, domain), model_id)
    theirs = calibrated_answer_ids(tokenizer, item, model_id)
    if mine != theirs:
        raise AssertionError(f"P-answer rendering differs from calibrated_answer_chat_text for model {model_id}")


# ---------------------------------------------------------------------------------------
# items
# ---------------------------------------------------------------------------------------

def split_digest(item_key: str) -> str:
    return hashlib.sha256((SPLIT_SALT + item_key).encode()).hexdigest()


def ledger_keys() -> frozenset[str]:
    """The item keys of ``results/holdout_ledger.jsonl`` (only that field is read)."""
    keys = set()
    if not LEDGER_PATH.is_file():  # the ledger of the study is not part of this release: the cross-check against it is skipped
        from nmi.provenance import note_skipped

        note_skipped(LEDGER_PATH, "holdout ledger")
        return frozenset()
    with LEDGER_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                k = json.loads(line).get("item_key")
                if k:
                    keys.add(k)
    return frozenset(keys)


def split_cal(items: Sequence[Any]) -> tuple[list[Any], list[Any]]:
    """The calibration pool into (train 800, validation 200): the 200 smallest ``sha256("part3-b-split|" + item_key)`` form the validation set. Both lists are in ascending item_key order."""
    ordered = sorted(items, key=lambda i: split_digest(i.item_key))
    if len(ordered) <= VAL_N:
        raise ValueError("pool too small for the split")
    val = sorted(ordered[:VAL_N], key=lambda i: i.item_key)
    train = sorted(ordered[VAL_N:], key=lambda i: i.item_key)
    return train, val


def assert_item_pools(pools: Mapping[str, Sequence[Any]]) -> dict[str, Any]:
    """Train, val, eval and ARC are pairwise disjoint by item_key, keys are unique inside a pool, and no key is in the holdout ledger."""
    keysets = {name: [i.item_key for i in items] for name, items in pools.items()}
    report: dict[str, Any] = {"sizes": {k: len(v) for k, v in keysets.items()}}
    for name, keys in keysets.items():
        if len(set(keys)) != len(keys):
            raise AssertionError(f"pool {name} has duplicate item keys")
    names = list(keysets)
    for a, b in itertools.combinations(names, 2):
        shared = set(keysets[a]) & set(keysets[b])
        if shared:
            raise AssertionError(f"pools {a} and {b} share {len(shared)} item keys")
    led = ledger_keys()
    for name, keys in keysets.items():
        hit = led & set(keys)
        if hit:
            raise AssertionError(f"pool {name} has {len(hit)} keys in the holdout ledger")
    report["ledger_keys_read"] = len(led)
    report["pairwise_disjoint"] = True
    report["ledger_overlap"] = 0
    return report


_ITEM_CACHE: dict[str, Any] = {}


def load_items() -> dict[str, list[Any]]:
    """``{"train": 800, "val": 200, "eval": 2000, "arc": 512}`` as ``McItem`` lists in ascending item_key order, with the disjointness and ledger assertions (cached).

    The calibration and evaluation pools come through ``nmi.data.registry`` and ``nmi.data.mmlu_redux`` (pools ``mmlu_redux_cal`` and ``mmlu_redux_eval``, development pools);
    the ARC items are the 512 of the split ``test`` of the mechanism experiments (``nmi.mechanism.data.load_split("test", allow_test=True)``)."""
    if "items" in _ITEM_CACHE:
        return _ITEM_CACHE["items"]
    from nmi.data import mmlu_redux
    from nmi.data.registry import SplitRegistry
    from nmi.mechanism.data import MMLU_ROOT, load_split

    registry = SplitRegistry.load()
    cal = mmlu_redux.load_pool(registry, "mmlu_redux_cal", MMLU_ROOT)
    ev = mmlu_redux.load_pool(registry, "mmlu_redux_eval", MMLU_ROOT)
    arc = load_split("test", allow_test=True)
    train, val = split_cal(cal)
    pools = {"train": train, "val": val, "eval": sorted(ev, key=lambda i: i.item_key), "arc": sorted(arc, key=lambda i: i.item_key)}
    if (len(train), len(val), len(pools["eval"]), len(pools["arc"])) != (800, 200, 2000, 512):
        raise AssertionError("unexpected pool sizes")
    report = assert_item_pools(pools)
    _ITEM_CACHE["items"] = pools
    _ITEM_CACHE["report"] = report
    return pools


def items_for_set(set_name: str, items: Mapping[str, Sequence[Any]] | None = None) -> list[Any]:
    items = items if items is not None else load_items()
    pool = {"cal_train": "train", "cal_val": "val", "eval": "eval", "arc": "arc"}[SET_POOL[set_name]]
    out = list(items[pool])
    if set_name == "prereg_grid":
        out = sorted(out, key=lambda i: hashlib.sha256((PREREG_SALT + i.item_key).encode()).hexdigest())[:PREREG_N_ITEMS]
        out = sorted(out, key=lambda i: i.item_key)
    return out


# ---------------------------------------------------------------------------------------
# z0
# ---------------------------------------------------------------------------------------

W5_SOURCES: dict[str, list[tuple[str, tuple[str, ...]]]] = {
    "qwen3-4b": [("results/calib/20261010_231628_w5a11_qwen3-4b", ("mmlu_redux_cal", "mmlu_redux_eval"))],
    "gemma-4-e4b-it": [("results/calib/20261010_232455_w5a11_gemma-4-e4b-it", ("mmlu_redux_cal", "mmlu_redux_eval"))],
    # the first SmolLM3-3B run holds the calibration pool only; its evaluation pool is in the second run (same config family, 1,000 + 2,000 items).
    # CHOICE: the eval z0 of SmolLM3-3B is recomputed from that run's raw option logprobs with the FROZEN temperature (the run itself re-fitted 2.7762 on its own calibration rows)
    "smollm3-3b": [("results/calib/20261011_064541_gen_w5_smollm3-3b", ("mmlu_redux_cal",)), ("results/calib/20261011_160335_gen_w5eval_smollm3-3b", ("mmlu_redux_eval",))],
}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def load_z0(model_id: str, pool: str) -> tuple[dict[str, float], dict[str, Any]]:
    """Frozen z0 of a model on ``mmlu_redux_cal`` or ``mmlu_redux_eval``: ``({item_key: z0}, provenance)``.

    z0 = max of softmax(raw option logprobs / T) with the FROZEN temperature ``CALIBRATION_TEMPERATURES[model_id]`` (``nmi.measure.calibration.confidence``), from the
    ``p_options_raw`` of the calibration table. Where the run's own temperature is the frozen one this equals the stored ``z`` (checked to 1e-9; the maximum difference is in the provenance)."""
    import pyarrow.parquet as pq
    from nmi.measure.calibration import confidence

    if model_id not in W5_SOURCES:
        raise ValueError(f"no calibration source for {model_id}")
    frozen_t = temperature(model_id)
    for run, pools in W5_SOURCES[model_id]:
        if pool not in pools:
            continue
        path = REPO / run / "calib.parquet"
        table = pq.read_table(path, columns=["item_key", "model", "pool", "temperature", "p_options_raw", "z"]).to_pylist()
        rows = [r for r in table if r["pool"] == pool]
        if not rows:
            raise ValueError(f"{path}: no rows of pool {pool}")
        if {r["model"] for r in rows} != {model_id}:
            raise ValueError(f"{path}: rows of another model")
        keys = [r["item_key"] for r in rows]
        if len(set(keys)) != len(keys):
            raise ValueError(f"{path}: duplicate item keys")
        lp = np.asarray([r["p_options_raw"] for r in rows], dtype=np.float64)
        z = confidence(lp, frozen_t)
        stored = np.asarray([r["z"] for r in rows], dtype=np.float64)
        run_temps = sorted({float(r["temperature"]) for r in rows})
        diff = float(np.max(np.abs(z - stored)))
        same_t = len(run_temps) == 1 and abs(run_temps[0] - frozen_t) < 1e-12
        if same_t and diff > 1e-9:
            raise ValueError(f"{path}: recomputed z differs from the stored z by {diff:g} although the temperature is the frozen one")
        info = {
            "run": run, "parquet": str(path.relative_to(REPO)), "parquet_sha256": sha256_file(path), "pool": pool, "n": len(rows), "frozen_temperature": frozen_t, "run_temperatures": run_temps,
            "run_temperature_is_frozen": same_t, "max_abs_diff_recomputed_vs_stored_z": diff,
        }
        manifest_path = REPO / run / "manifest.json"
        if manifest_path.exists():  # the run must have used the current P-answer prompt file (ask_v1_1.yaml)
            import hashlib as _h

            recorded = json.loads(manifest_path.read_text(encoding="utf-8")).get("input_md5", {}).get("prompts")
            current = _h.md5(PROMPT_BOOK_PATH.read_bytes(), usedforsecurity=False).hexdigest()
            if recorded is not None and recorded != current:
                raise ValueError(f"{manifest_path}: the run used prompts md5 {recorded}, the file now has {current}")
            info.update(manifest_sha256=sha256_file(manifest_path), prompts_md5=recorded)
        return {k: float(v) for k, v in zip(keys, z)}, info
    raise ValueError(f"{model_id}: no calibration run lists pool {pool}")


def load_z0_arc(model_id: str) -> tuple[dict[str, float], dict[str, Any]]:
    """Frozen z0 of a model on the 512 ARC items: the calibrated answer readout of the mechanism experiments, ``<short>_test_ood/answers.jsonl`` under ``PART2_RESULTS`` (``--with-calibrated-z``,
    ``calibrated_answer_chat_text``, the frozen temperature). ``({item_key: z0}, provenance)``; z is recomputed from ``raw_answer_logits`` and compared with the stored z."""
    from nmi.measure.calibration import confidence

    path = PART2_RESULTS / f"{PART2_SHORT[model_id]}_test_ood" / "answers.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    keys = [r["item_key"] for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{path}: duplicate item keys")
    frozen_t = temperature(model_id)
    if {float(r["temperature"]) for r in rows} != {frozen_t}:
        raise ValueError(f"{path}: the temperature is not the frozen one")
    z = confidence(np.asarray([r["raw_answer_logits"] for r in rows], dtype=np.float64), frozen_t)  # softmax(logits / T) = softmax(logprobs / T)
    stored = np.asarray([r["z"] for r in rows], dtype=np.float64)
    diff = float(np.max(np.abs(z - stored)))
    if diff > 1e-9:
        raise ValueError(f"{path}: recomputed z differs from the stored z by {diff:g}")
    info = {"answers_jsonl": str(path.relative_to(REPO)), "sha256": sha256_file(path), "n": len(rows), "frozen_temperature": frozen_t, "max_abs_diff_recomputed_vs_stored_z": diff,
            "readout": "HF bf16 forward of Part 2 (run_mechanism_internal.py --with-calibrated-z), calibrated_answer_chat_text"}
    return {k: float(v) for k, v in zip(keys, z)}, info


_Z0_CACHE: dict[str, Any] = {}


def z0_all(model_id: str) -> tuple[dict[str, float], dict[str, Any]]:
    """z0 of a model on the calibration, evaluation and ARC items together (the key sets are disjoint) with the provenance of each part (cached)."""
    if model_id in _Z0_CACHE:
        return _Z0_CACHE[model_id]
    cal, i1 = load_z0(model_id, "mmlu_redux_cal")
    ev, i2 = load_z0(model_id, "mmlu_redux_eval")
    arc, i3 = load_z0_arc(model_id)
    merged: dict[str, float] = {}
    for part in (cal, ev, arc):
        if set(part) & set(merged):
            raise AssertionError("z0 key sets overlap")
        merged.update(part)
    _Z0_CACHE[model_id] = (merged, {"cal": i1, "eval": i2, "arc": i3})
    return _Z0_CACHE[model_id]


# ---------------------------------------------------------------------------------------
# samplers: one default_rng(seed) per set, items in ascending item_key order, per condition the draw order c0, w | c0, a_H, mapping
# ---------------------------------------------------------------------------------------

class _Draw:
    """Uniform draws from one generator; a single-valued field consumes nothing (CHOICE: a field with one possible value has no draw)."""

    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def pick(self, values: Sequence[Any]) -> Any:
        if len(values) == 1:
            return values[0]
        return values[int(self.rng.integers(0, len(values)))]


def _w_grid_for_c0(c0_cents: int) -> Sequence[int]:
    return W_TRAIN_C010_CENTS if c0_cents == 10 else W_TRAIN_CENTS


def _train_triple(d: _Draw, a_h_cents: Sequence[int] = A_H_TRAIN_CENTS) -> tuple[int, int, int]:
    """(a_H, c0, w) in hundredths by the training rule: c0 uniform {0, 10, 20}; w uniform on {-20..40}, on {-20..20} when c0 = 10 (so P(c0 = 0.10) = 1/3); a_H uniform on the grid."""
    c0 = d.pick(C0_TRAIN_CENTS)
    w = d.pick(_w_grid_for_c0(c0))
    a = d.pick(a_h_cents)
    return a, c0, w


def _trial(set_name: str, item: Any, k: int, cond: Cond, **extra: Any) -> dict[str, Any]:
    labels, help_label = label_help(item.item_key, cond.mapping_id)
    row = {
        "set": set_name, "item_key": item.item_key, "k": k, "family": "mech", **cond.to_dict(), "option_labels": list(labels), "help_label": help_label, "expr": "single", "group": None, "net": None,
        "tie_point": False,
    }
    row.update(extra)
    return row


def _sample_training_like(name: str, items: Sequence[Any], n_per_item: int, *, domain: str = "id", frame: str = "neutral", seed: int | None = None) -> list[dict[str, Any]]:
    d = _Draw(SEEDS[name] if seed is None else seed)
    rows = []
    for item in sorted(items, key=lambda i: i.item_key):
        for k in range(n_per_item):
            a, c0, w = _train_triple(d)
            m = d.pick((0, 1))
            cond = Cond(a_h=cents(a), c0=cents(c0), w=cents(w), frame=frame, mapping_id=m, domain=domain)
            rows.append(_trial(name, item, k, cond))
    return rows


def _sample_unseen_combo(items: Sequence[Any]) -> list[dict[str, Any]]:
    d = _Draw(SEEDS["unseen_combo"])
    rows = []
    for item in sorted(items, key=lambda i: i.item_key):
        for k in range(4):
            if k < 2:  # conditions 1-2: c0 = 0.10 with w in (0.20, 0.40], a_H on the training grid
                c0 = 10
                w = d.pick(W_UNSEEN_CENTS)
                a = d.pick(A_H_TRAIN_CENTS)
                kind = "c0_010_high_w"
            else:  # conditions 3-4: a_H in the hole, c0 and w by the training rule
                c0 = d.pick(C0_TRAIN_CENTS)
                w = d.pick(_w_grid_for_c0(c0))
                a = d.pick(A_H_HOLE_CENTS)
                kind = "a_h_in_hole"
            m = d.pick((0, 1))
            rows.append(_trial("unseen_combo", item, k, Cond(a_h=cents(a), c0=cents(c0), w=cents(w), mapping_id=m), group=kind))
    return rows


def sample_price_extrap(items: Sequence[Any], w_max: float = 0.70) -> list[dict[str, Any]]:
    """w uniform on (0.40, w_max] (hundredths 41..), a_H uniform on [0.85, 0.99], c0 = 0, 4 per item, seed 20261038; the mapping is uniform on {0, 1}."""
    w_grid = tuple(range(41, to_cents(w_max) + 1))
    if not w_grid:
        raise ValueError("w_max must exceed 0.40")
    d = _Draw(SEEDS["price_extrap"])
    rows = []
    for item in sorted(items, key=lambda i: i.item_key):
        for k in range(4):
            w = d.pick(w_grid)
            a = d.pick(A_H_EXTRAP_CENTS)
            m = d.pick((0, 1))
            rows.append(_trial("price_extrap", item, k, Cond(a_h=cents(a), c0=0.0, w=cents(w), mapping_id=m)))
    return rows


def _sample_stress(items: Sequence[Any]) -> list[dict[str, Any]]:
    # CHOICE: only w is specified for this set. c0 = 0, a_H on the training grid, mapping uniform.
    d = _Draw(SEEDS["stress"])
    rows = []
    for item in sorted(items, key=lambda i: i.item_key):
        for k in range(2):
            w = d.pick(W_STRESS_CENTS)
            a = d.pick(A_H_TRAIN_CENTS)
            m = d.pick((0, 1))
            rows.append(_trial("stress", item, k, Cond(a_h=cents(a), c0=0.0, w=cents(w), mapping_id=m)))
    return rows


def _sample_deduct_position(items: Sequence[Any]) -> list[dict[str, Any]]:
    # CHOICE: 2 conditions (x, a_H, mapping) per item, each rendered under both variants (crossed), so that the two positions are paired. Trials are ordered variant-major per item.
    d = _Draw(SEEDS["deduct_position"])
    rows = []
    for item in sorted(items, key=lambda i: i.item_key):
        conds = []
        for _ in range(2):
            x = d.pick(X_DEDUCT_CENTS)
            a = d.pick(A_H_TRAIN_CENTS)
            m = d.pick((0, 1))
            conds.append((x, a, m))
        k = 0
        for variant in ("deduct_self", "deduct_wrong"):
            for j, (x, a, m) in enumerate(conds):
                rows.append(_trial("deduct_position", item, k, Cond(a_h=cents(a), c0=0.0, w=0.0, mapping_id=m, variant=variant, x=cents(x)), group=f"pair{j}"))
                k += 1
    return rows


def _sample_net_score(items: Sequence[Any]) -> list[dict[str, Any]]:
    # 2 conditions per item at the same a_H (drawn from the training grid): c0 = 0 with w = 0 and w = 0.10. CHOICE: the mapping is drawn once per item, so that the pair differs in w only.
    d = _Draw(SEEDS["net_score"])
    rows = []
    for item in sorted(items, key=lambda i: i.item_key):
        a = d.pick(A_H_TRAIN_CENTS)
        m = d.pick((0, 1))
        for k, w in enumerate((0, 10)):
            rows.append(_trial("net_score", item, k, Cond(a_h=cents(a), c0=0.0, w=cents(w), mapping_id=m, variant="net_score")))
    return rows


def _sample_wait_mention(items: Sequence[Any]) -> list[dict[str, Any]]:
    rows = []
    for item in sorted(items, key=lambda i: i.item_key):
        m = int(hashlib.sha256((WAIT_SALT + item.item_key).encode()).hexdigest(), 16) % 2
        for k, seconds in enumerate(WAIT_MENTION_SECONDS):
            rows.append(_trial("wait_mention", item, k, Cond(a_h=WAIT_MENTION_A_H, c0=0.0, w=0.0, frame="wait_mention", mapping_id=m, wait_seconds=seconds)))
    return rows


def _sample_prereg_grid(items: Sequence[Any]) -> list[dict[str, Any]]:
    rows = []
    for item in sorted(items, key=lambda i: i.item_key):
        k = 0
        for a, c0, w in itertools.product(PREREG_A_H, PREREG_C0, PREREG_W):
            rows.append(_trial("prereg_grid", item, k, Cond(a_h=a, c0=c0, w=w, mapping_id=0)))
            k += 1
    return rows


def _sample_population(items: Sequence[Any]) -> list[dict[str, Any]]:
    rows = []
    for item in sorted(items, key=lambda i: i.item_key):
        for k, w in enumerate(POPULATION_W):
            rows.append({
                "set": "population", "item_key": item.item_key, "k": k, "family": "pask", "domain": "id", "variant": "pask", "frame": "f2", "a_h": POPULATION_A_H, "c0": POPULATION_C0, "w": w, "x": None,
                "mapping_id": None, "wait_seconds": None, "option_labels": ["1", "2", "3", "4"], "help_label": "5", "expr": "single", "group": None, "net": None, "tie_point": False,
                "paraphrase": POPULATION_PARAPHRASE,
            })
    return rows


def sweep_pairs(r_cent: int) -> list[tuple[int, int]]:
    """The feasible (a_H, c0) pairs of one sweep item in hundredths: a_H on the training grid outside the hole, c0 in {0, 0.20}, a_H - c0 in [r - 0.04, r + 0.25]. Enumerated with a_H
    ascending and c0 ascending inside (CHOICE of the order of the enumeration, which fixes what ``default_rng`` picks)."""
    return [(a, c) for a in A_H_TRAIN_CENTS for c in SWEEP_C0_CENTS if r_cent - 4 <= a - c <= r_cent + 25]


def _sample_sweep(items: Sequence[Any], z0: Mapping[str, float]) -> list[dict[str, Any]]:
    d = _Draw(SEEDS["sweep"])
    rows = []
    for item in sorted(items, key=lambda i: i.item_key):
        z = z0[item.item_key]
        r = r_cents(z)
        pairs = sweep_pairs(r)
        if not pairs:
            raise AssertionError(f"no feasible (a_H, c0) pair for a sweep item (r = {r / 100:.2f})")
        a, c0 = d.pick(pairs)
        tie = z0_equals_r(z)
        for j, off in enumerate(SWEEP_OFFSETS_CENTS):
            net = r + off
            w = a - c0 - net
            if not -20 <= w <= 40:
                raise AssertionError("a sweep w left the training support")
            rows.append(_trial("sweep", item, j, Cond(a_h=cents(a), c0=cents(c0), w=cents(w), mapping_id=0), group=j, net=cents(net), tie_point=bool(tie and off == 0)))
    return rows


def equal_net_triples(net_cents: int) -> list[tuple[int, int, int]]:
    """All (a_H, c0, w) in hundredths with a_H on the training grid (outside the hole), c0 in {0, 10, 20}, w in [-20, 40], w <= 20 when c0 = 10, and a_H - c0 - w = net. Sorted."""
    out = []
    for a in A_H_TRAIN_CENTS:
        for c0 in C0_TRAIN_CENTS:
            w = a - c0 - net_cents
            if -20 <= w <= 40 and (c0 != 10 or w <= 20):
                out.append((a, c0, w))
    return out


def _draw_three_distinct(d: _Draw, candidates: Sequence[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    """Three triples that differ pairwise in a_H and in w, by sequential uniform draws; a failed greedy attempt is restarted (the stream goes on), up to 50 times."""
    for _ in range(50):
        pool = list(candidates)
        chosen: list[tuple[int, int, int]] = []
        while len(chosen) < 3 and pool:
            t = pool[int(d.rng.integers(0, len(pool)))]
            chosen.append(t)
            pool = [u for u in pool if u[0] != t[0] and u[2] != t[2]]
        if len(chosen) == 3:
            return chosen
    raise AssertionError("no 3 triples that differ in a_H and in w")


def _sample_equal_net(items: Sequence[Any], z0: Mapping[str, float]) -> list[dict[str, Any]]:
    d = _Draw(SEEDS["equal_net"])
    rows = []
    for item in sorted(items, key=lambda i: i.item_key):
        z = z0[item.item_key]
        r = r_cents(z)
        k = 0
        for g, off in enumerate(EQUAL_NET_OFFSETS_CENTS):
            net = r + off
            triples = _draw_three_distinct(d, equal_net_triples(net))
            for e, (a, c0, w) in enumerate(triples):
                rows.append(_trial("equal_net", item, k, Cond(a_h=cents(a), c0=cents(c0), w=cents(w), mapping_id=0), group=g, expr=f"e{e}", net=cents(net)))
                k += 1
    return rows


def _sample_panswer(name: str, items: Sequence[Any]) -> list[dict[str, Any]]:
    rows = []
    domain = "ood" if name == "arc_panswer" else "id"
    for item in sorted(items, key=lambda i: i.item_key):
        rows.append({
            "set": name, "item_key": item.item_key, "k": 0, "family": "panswer", "domain": domain, "variant": "panswer", "frame": "panswer", "gt_position": int(item.gt_position),
            "option_labels": ["1", "2", "3", "4"], "help_label": "5", "expr": "single", "group": None,
        })
    return rows


def sample_set(name: str, items: Sequence[Any], *, z0: Mapping[str, float] | None = None, w_max: float = 0.70) -> list[dict[str, Any]]:
    """The trials of one set (no z0 and no payoffs yet): dicts in ascending (item_key, k) order. ``z0`` is needed for ``equal_net`` and ``sweep``; ``w_max`` only for ``price_extrap``."""
    if name in ("train", "val", "in_dist"):
        return _sample_training_like(name, items, PER_ITEM[name])
    if name == "waiting":
        return _sample_training_like(name, items, PER_ITEM[name], frame="waiting")
    if name == "arc_in_dist":
        return _sample_training_like(name, items, PER_ITEM[name], domain="ood")
    if name == "arc_waiting":
        return _sample_training_like(name, items, PER_ITEM[name], domain="ood", frame="waiting")
    if name == "unseen_combo":
        return _sample_unseen_combo(items)
    if name == "price_extrap":
        return sample_price_extrap(items, w_max)
    if name == "stress":
        return _sample_stress(items)
    if name == "deduct_position":
        return _sample_deduct_position(items)
    if name == "net_score":
        return _sample_net_score(items)
    if name == "wait_mention":
        return _sample_wait_mention(items)
    if name == "prereg_grid":
        return _sample_prereg_grid(items)
    if name == "population":
        return _sample_population(items)
    if name in PANSWER_SETS:
        return _sample_panswer(name, items)
    if name in ("sweep", "equal_net"):
        if z0 is None:
            raise ValueError(f"{name} needs z0")
        return _sample_sweep(items, z0) if name == "sweep" else _sample_equal_net(items, z0)
    raise ValueError(f"unknown set {name!r}")


# ---------------------------------------------------------------------------------------
# plans: rows with z0, payoffs and trial ids
# ---------------------------------------------------------------------------------------

def gate_unit(row: Mapping[str, Any]) -> str:
    """The unit of the main quality gate: test set x version, version = frame x variant x expression type; the mapping is NOT a stratum (it is reported as a description)."""
    return f"{row['set']}/{row['frame']}/{row['variant']}/{row['expr']}"


def finish_rows(model_id: str, rows: list[dict[str, Any]], z0: Mapping[str, float] | None) -> list[dict[str, Any]]:
    """Add ``trial_id``, ``model`` and, when ``z0`` is given, the frozen ``z0`` and for decision rows ``payoff_help``, ``payoff_self`` and ``optimal`` (exact, see ``payoffs``)."""
    out = []
    for idx, row in enumerate(rows):
        r = dict(row)
        r["trial_id"] = f"{r['set']}:{idx:06d}"
        r["model"] = model_id
        r["unit"] = gate_unit(r)
        if z0 is not None:
            z = float(z0[r["item_key"]])
            r["z0"] = z
            if r["family"] == "mech":
                r["payoff_help"], r["payoff_self"], r["optimal"] = payoffs(r, z)
            elif r["family"] == "pask":  # the population prompt: help = a_H - c0 - w
                r["payoff_help"], r["payoff_self"], r["optimal"] = payoffs(Cond(a_h=r["a_h"], c0=r["c0"], w=r["w"]), z)
        out.append(r)
    return out


def price_extrap_feasibility(items: Sequence[Any], z0: Mapping[str, float], *, w_max_options: Sequence[float] = PRICE_EXTRAP_W_MAX, min_share: float = PRICE_EXTRAP_MIN_SHARE) -> dict[str, Any]:
    """Decided BEFORE training, from the frozen z0 only: the share of help-optimal (item, condition) pairs of ``price_extrap`` on the evaluation items; if it is below
    ``min_share`` the upper end of w is narrowed to 0.60 and then 0.50, regenerating with the same seed; if still below, the extrapolation is "not testable" for the model.
    Help-optimal means payoff_help > z0 exactly. Returns the chosen ``w_max`` (None when not testable) and every step tried."""
    tried = []
    for w_max in w_max_options:
        rows = sample_price_extrap(items, w_max)
        opt = [payoffs(r, z0[r["item_key"]])[2] for r in rows]
        n_help = sum(o == "help" for o in opt)
        share = n_help / len(opt)
        tried.append({"w_max": w_max, "n_pairs": len(opt), "n_help_optimal": n_help, "share_help_optimal": share, "n_tie": sum(o == "tie" for o in opt), "meets_min_share": bool(share >= min_share)})
        if share >= min_share:
            return {"testable": True, "w_max": w_max, "share_help_optimal": share, "min_share": min_share, "tried": tried}
    return {"testable": False, "w_max": None, "share_help_optimal": tried[-1]["share_help_optimal"], "min_share": min_share, "tried": tried}


def unseen_combo_check(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The share of help-optimal pairs of ``unseen_combo`` from the frozen z0; below 10% or above 90% marks the set "single-action" for the model."""
    n = len(rows)
    n_help = sum(r["optimal"] == "help" for r in rows)
    share = n_help / n
    lo, hi = UNSEEN_SINGLE_ACTION
    by_kind = {}
    for kind in sorted({r["group"] for r in rows}):
        sub = [r for r in rows if r["group"] == kind]
        by_kind[kind] = {"n": len(sub), "share_help_optimal": sum(r["optimal"] == "help" for r in sub) / len(sub)}
    return {"n_pairs": n, "n_help_optimal": n_help, "share_help_optimal": share, "single_action": bool(share < lo or share > hi), "bounds": [lo, hi], "by_kind": by_kind}


def plan_for_set(model_id: str, set_name: str, items: Mapping[str, Sequence[Any]], z0: Mapping[str, float], w_max: float = PRICE_EXTRAP_W_MAX[0]) -> list[dict[str, Any]]:
    its = items_for_set(set_name, items)
    raw = sample_set(set_name, its, z0=z0 if set_name in ("sweep", "equal_net") else None, w_max=w_max)
    return finish_rows(model_id, raw, z0)


def build_model_plans(model_id: str, items: Mapping[str, Sequence[Any]] | None = None) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Every plan of one model: ``({set: rows}, report)``. Rows carry z0, payoffs and trial ids; ``report`` holds the price-extrapolation feasibility, the single-action check of
    ``unseen_combo``, the tie points of ``sweep`` and the z0 provenance. The price-extrapolation range is fixed here, before any training."""
    items = items if items is not None else load_items()
    z0, z0_info = z0_all(model_id)
    feas = price_extrap_feasibility(items_for_set("in_dist", items), z0)
    w_max = feas["w_max"] if feas["testable"] else PRICE_EXTRAP_W_MAX[0]  # CHOICE: a not-testable model keeps the first range (0.40, 0.70] for descriptive use
    plans = {name: plan_for_set(model_id, name, items, z0, w_max) for name in ALL_SETS}
    sweep_items = items_for_set("sweep", items)
    report = {
        "model": model_id, "price_extrap": {**feas, "w_max_used_in_plan": w_max}, "unseen_combo": unseen_combo_check(plans["unseen_combo"]), "z0": z0_info,
        "sweep": {
            "n_items": len(sweep_items), "n_trials": len(plans["sweep"]), "n_tie_points": int(sum(r["tie_point"] for r in plans["sweep"])), "unbuildable": 0,
            "feasible_pairs_per_item_min": min(len(sweep_pairs(r_cents(z0[i.item_key]))) for i in sweep_items),
        },
        "equal_net": {"n_items": len(items_for_set("equal_net", items)), "n_trials": len(plans["equal_net"]), "unbuildable": 0},
    }
    return plans, report


# ---------------------------------------------------------------------------------------
# plan files and the text of a plan row
# ---------------------------------------------------------------------------------------

def plan_text_and_prefix(item: Any, row: Mapping[str, Any]) -> tuple[str, str]:
    """``(user_text, assistant_prefix)`` of a plan row. mech: ``decision_prompt`` and the prefix ``"Answer: "``; pask (population): the operating-curve P-ask prompt and NO prefix; panswer: the Kumaran
    prompt and the prefix ``"Answer: "``."""
    fam = row["family"]
    if fam == "mech":
        return decision_prompt(item, row)["text"], ASSISTANT_PREFIX
    if fam == "pask":
        return population_prompt(item, row["w"]), ""
    if fam == "panswer":
        return panswer_prompt(item, row.get("domain", "id")), ASSISTANT_PREFIX
    raise ValueError(f"unknown family {fam!r}")


def labels_of_row(row: Mapping[str, Any]) -> tuple[tuple[str, ...], str]:
    """The answer digits and the help digit of a plan row, recomputed from the item key and the mapping (and checked against the stored ones by the callers)."""
    if row["family"] == "mech":
        return label_help(row["item_key"], int(row["mapping_id"]))
    return ("1", "2", "3", "4"), "5"


def _row_line(row: Mapping[str, Any]) -> str:
    return json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def write_plan_file(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Write one plan as JSONL (rows in trial order) and return ``{"path", "n", "sha256"}``."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = ("\n".join(_row_line(r) for r in rows) + "\n").encode()
    tmp = p.with_name(p.name + ".partial")
    tmp.write_bytes(data)
    tmp.replace(p)
    return {"path": str(p), "n": len(rows), "sha256": hashlib.sha256(data).hexdigest()}


def read_plan_file(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def plan_path(plans_dir: str | Path, model_id: str, set_name: str) -> Path:
    return Path(plans_dir) / model_id / f"{set_name}.jsonl"


def load_plan(plans_dir: str | Path, model_id: str, set_name: str) -> list[dict[str, Any]]:
    """The frozen plan rows of a set (``<plans_dir>/<model>/<set>.jsonl``, written by ``run_part3_b_eval.py build``)."""
    return read_plan_file(plan_path(plans_dir, model_id, set_name))
