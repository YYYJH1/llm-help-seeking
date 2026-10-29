"""Prompt family of ``configs/prompts/ask_v1_1.yaml`` and the display rules.

* P-answer is Kumaran's Phase 0/1 prompt, character for character. An END-SIDE model reads it
  at the position after a fixed assistant prefix (``answer_kumaran.assistant_prefix`` of ``ask_v1_1.yaml``, ``"Answer: "``): the rendered chat text of the
  prompt, which ends at the start of the assistant turn, gets the prefix appended (``PromptBook.answer_chat_text``). The user message is the same
  with or without it, and the helper's direct answer never gets the prefix. A prompt file without ``assistant_prefix`` has the empty prefix.
* P-ask = rules (one of ten paraphrases) + question and options + a "Current conditions"
  block + ``Answer:``. Two layouts: the standard one and the exploratory variant that puts
  the conditions block before the question. Two framings: F2 shows the waiting cost ``w``
  directly, F1 shows the expected wait ``D_hat`` in seconds and the cost per second ``kappa``.
* The whole prompt goes to the model as ONE user message through the chat template.

Display rules (the same in measurement, DES, fluid model and real loop):

* ``a_H`` is shown as an integer percent;
* ``c0`` and ``kappa`` are shown with two decimals (a value that two decimals cannot
  represent is an error, not a silent rounding);
* ``w`` is quantised to 0.01, then capped at 5.0, then shown with two decimals;
* ``D_hat`` is shown as integer seconds, rounded half up.

``quantize_w`` is the single implementation of the ``w`` rule; the simulators import it.
Ties (a value exactly half way between two lattice points, for example kappa 0.0025 times
6 s = 0.015) round up. Floating point makes such values land a few ulps below the tie, so
the rule adds 1e-9 of a quantum before taking the floor; the scalar and the array path use
the same arithmetic and agree element by element.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from string import Formatter
from typing import Any

import numpy as np
import yaml

from nmi.paths import DEFAULT_PROMPTS


class PromptError(ValueError):
    """A prompt file or a prompt argument is invalid."""


# ---------------------------------------------------------------------------------------
# display rules
# ---------------------------------------------------------------------------------------

_TIE_EPS = 1e-9  # in units of one quantum


@dataclass(frozen=True)
class DisplayRule:
    """Quantum and cap of the displayed waiting cost (config ``grid.displayed_w``)."""

    quantum: float = 0.01
    cap: float = 5.0

    def __post_init__(self) -> None:
        if not (self.quantum > 0 and self.cap > 0):
            raise PromptError("quantum and cap must be positive")
        scale = round(1.0 / self.quantum)
        if scale < 1 or abs(scale * self.quantum - 1.0) > 1e-9:
            raise PromptError(f"quantum {self.quantum} must be the reciprocal of an integer")
        cap_index = self.cap * scale
        if abs(cap_index - round(cap_index)) > 1e-9:
            raise PromptError(f"cap {self.cap} must be a whole number of quanta")

    @property
    def scale(self) -> int:
        return round(1.0 / self.quantum)

    @property
    def cap_index(self) -> int:
        return round(self.cap * self.scale)

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> DisplayRule:
        d = cfg["grid"]["displayed_w"]
        return cls(quantum=float(d["quantum"]), cap=float(d["cap"]))


DEFAULT_RULE = DisplayRule()


def quantize_w_index(w: Any, rule: DisplayRule = DEFAULT_RULE) -> Any:
    """Lattice index of the displayed ``w``: ``floor(w / quantum + 1/2)``, capped.

    Accepts a scalar (returns ``int``) or an array (returns an ``int64`` array). ``w`` must
    be finite and non-negative.
    """
    arr = np.asarray(w, dtype=np.float64)
    if not np.all(np.isfinite(arr)) or np.any(arr < 0):
        raise PromptError("w must be finite and non-negative")
    index = np.floor(arr * rule.scale + 0.5 + _TIE_EPS)
    index = np.minimum(index, rule.cap_index).astype(np.int64)
    return int(index) if index.ndim == 0 else index


def quantize_w(w: Any, rule: DisplayRule = DEFAULT_RULE) -> Any:
    """Displayed ``w``: quantised, then capped (a float, or an array of floats)."""
    index = quantize_w_index(w, rule)
    value = np.asarray(index, dtype=np.float64) / rule.scale
    return float(value) if value.ndim == 0 else value


def display_a_H_pct(a_H: float) -> int:
    """``a_H`` as an integer percent, rounded half up on its decimal representation."""
    if not (isinstance(a_H, (int, float, np.floating)) and 0.0 <= float(a_H) <= 1.0):
        raise PromptError(f"a_H must be in [0, 1], got {a_H!r}")
    pct = Decimal(repr(float(a_H))) * 100
    return int(pct.quantize(Decimal(1), rounding=ROUND_HALF_UP))


def display_D_hat_s(D_hat: float) -> int:
    """Expected wait as integer seconds, rounded half up (2.5 shows as 3)."""
    value = float(D_hat)
    if not math.isfinite(value) or value < 0:
        raise PromptError(f"D_hat must be finite and non-negative, got {D_hat!r}")
    return int(Decimal(repr(value)).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def two_decimals(value: float, name: str) -> float:
    """``c0`` and ``kappa`` are shown with two decimals; refuse a value they cannot represent."""
    x = float(value)
    if not math.isfinite(x) or x < 0:
        raise PromptError(f"{name} must be finite and non-negative, got {value!r}")
    if abs(round(x, 2) - x) > 1e-9:
        raise PromptError(f"{name}={value!r} cannot be shown with two decimals without changing it")
    return x


# ---------------------------------------------------------------------------------------
# the prompt book
# ---------------------------------------------------------------------------------------

_OPTION_FIELDS = {("o1", ""), ("o2", ""), ("o3", ""), ("o4", "")}
_QA_FIELDS = {("question", "")} | _OPTION_FIELDS
_LAYOUT_FIELDS = {("rules", ""), ("conditions", "")} | _QA_FIELDS
_F2_FIELDS = {("a_H_pct", ""), ("c0", ".2f"), ("w", ".2f")}
_F1_FIELDS = {("a_H_pct", ""), ("c0", ".2f"), ("D_hat_s", ""), ("kappa", ".2f")}
_BLOCK_NAMES = ("B1", "B2")
_FRAMINGS = ("f1", "f2")
LAYOUTS = ("standard", "block_position")
EXPECTED_PARAPHRASES = tuple(f"a{i:02d}" for i in range(10))


def _fields(template: str) -> set[tuple[str, str]]:
    found = set()
    for _, name, spec, conversion in Formatter().parse(template):
        if name is None:
            continue
        if conversion:
            raise PromptError(f"conversion !{conversion} is not allowed in a prompt template")
        found.add((name, spec or ""))
    return found


def _require_fields(label: str, template: Any, expected: set[tuple[str, str]]) -> str:
    if not isinstance(template, str) or not template.strip():
        raise PromptError(f"{label}: template must be a non-empty string")
    got = _fields(template)
    if got != expected:
        raise PromptError(
            f"{label}: placeholders {sorted(got)} differ from the expected {sorted(expected)}"
        )
    return template


@dataclass(frozen=True)
class Paraphrase:
    id: str
    block: str  # "B1" (bulleted conditions) or "B2" (plain lines)
    rules: str


@dataclass(frozen=True)
class PromptBook:
    """Validated content of a prompt file such as ``ask_v1_1.yaml`` plus the rendering functions."""

    path: Path
    md5: str
    answer_template: str
    helper_template: str
    ask_layout: str
    block_position_layout: str
    condition_blocks: Mapping[str, str]
    paraphrases: Mapping[str, Paraphrase]
    interventions: Mapping[str, str]
    rule: DisplayRule
    assistant_prefix: str = ""  # text appended to the rendered chat text of the END-SIDE P-answer; "" when the prompt file has none

    # ---- loading -----------------------------------------------------------------------
    @classmethod
    def load(
        cls, path: str | Path = DEFAULT_PROMPTS, rule: DisplayRule = DEFAULT_RULE
    ) -> PromptBook:
        path = Path(path)
        data = path.read_bytes()
        raw = yaml.safe_load(data)
        if not isinstance(raw, dict):
            raise PromptError(f"{path.name}: top level must be a mapping")
        expected_top = {
            "answer_kumaran", "helper_cot", "ask_layout", "condition_blocks", "ask_paraphrases",
            "interventions", "block_position_variant_layout",
        }
        if set(raw) != expected_top:
            raise PromptError(f"{path.name}: top-level keys {sorted(raw)} differ from {sorted(expected_top)}")
        answer_raw = raw["answer_kumaran"]
        if not isinstance(answer_raw, dict) or not set(answer_raw) <= {"template", "assistant_prefix"}:
            raise PromptError("answer_kumaran must be a mapping with template and, optionally, assistant_prefix")
        answer = _require_fields("answer_kumaran.template", answer_raw.get("template"), _QA_FIELDS)
        prefix = answer_raw.get("assistant_prefix", "")
        if not isinstance(prefix, str) or not prefix.isascii() or any(ord(c) < 32 or ord(c) == 127 for c in prefix):
            raise PromptError("answer_kumaran.assistant_prefix must be a string of printable ASCII characters (no newline, no control character, no end marker)")
        helper = _require_fields("helper_cot.template", raw["helper_cot"].get("template"), _QA_FIELDS)
        layout = _require_fields("ask_layout", raw["ask_layout"], _LAYOUT_FIELDS)
        variant = _require_fields(
            "block_position_variant_layout", raw["block_position_variant_layout"], _LAYOUT_FIELDS
        )
        blocks_raw = raw["condition_blocks"]
        expected_blocks = {f"{b}_{f}" for b in _BLOCK_NAMES for f in _FRAMINGS}
        if not isinstance(blocks_raw, dict) or set(blocks_raw) != expected_blocks:
            raise PromptError(f"condition_blocks must have exactly the keys {sorted(expected_blocks)}")
        blocks = {
            name: _require_fields(
                f"condition_blocks.{name}", text, _F2_FIELDS if name.endswith("_f2") else _F1_FIELDS
            )
            for name, text in blocks_raw.items()
        }
        paraphrases_raw = raw["ask_paraphrases"]
        if not isinstance(paraphrases_raw, dict) or tuple(sorted(paraphrases_raw)) != EXPECTED_PARAPHRASES:
            raise PromptError(f"ask_paraphrases must be exactly {list(EXPECTED_PARAPHRASES)}")
        paraphrases = {}
        for pid in EXPECTED_PARAPHRASES:
            entry = paraphrases_raw[pid]
            if not isinstance(entry, dict) or set(entry) != {"block", "rules"}:
                raise PromptError(f"ask_paraphrases.{pid} must have exactly block and rules")
            if entry["block"] not in _BLOCK_NAMES:
                raise PromptError(f"ask_paraphrases.{pid}.block must be one of {_BLOCK_NAMES}")
            if not isinstance(entry["rules"], str) or not entry["rules"].strip():
                raise PromptError(f"ask_paraphrases.{pid}.rules must be a non-empty string")
            paraphrases[pid] = Paraphrase(pid, entry["block"], entry["rules"])
        interventions = raw["interventions"]
        if not isinstance(interventions, dict) or set(interventions) != {"externality_note", "objective_note"}:
            raise PromptError("interventions must have exactly externality_note and objective_note")
        return cls(
            path=path,
            md5=hashlib.md5(data, usedforsecurity=False).hexdigest(),
            answer_template=answer,
            helper_template=helper,
            ask_layout=layout,
            block_position_layout=variant,
            condition_blocks=blocks,
            paraphrases=paraphrases,
            interventions={k: str(v) for k, v in interventions.items()},
            rule=rule,
            assistant_prefix=prefix,
        )

    @property
    def paraphrase_ids(self) -> tuple[str, ...]:
        return tuple(self.paraphrases)

    # ---- rendering ---------------------------------------------------------------------
    @staticmethod
    def _options(options: Sequence[str]) -> dict[str, str]:
        if len(options) != 4 or not all(isinstance(o, str) for o in options):
            raise PromptError("exactly four string options are required")
        return {"o1": options[0], "o2": options[1], "o3": options[2], "o4": options[3]}

    def answer_prompt(self, question: str, options: Sequence[str]) -> str:
        """P-answer: Kumaran's Phase 0/1 prompt."""
        return self.answer_template.format(question=question, **self._options(options))

    def answer_chat_text(
        self, tokenizer: Any, question: str, options: Sequence[str], chat_template_kwargs: Mapping[str, Any] | None = None
    ) -> str:
        """The P-answer of an END-SIDE model as the model reads it: Kumaran's prompt as one user message through the chat template, then the
        assistant prefix of this prompt file (``"Answer: "`` in ``ask_v1_1.yaml``, empty when the file has none). The digit is read at the position after the returned text.
        The helper's direct answer is NOT built with this function: it renders ``answer_prompt`` with ``render_chat`` and no prefix."""
        return render_chat(
            tokenizer, self.answer_prompt(question, options), chat_template_kwargs, assistant_prefix=self.assistant_prefix
        )

    def helper_prompt(self, question: str, options: Sequence[str]) -> str:
        """Prompt of the helper (expert) model: brief reasoning, then ``Answer: N``."""
        return self.helper_template.format(question=question, **self._options(options))

    def conditions_block(
        self,
        block: str,
        framing: str,
        *,
        a_H: float,
        c0: float,
        w: float | None = None,
        D_hat_s: float | None = None,
        kappa: float | None = None,
    ) -> str:
        """The "Current conditions" text. F2 needs ``w``; F1 needs ``D_hat_s`` and ``kappa``."""
        if block not in _BLOCK_NAMES:
            raise PromptError(f"block must be one of {_BLOCK_NAMES}, got {block!r}")
        if framing not in _FRAMINGS:
            raise PromptError(f"framing must be one of {_FRAMINGS}, got {framing!r}")
        template = self.condition_blocks[f"{block}_{framing}"]
        common = {"a_H_pct": display_a_H_pct(a_H), "c0": two_decimals(c0, "c0")}
        if framing == "f2":
            if w is None or D_hat_s is not None or kappa is not None:
                raise PromptError("framing f2 takes w, and neither D_hat_s nor kappa")
            return template.format(w=quantize_w(w, self.rule), **common)
        if D_hat_s is None or kappa is None or w is not None:
            raise PromptError("framing f1 takes D_hat_s and kappa, and not w")
        return template.format(
            D_hat_s=display_D_hat_s(D_hat_s), kappa=two_decimals(kappa, "kappa"), **common
        )

    def ask_prompt(
        self,
        paraphrase: str,
        question: str,
        options: Sequence[str],
        *,
        framing: str,
        a_H: float,
        c0: float,
        w: float | None = None,
        D_hat_s: float | None = None,
        kappa: float | None = None,
        layout: str = "standard",
    ) -> str:
        """P-ask for one paraphrase, framing and layout (the user message, before the chat template)."""
        if paraphrase not in self.paraphrases:
            raise PromptError(f"unknown paraphrase {paraphrase!r}")
        if layout not in LAYOUTS:
            raise PromptError(f"layout must be one of {LAYOUTS}, got {layout!r}")
        para = self.paraphrases[paraphrase]
        conditions = self.conditions_block(
            para.block, framing, a_H=a_H, c0=c0, w=w, D_hat_s=D_hat_s, kappa=kappa
        )
        template = self.ask_layout if layout == "standard" else self.block_position_layout
        return template.format(
            rules=para.rules, question=question, conditions=conditions, **self._options(options)
        )

    def intervention_text(self, name: str) -> str:
        """Wording of an intervention (main phase only; never used in the kill test)."""
        try:
            return self.interventions[name]
        except KeyError as error:
            raise PromptError(f"unknown intervention {name!r}") from error


def render_chat(
    tokenizer: Any,
    user_text: str,
    chat_template_kwargs: Mapping[str, Any] | None = None,
    *,
    assistant_prefix: str = "",
) -> str:
    """The prompt as the model sees it: one user message through the chat template, ending at
    the generation prompt (Qwen3: ``enable_thinking=False`` adds an empty think block). A non-empty
    ``assistant_prefix`` is appended to that text, directly (no separator, no end marker): the model
    then continues inside its own turn after the prefix."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
        **dict(chat_template_kwargs or {}),
    )
    if not isinstance(text, str):
        raise PromptError(f"apply_chat_template returned {type(text).__name__}, expected str")
    return text + assistant_prefix
