"""Prompt family of the mechanism group M (``configs/prompts/mech_v1.yaml``).

Eight variants of the P-ask user message, all in the standard layout of ``ask_v1_1.yaml`` (rules, blank line, ``Question:`` and the four options, blank line, the
"Current conditions" block, blank line, ``Answer:``; the layout string is that of the prompt book, not a copy):

* ``M1`` (the deduction falls on passing the question), ``M1w`` (the same, said as a waiting cost), ``M2`` (the deduction falls on answering yourself), ``M3`` (on answering yourself and
  being wrong), ``M0`` (a reply delay of ``D = round(100 x)`` seconds, no deduction) and ``M0n`` (no deduction and no delay: the one variant without a level). These use the rules paragraph
  ``rules_m`` of the file and a block template of the file.
* ``F3`` (the net expected score of passing, worked out in the prompt) uses the rules paragraph ``a00`` of ``ask_v1_1.yaml`` and a block template of the file; ``OBJ`` (the objective
  note) uses ``a00``, the block ``B1_f2`` as ``PromptBook.conditions_block`` renders it (a_H 80%, c0 0.00, w = x) and, as one more line, ``interventions.objective_note``. What these two take from
  ``ask_v1_1.yaml`` is named in ``v1_1_references`` of the file and read from the prompt book that is passed in: nothing of it is copied.

Every placeholder is validated against the pre-specified set (``{x:.2f}`` for M1, M1w, M2, M3; ``{D}`` for M0; none for M0n; ``{x:.2f}`` and ``{ev:.2f}`` for F3), the keys of the file are
exactly the expected ones, and the prompt book must be the file the metadata pins (its md5). A level ``x`` must be shown by two decimals without change (the rule of ``c0`` and ``kappa`` in
``nmi.prompts``) and within the display rule of the book (the cap of ``w``); ``ev = a_H - c0 - x`` is worked out in decimal arithmetic and shown with an ASCII minus sign when negative
(``-0.70``). ``M0n`` has no level: its ``x`` is ``None``.

``user_message`` is the user message before the chat template; the entry (``scripts/run_mechanism.py``) renders it with ``nmi.prompts.render_chat`` like the P-ask of the decision grid.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from string import Formatter
from types import MappingProxyType
from typing import Any

import numpy as np
import yaml

from nmi.paths import REPO_ROOT
from nmi.prompts import PromptBook, PromptError, display_a_H_pct, quantize_w, two_decimals

DEFAULT_MECH_PROMPTS = REPO_ROOT / "configs" / "prompts" / "mech_v1.yaml"

# the variants, in a fixed order; M0n has no level; the others are run at every level
VARIANTS: tuple[str, ...] = ("M1", "M1w", "M2", "M3", "M0", "M0n", "F3", "OBJ")
BLOCK_VARIANTS: tuple[str, ...] = ("M1", "M1w", "M2", "M3", "M0", "M0n", "F3")  # the variants whose conditions block is a template of the file
RULES_M_VARIANTS: tuple[str, ...] = ("M1", "M1w", "M2", "M3", "M0", "M0n")  # the variants that use the rules paragraph rules_m (F3 and OBJ use a00)
LEVEL_FREE: frozenset[str] = frozenset({"M0n"})
SHOWS_W: frozenset[str] = frozenset({"M1", "M1w", "F3", "OBJ"})  # the variants that show the level as a waiting cost or deduction on the expert side (the column w_displayed)
SHOWS_DELAY: frozenset[str] = frozenset({"M0"})  # the variant that shows the delay D in seconds (the column D_hat_s)
DELAY_RULE = "D = round(100 * x) seconds"

HEADER = "Current conditions:"
EXPECTED_FIELDS: Mapping[str, frozenset[tuple[str, str]]] = MappingProxyType({
    "M1": frozenset({("x", ".2f")}), "M1w": frozenset({("x", ".2f")}), "M2": frozenset({("x", ".2f")}), "M3": frozenset({("x", ".2f")}),
    "M0": frozenset({("D", "")}), "M0n": frozenset(), "F3": frozenset({("x", ".2f"), ("ev", ".2f")}),
})
EXPECTED_REFERENCES: Mapping[str, Mapping[str, str]] = MappingProxyType({
    "F3": MappingProxyType({"rules": "a00"}),
    "OBJ": MappingProxyType({"rules": "a00", "block": "B1_f2", "append": "objective_note"}),
})
_TOP_KEYS = frozenset({"meta", "rules_m", "condition_blocks", "v1_1_references"})
_META_KEYS = frozenset({"version", "date", "design", "levels", "a_H", "c0", "base_prompts"})
_MD5_RE = re.compile(r"[0-9a-f]{32}")


# ---------------------------------------------------------------------------------------
# reading the file
# ---------------------------------------------------------------------------------------

class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that refuses a mapping with a repeated key (PyYAML keeps the last one)."""


def _construct_unique_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False) -> Any:
    loader.flatten_mapping(node)
    seen: set[Any] = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise PromptError(f"duplicate key {key!r} (line {key_node.start_mark.line + 1})")
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


def _fields(template: str) -> set[tuple[str, str]]:
    """The ``(name, format spec)`` pairs of the placeholders of a ``str.format`` template (a conversion such as ``!r`` is refused)."""
    found: set[tuple[str, str]] = set()
    try:
        for _, name, spec, conversion in Formatter().parse(template):
            if name is None:
                continue
            if conversion:
                raise PromptError(f"conversion !{conversion} is not allowed in a prompt template")
            found.add((name, spec or ""))
    except PromptError:
        raise
    except ValueError as error:  # an unmatched brace
        raise PromptError(f"malformed format string: {error}") from error
    return found


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_text(label: str, text: Any) -> str:
    """A prompt text: a non-empty string of printable ASCII (the line breaks of a block aside), without white space at either end of the text or of a line."""
    if not isinstance(text, str) or not text.strip():
        raise PromptError(f"{label}: must be a non-empty string")
    if not text.isascii() or any(ord(c) < 32 and c != "\n" for c in text) or "\x7f" in text:
        raise PromptError(f"{label}: must be printable ASCII (a line break is the only control character allowed)")
    if text != text.strip() or any(line != line.rstrip() for line in text.split("\n")):
        raise PromptError(f"{label}: must not have white space at the end of the text or of a line (use |- for a block)")
    return text


@dataclass(frozen=True)
class Condition:
    """One condition of the group: a variant and its level ``x`` (``None`` for ``M0n``, which has none)."""

    variant: str
    x: float | None

    def key(self) -> tuple[str, float | None]:
        return (self.variant, None if self.x is None else round(float(self.x), 9))

    def label(self) -> str:
        return self.variant if self.x is None else f"{self.variant}@{self.x:.2f}"


@dataclass(frozen=True)
class MechPromptBook:
    """Validated content of ``mech_v1.yaml`` and the rendering functions; ``base`` is the prompt book (``ask_v1_1.yaml``) that ``v1_1_references`` is resolved in."""

    path: Path
    md5: str
    version: str
    date: str
    design: str
    levels: tuple[float, ...]
    a_H: float
    c0: float
    rules_m: str
    blocks: Mapping[str, str]
    references: Mapping[str, Mapping[str, str]]
    base: PromptBook
    base_md5: str  # the md5 that ``meta.base_prompts`` pins (``base.md5`` is the same)

    # ---- loading -----------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path = DEFAULT_MECH_PROMPTS, base: PromptBook | None = None) -> MechPromptBook:
        """Read and validate the file. ``base`` is the prompt book of the configuration of the run (its md5 must be the one the file pins); without it the file named in
        ``meta.base_prompts`` is loaded from the repository with the default display rule."""
        path = Path(path)
        data = path.read_bytes()
        name = path.name
        try:
            raw = yaml.load(data, Loader=_UniqueKeyLoader)  # noqa: S506
        except yaml.YAMLError as error:
            raise PromptError(f"{name}: not valid YAML: {error}") from error
        if not isinstance(raw, dict) or set(raw) != _TOP_KEYS:
            keys = sorted(raw) if isinstance(raw, dict) else type(raw).__name__
            raise PromptError(f"{name}: top-level keys {keys} differ from {sorted(_TOP_KEYS)}")
        meta = cls._check_meta(name, raw["meta"])
        a_H, c0 = float(meta["a_H"]), float(meta["c0"])
        rules_m = _check_text(f"{name}: rules_m", raw["rules_m"])
        if "\n" in rules_m:
            raise PromptError(f"{name}: rules_m must be one paragraph (use >- for a folded block)")
        blocks = cls._check_blocks(name, raw["condition_blocks"], a_H, c0)
        references = cls._check_references(name, raw["v1_1_references"])
        pinned = meta["base_prompts"]
        if base is None:
            base = PromptBook.load(REPO_ROOT / pinned["file"])
        if base.md5 != pinned["md5"]:
            raise PromptError(
                f"{name}: the prompt book {base.path.name} has md5 {base.md5}, the file pins {pinned['md5']} ({pinned['file']}): the mechanism group is built on that file"
            )
        for variant, ref in references.items():
            if ref["rules"] not in base.paraphrases:
                raise PromptError(f"{name}: v1_1_references.{variant}.rules {ref['rules']!r} is not a paraphrase of {base.path.name}")
            if "block" in ref and ref["block"] not in base.condition_blocks:
                raise PromptError(f"{name}: v1_1_references.{variant}.block {ref['block']!r} is not a condition block of {base.path.name}")
            if "append" in ref and ref["append"] not in base.interventions:
                raise PromptError(f"{name}: v1_1_references.{variant}.append {ref['append']!r} is not an intervention of {base.path.name}")
        return cls(
            path=path, md5=hashlib.md5(data, usedforsecurity=False).hexdigest(), version=meta["version"], date=meta["date"], design=meta["design"], levels=tuple(meta["levels"]),
            a_H=a_H, c0=c0, rules_m=rules_m, blocks=MappingProxyType(blocks), references=MappingProxyType({k: MappingProxyType(dict(v)) for k, v in references.items()}),
            base=base, base_md5=pinned["md5"],
        )

    @staticmethod
    def _check_meta(name: str, meta: Any) -> dict[str, Any]:
        if not isinstance(meta, dict) or set(meta) != _META_KEYS:
            keys = sorted(meta) if isinstance(meta, dict) else type(meta).__name__
            raise PromptError(f"{name}: meta keys {keys} differ from {sorted(_META_KEYS)}")
        for key in ("version", "date", "design"):
            if not isinstance(meta[key], str) or not meta[key].strip():
                raise PromptError(f"{name}: meta.{key} must be a non-empty string")
        levels = meta["levels"]
        if not isinstance(levels, list) or not levels or not all(_is_number(v) for v in levels):
            raise PromptError(f"{name}: meta.levels must be a non-empty list of numbers")
        try:
            levels = [two_decimals(v, "meta.levels") + 0.0 for v in levels]  # + 0.0: a negative zero would be shown as -0.00
        except PromptError as error:
            raise PromptError(f"{name}: {error}") from error
        if any(b <= a for a, b in zip(levels, levels[1:], strict=False)):
            raise PromptError(f"{name}: meta.levels must increase strictly, got {levels}")
        if not _is_number(meta["a_H"]) or not 0.0 < float(meta["a_H"]) <= 1.0 or abs(float(meta["a_H"]) * 100 - round(float(meta["a_H"]) * 100)) > 1e-9:
            raise PromptError(f"{name}: meta.a_H must be in (0, 1] and a whole percent, got {meta['a_H']!r}")
        if not _is_number(meta["c0"]):
            raise PromptError(f"{name}: meta.c0 must be a number, got {meta['c0']!r}")
        try:
            two_decimals(meta["c0"], "meta.c0")
        except PromptError as error:
            raise PromptError(f"{name}: {error}") from error
        base = meta["base_prompts"]
        if not isinstance(base, dict) or set(base) != {"file", "md5"} or not isinstance(base["file"], str) or not base["file"].strip():
            raise PromptError(f"{name}: meta.base_prompts must have exactly file and md5")
        if not isinstance(base["md5"], str) or not _MD5_RE.fullmatch(base["md5"]):
            raise PromptError(f"{name}: meta.base_prompts.md5 must be 32 lower-case hex digits, got {base['md5']!r}")
        return {**meta, "levels": levels}

    @staticmethod
    def _check_blocks(name: str, blocks: Any, a_H: float, c0: float) -> dict[str, str]:
        if not isinstance(blocks, dict) or set(blocks) != set(BLOCK_VARIANTS):
            keys = sorted(blocks) if isinstance(blocks, dict) else type(blocks).__name__
            raise PromptError(f"{name}: condition_blocks keys {keys} differ from {sorted(BLOCK_VARIANTS)}")
        accuracy_line = f"- The expert answers correctly {display_a_H_pct(a_H)}% of the time."
        out: dict[str, str] = {}
        for variant in BLOCK_VARIANTS:
            label = f"{name}: condition_blocks.{variant}"
            text = _check_text(label, blocks[variant])
            lines = text.split("\n")
            if lines[0] != HEADER or len(lines) < 2:
                raise PromptError(f"{label}: must start with the line {HEADER!r} and have at least one condition line")
            if any(not line.startswith("- ") for line in lines[1:]):
                raise PromptError(f"{label}: every line after the header must be a bullet starting with '- '")
            try:
                got = _fields(text)
            except PromptError as error:
                raise PromptError(f"{label}: {error}") from error
            if got != EXPECTED_FIELDS[variant]:
                raise PromptError(f"{label}: placeholders {sorted(got)} differ from the expected {sorted(EXPECTED_FIELDS[variant])}")
            if variant == "F3":
                if accuracy_line in lines:
                    raise PromptError(f"{label}: F3 does not use the accuracy line (it states the accuracy in its second line)")
                # the arithmetic that the block shows must be the one ev is worked out with (a_H, c0 of the metadata)
                shown = f"is {a_H:.2f} - {c0:.2f} - " + "{x:.2f} = {ev:.2f} points."
                if shown not in text or f"the fixed fee is {c0:.2f} points" not in text:
                    raise PromptError(f"{label}: the text must show {shown!r} and the fixed fee {c0:.2f} (a_H and c0 of meta)")
            elif lines[1] != accuracy_line:
                raise PromptError(f"{label}: the first line after the header must be {accuracy_line!r}, got {lines[1]!r}")
            out[variant] = text
        return out

    @staticmethod
    def _check_references(name: str, references: Any) -> dict[str, dict[str, str]]:
        expected = {k: dict(v) for k, v in EXPECTED_REFERENCES.items()}
        if references != expected:
            raise PromptError(f"{name}: v1_1_references {references!r} differ from the expected {expected!r}")
        return expected

    # ---- the variants and their conditions ---------------------------------------------
    @property
    def variants(self) -> tuple[str, ...]:
        return VARIANTS

    def levels_of(self, variant: str) -> tuple[float | None, ...]:
        """The levels a variant is run at: the levels of the file, or ``(None,)`` for ``M0n``, which has one condition and no level."""
        self._check_variant(variant)
        return (None,) if variant in LEVEL_FREE else self.levels

    def conditions(self, variants: Sequence[str] | None = None) -> list[Condition]:
        """The conditions of the chosen variants (all by default) in the order of ``VARIANTS`` and, within a variant, of the levels of the file."""
        chosen = list(VARIANTS if variants is None else variants)
        for variant in chosen:
            self._check_variant(variant)
        if len(set(chosen)) != len(chosen):
            raise PromptError(f"a variant is listed twice: {chosen}")
        return [Condition(v, x) for v in VARIANTS if v in chosen for x in self.levels_of(v)]

    @staticmethod
    def _check_variant(variant: str) -> None:
        if variant not in VARIANTS:
            raise PromptError(f"unknown variant {variant!r}; known: {list(VARIANTS)}")

    def _level(self, variant: str, x: Any) -> float | None:
        """The level ``x`` of a variant as it is shown: ``None`` for ``M0n`` (which takes none), else a number that two decimals show without change and the display rule of the base book keeps."""
        self._check_variant(variant)
        if variant in LEVEL_FREE:
            if x is not None:
                raise PromptError(f"variant {variant} has no level: x must be None, got {x!r}")
            return None
        if x is None:
            raise PromptError(f"variant {variant} needs a level x")
        if not isinstance(x, (int, float, np.integer, np.floating)) or isinstance(x, (bool, np.bool_)):
            raise PromptError(f"x must be a number, got {x!r}")
        value = two_decimals(x, "x") + 0.0  # + 0.0: a negative zero would be shown as -0.00
        if abs(quantize_w(value, self.base.rule) - value) > 1e-9:
            raise PromptError(f"x={x!r} is not shown as it is by the display rule of the base prompts (quantum {self.base.rule.quantum}, cap {self.base.rule.cap})")
        return value

    # ---- what a condition shows ---------------------------------------------------------
    def delay_seconds(self, x: float) -> int:
        """The delay of ``M0`` in whole seconds, ``D = round(100 x)`` (half up, on the decimal form of ``x``)."""
        value = self._level("M0", x)
        assert value is not None
        return int((Decimal(repr(value)) * 100).quantize(Decimal(1), rounding=ROUND_HALF_UP))

    def w_displayed(self, variant: str, x: float | None) -> float:
        """The waiting cost or deduction on the expert side that the prompt shows (M1, M1w, F3 and OBJ: ``x``); NaN for the variants that show none (M2, M3, M0, M0n)."""
        level = self._level(variant, x)
        return float(level) if variant in SHOWS_W and level is not None else math.nan

    def delay_shown(self, variant: str, x: float | None) -> int | None:
        """The delay in seconds that the prompt shows (M0 only), else ``None``."""
        level = self._level(variant, x)
        return self.delay_seconds(level) if variant in SHOWS_DELAY and level is not None else None

    def _ev(self, x: float) -> Decimal:
        """``a_H - c0 - x`` in decimal arithmetic (so that no rounding error can show as -0.00 or 0.35000000000000003)."""
        return Decimal(repr(self.a_H)) - Decimal(repr(self.c0)) - Decimal(repr(x))

    # ---- rendering ---------------------------------------------------------------------
    def rules_of(self, variant: str) -> str:
        """The rules paragraph of a variant: ``rules_m`` for M1 to M0n, the paraphrase named in ``v1_1_references`` of the base book for F3 and OBJ."""
        self._check_variant(variant)
        if variant in RULES_M_VARIANTS:
            return self.rules_m
        return self.base.paraphrases[self.references[variant]["rules"]].rules

    def conditions_block(self, variant: str, x: float | None) -> str:
        """The "Current conditions" block of a condition."""
        level = self._level(variant, x)
        if variant == "OBJ":
            ref = self.references["OBJ"]
            block, framing = ref["block"].split("_")
            shown = self.base.conditions_block(block, framing, a_H=self.a_H, c0=self.c0, w=level)
            return shown + "\n" + self.base.intervention_text(ref["append"])
        assert variant in self.blocks
        if variant == "M0n":
            return self.blocks[variant]
        if variant == "M0":
            return self.blocks[variant].format(D=self.delay_seconds(level))  # type: ignore[arg-type]
        if variant == "F3":
            return self.blocks[variant].format(x=level, ev=self._ev(level))  # type: ignore[arg-type]
        return self.blocks[variant].format(x=level)

    def user_message(self, variant: str, x: float | None, question: str, choices: Sequence[str]) -> str:
        """The P-ask user message of a condition for one question and its four options (before the chat template)."""
        if not isinstance(question, str):
            raise PromptError("the question must be a string")
        if isinstance(choices, str) or len(choices) != 4 or not all(isinstance(o, str) for o in choices):
            raise PromptError("exactly four string options are required")
        return self.base.ask_layout.format(
            rules=self.rules_of(variant), question=question, conditions=self.conditions_block(variant, x),
            o1=choices[0], o2=choices[1], o3=choices[2], o4=choices[3],
        )
