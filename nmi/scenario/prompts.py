"""Prompts of the real scenario S1 (``configs/prompts/scenario_s1.yaml``).

The file holds the role and scoring texts, the tool schema, the follow-up of the confidence readout and the system prompts of the three groups of P-decide:

* ``S-exp`` (explicit deduction, the wait costs points), ``S-nat`` (system-state wording: 0.01 points per second in the scoring paragraph, a status block with the wait; at D = 0 the status says the server is
  idle) and ``S-loc`` (running the model on the device costs battery worth x points). Six levels each, D in {0, 10, 20, 45, 80, 150} seconds and w = x = 0.01 D: 18 conditions.
* ``answer.system`` is the system prompt of P-answer and ``expert.system`` that of the expert phase; neither mentions a tool, the expert or a deduction. ``answer.follow_up`` is the confidence question.

Every placeholder is validated against the expected set (``{a_pct}`` an integer percent, ``{D}`` whole seconds, ``{w:.2f}`` and ``{x:.2f}`` points), the keys of the file are exactly the expected ones, the
three groups share the role paragraph and the scoring sentences, and no call note (the sentence ``CALL_NOTE`` below) is present. The expert's accuracy ``a_pct`` is not part of the file: the run configuration
fixes it after the expert phase, and every rendering takes it as an argument.

A message list is built here; the rendering through a model's chat template (and the injection of the tool) is ``nmi.scenario.render``. Nothing here reads a question.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from string import Formatter
from types import MappingProxyType
from typing import Any

import yaml

from nmi.paths import REPO_ROOT
from nmi.prompts import PromptError

DEFAULT_SCENARIO_PROMPTS = REPO_ROOT / "configs" / "prompts" / "scenario_s1.yaml"

TOOL_NAME = "ask_edge_expert"
GROUPS: tuple[str, ...] = ("S-exp", "S-nat", "S-loc")
LEVELS: tuple[int, ...] = (0, 10, 20, 45, 80, 150)
POINTS_PER_SECOND = Decimal("0.01")
A_PCT_RANGE = (1, 99)

# A call-note sentence that the role paragraph must not contain; prompts that contain it are rejected.
CALL_NOTE = "To pass a question to the edge expert, call the tool ask_edge_expert"

_TOP_KEYS = frozenset({"meta", "tool", "answer", "expert", "conditions"})
_META_KEYS = frozenset({"version", "date", "design", "levels", "points_per_second"})
_TOOL_KEYS = frozenset({"name", "description", "parameter", "parameter_description"})
_CONDITION_KEYS = {"S-exp": frozenset({"system"}), "S-nat": frozenset({"system_busy", "system_idle"}), "S-loc": frozenset({"system"})}
EXPECTED_FIELDS: Mapping[str, frozenset[tuple[str, str]]] = MappingProxyType({
    "answer.system": frozenset(), "answer.follow_up": frozenset(), "expert.system": frozenset(),
    "S-exp.system": frozenset({("a_pct", ""), ("D", ""), ("w", ".2f")}),
    "S-nat.system_busy": frozenset({("a_pct", ""), ("D", "")}),
    "S-nat.system_idle": frozenset({("a_pct", "")}),
    "S-loc.system": frozenset({("a_pct", ""), ("x", ".2f")}),
})
ROLE_END = " Scoring:"  # the role paragraph of a decide prompt ends before this
SHARED_SENTENCES = ("Apply each point deduction listed ", "once, and only in the case specified.", "Your final score may be negative.")


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


def _check_text(label: str, text: Any) -> str:
    """A prompt text: a non-empty string of printable ASCII (line breaks aside) without white space at either end of the text or of a line."""
    if not isinstance(text, str) or not text.strip():
        raise PromptError(f"{label}: must be a non-empty string")
    if not text.isascii() or any(ord(c) < 32 and c != "\n" for c in text) or "\x7f" in text:
        raise PromptError(f"{label}: must be printable ASCII (a line break is the only control character allowed)")
    if text != text.strip() or any(line != line.rstrip() for line in text.split("\n")):
        raise PromptError(f"{label}: must not have white space at the end of the text or of a line (use |- for a block)")
    return text


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def check_a_pct(a_pct: Any) -> int:
    """The expert's accuracy as shown: an integer from 1 to 99 (a bool, a float or a string is refused)."""
    if isinstance(a_pct, bool) or not isinstance(a_pct, int) or not A_PCT_RANGE[0] <= a_pct <= A_PCT_RANGE[1]:
        raise PromptError(f"a_pct must be an integer from {A_PCT_RANGE[0]} to {A_PCT_RANGE[1]}, got {a_pct!r}")
    return a_pct


@dataclass(frozen=True)
class Condition:
    """One condition of P-decide: a group and its level (D in seconds for S-exp and S-nat, 100 x for S-loc)."""

    group: str
    level: int

    @property
    def label(self) -> str:
        return f"{self.group}_{'x' if self.group == 'S-loc' else 'D'}{self.level:03d}"

    @property
    def delay_s(self) -> int | None:
        """The wait that the prompt announces, in seconds (S-exp and S-nat); None for S-loc, which announces none."""
        return None if self.group == "S-loc" else self.level

    @property
    def points(self) -> Decimal:
        """The deduction of the level in points, 0.01 per unit: w for S-exp and S-nat (the cost of the wait), x for S-loc (the battery cost)."""
        return Decimal(self.level) / 100

    @property
    def acts_on(self) -> str:
        """The action the deduction falls on: ``offload`` (S-exp, S-nat) or ``self`` (S-loc)."""
        return "self" if self.group == "S-loc" else "offload"


@dataclass(frozen=True)
class ScenarioPromptBook:
    """Validated content of ``scenario_s1.yaml`` and the functions that build the messages."""

    path: Path
    md5: str
    version: str
    date: str
    design: str
    levels: tuple[int, ...]
    tool: Mapping[str, str]
    answer_system: str
    follow_up: str
    expert_system: str
    templates: Mapping[str, str]  # "S-exp", "S-nat.busy", "S-nat.idle", "S-loc" -> system template

    # ---- loading -----------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path = DEFAULT_SCENARIO_PROMPTS) -> "ScenarioPromptBook":
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
        tool = cls._check_tool(name, raw["tool"])
        answer, expert = raw["answer"], raw["expert"]
        if not isinstance(answer, dict) or set(answer) != {"system", "follow_up"}:
            raise PromptError(f"{name}: answer must have exactly system and follow_up")
        if not isinstance(expert, dict) or set(expert) != {"system"}:
            raise PromptError(f"{name}: expert must have exactly system")
        conditions = raw["conditions"]
        if not isinstance(conditions, dict) or set(conditions) != set(GROUPS):
            keys = sorted(conditions) if isinstance(conditions, dict) else type(conditions).__name__
            raise PromptError(f"{name}: conditions keys {keys} differ from {list(GROUPS)}")
        texts: dict[str, str] = {
            "answer.system": answer["system"], "answer.follow_up": answer["follow_up"], "expert.system": expert["system"],
        }
        templates: dict[str, str] = {}
        for group in GROUPS:
            entry = conditions[group]
            if not isinstance(entry, dict) or set(entry) != _CONDITION_KEYS[group]:
                got = sorted(entry) if isinstance(entry, dict) else type(entry).__name__
                raise PromptError(f"{name}: conditions.{group} keys {got} differ from {sorted(_CONDITION_KEYS[group])}")
            for key in sorted(entry):
                texts[f"{group}.{key}"] = entry[key]
        for label, text in texts.items():
            _check_text(f"{name}: {label}", text)
            got = _fields(text)
            if got != EXPECTED_FIELDS[label]:
                raise PromptError(f"{name}: {label}: placeholders {sorted(got)} differ from the expected {sorted(EXPECTED_FIELDS[label])}")
        cls._check_content(name, texts, tool["name"])
        templates = {
            "S-exp": texts["S-exp.system"], "S-nat.busy": texts["S-nat.system_busy"], "S-nat.idle": texts["S-nat.system_idle"], "S-loc": texts["S-loc.system"],
        }
        return cls(
            path=path, md5=hashlib.md5(data, usedforsecurity=False).hexdigest(), version=meta["version"], date=meta["date"], design=meta["design"], levels=tuple(meta["levels"]),
            tool=MappingProxyType(dict(tool)), answer_system=texts["answer.system"], follow_up=texts["answer.follow_up"], expert_system=texts["expert.system"],
            templates=MappingProxyType(templates),
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
        if not isinstance(levels, list) or not levels or not all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in levels):
            raise PromptError(f"{name}: meta.levels must be a non-empty list of non-negative integers (seconds)")
        if any(b <= a for a, b in zip(levels, levels[1:], strict=False)):
            raise PromptError(f"{name}: meta.levels must increase strictly, got {levels}")
        if list(levels) != list(LEVELS):
            raise PromptError(f"{name}: meta.levels {levels} differ from the design's {list(LEVELS)}")
        if not _is_number(meta["points_per_second"]) or Decimal(repr(float(meta["points_per_second"]))) != POINTS_PER_SECOND:
            raise PromptError(f"{name}: meta.points_per_second must be {POINTS_PER_SECOND}, got {meta['points_per_second']!r}")
        return dict(meta)

    @staticmethod
    def _check_tool(name: str, tool: Any) -> dict[str, str]:
        if not isinstance(tool, dict) or set(tool) != _TOOL_KEYS:
            keys = sorted(tool) if isinstance(tool, dict) else type(tool).__name__
            raise PromptError(f"{name}: tool keys {keys} differ from {sorted(_TOOL_KEYS)}")
        for key in _TOOL_KEYS:
            _check_text(f"{name}: tool.{key}", tool[key])
            if "\n" in tool[key]:
                raise PromptError(f"{name}: tool.{key} must be one line")
        if tool["name"] != TOOL_NAME:
            raise PromptError(f"{name}: tool.name {tool['name']!r} must be {TOOL_NAME!r}")
        if tool["parameter"] != "question":
            raise PromptError(f"{name}: tool.parameter must be 'question', got {tool['parameter']!r}")
        return {k: tool[k] for k in sorted(_TOOL_KEYS)}

    @staticmethod
    def _check_content(name: str, texts: Mapping[str, str], tool_name: str) -> None:
        """Wording constraints: the plain prompts say nothing of the expert, the decide prompts share their role paragraph and scoring sentences, the call note is absent."""
        for label in ("answer.system", "expert.system", "answer.follow_up"):
            lowered = texts[label].lower()
            if "\n" in texts[label]:
                raise PromptError(f"{name}: {label} must be one line")
            for word in (tool_name, "edge", "deduct", "tool"):
                if word in lowered:
                    raise PromptError(f"{name}: {label} must not mention {word!r} (no tool, no expert, no deduction)")
        decide = {k: v for k, v in texts.items() if k.startswith("S-")}
        roles = set()
        for label, text in decide.items():
            if CALL_NOTE in text:
                raise PromptError(f"{name}: {label} contains the call note sentence, which is not allowed")
            if ROLE_END not in text:
                raise PromptError(f"{name}: {label} has no scoring paragraph")
            roles.add(text.split(ROLE_END, 1)[0])
            for sentence in SHARED_SENTENCES:
                if sentence not in text:
                    raise PromptError(f"{name}: {label} lacks the shared scoring sentence {sentence!r}")
            if f"call the tool {tool_name} " not in text:
                raise PromptError(f"{name}: {label} must name the tool in the role paragraph")
            if "{a_pct}%" not in text:
                raise PromptError(f"{name}: {label} must state the expert's accuracy as {{a_pct}}%")
        if len(roles) != 1:
            raise PromptError(f"{name}: the three groups must share one role paragraph")
        for label in ("S-nat.system_busy", "S-nat.system_idle"):
            if "Each second of waiting for the edge expert costs 0.01 points, deducted from your score." not in texts[label]:
                raise PromptError(f"{name}: {label} must state the cost per second of 0.01 points")
        busy, idle = texts["S-nat.system_busy"].split("\n"), texts["S-nat.system_idle"].split("\n")
        if len(busy) != len(idle) or busy[:-1] != idle[:-1]:
            raise PromptError(f"{name}: S-nat busy and idle may differ in the last line only")
        if busy[-1] == idle[-1] or "idle" not in idle[-1] or "busy" not in busy[-1]:
            raise PromptError(f"{name}: the last line of S-nat must say busy with the wait, and idle without it")

    # ---- the conditions ----------------------------------------------------------------
    def conditions(self, groups: Sequence[str] | None = None) -> list[Condition]:
        """The conditions of the chosen groups (all three by default): group order S-exp, S-nat, S-loc and, within a group, ascending level."""
        chosen = list(GROUPS if groups is None else groups)
        for group in chosen:
            if group not in GROUPS:
                raise PromptError(f"unknown group {group!r}; known: {list(GROUPS)}")
        if len(set(chosen)) != len(chosen):
            raise PromptError(f"a group is listed twice: {chosen}")
        return [Condition(g, level) for g in GROUPS if g in chosen for level in self.levels]

    def condition(self, label: str) -> Condition:
        for cond in self.conditions():
            if cond.label == label:
                return cond
        raise PromptError(f"unknown condition {label!r}; known: {[c.label for c in self.conditions()]}")

    # ---- the tool ------------------------------------------------------------------------
    def tool_schema(self) -> dict[str, Any]:
        """The one function ``ask_edge_expert(question: string)``, in the OpenAI function format that the chat templates take."""
        t = self.tool
        return {
            "type": "function",
            "function": {
                "name": t["name"], "description": t["description"],
                "parameters": {"type": "object", "properties": {t["parameter"]: {"type": "string", "description": t["parameter_description"]}}, "required": [t["parameter"]]},
            },
        }

    def tools(self) -> list[dict[str, Any]]:
        return [self.tool_schema()]

    # ---- system prompts and messages -----------------------------------------------------
    def system_for(self, cond: Condition, a_pct: int) -> str:
        """The system prompt of a condition for the expert accuracy ``a_pct`` (an integer percent)."""
        a_pct = check_a_pct(a_pct)
        if cond.level not in self.levels or cond.group not in GROUPS:
            raise PromptError(f"{cond!r} is not a condition of this file")
        shown = Decimal(cond.level) / 100
        if cond.group == "S-exp":
            return self.templates["S-exp"].format(a_pct=a_pct, D=cond.level, w=shown)
        if cond.group == "S-nat":
            return self.templates["S-nat.idle"].format(a_pct=a_pct) if cond.level == 0 else self.templates["S-nat.busy"].format(a_pct=a_pct, D=cond.level)
        return self.templates["S-loc"].format(a_pct=a_pct, x=shown)

    def decide_messages(self, cond: Condition, a_pct: int, question: str) -> list[dict[str, str]]:
        """The conversation of P-decide: the system prompt of the condition and the question as the user message (the tool is passed to the template separately)."""
        self._check_question(question)
        return [{"role": "system", "content": self.system_for(cond, a_pct)}, {"role": "user", "content": question}]

    def answer_messages(self, question: str) -> list[dict[str, str]]:
        """P-answer: the plain system prompt and the question."""
        self._check_question(question)
        return [{"role": "system", "content": self.answer_system}, {"role": "user", "content": question}]

    def follow_up_messages(self, question: str, answer_text: str) -> list[dict[str, str]]:
        """The confidence follow-up: P-answer's conversation, the model's complete answer as the assistant turn, and the follow-up question as the next user turn."""
        self._check_question(question)
        if not isinstance(answer_text, str):
            raise PromptError("the answer must be a string")
        return [*self.answer_messages(question), {"role": "assistant", "content": answer_text}, {"role": "user", "content": self.follow_up}]

    def expert_messages(self, question: str) -> list[dict[str, str]]:
        """The expert phase: the expert's system prompt and the question."""
        self._check_question(question)
        return [{"role": "system", "content": self.expert_system}, {"role": "user", "content": question}]

    @staticmethod
    def _check_question(question: Any) -> None:
        if not isinstance(question, str) or not question.strip():
            raise PromptError("the question must be a non-empty string")

    # ---- a content-free description, for the manifest -------------------------------------
    def describe(self) -> dict[str, Any]:
        """Version, md5 and the md5 of every template (no text)."""
        return {
            "path": str(self.path), "md5": self.md5, "version": self.version, "date": self.date,
            "tool_schema_md5": hashlib.md5(json.dumps(self.tool_schema(), sort_keys=True).encode("utf-8"), usedforsecurity=False).hexdigest(),
            "template_md5": {
                k: hashlib.md5(v.encode("utf-8"), usedforsecurity=False).hexdigest()
                for k, v in {"answer.system": self.answer_system, "answer.follow_up": self.follow_up, "expert.system": self.expert_system, **self.templates}.items()
            },
            "levels": list(self.levels), "groups": list(GROUPS),
        }
