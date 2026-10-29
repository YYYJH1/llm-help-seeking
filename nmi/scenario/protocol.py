"""Three-phase generation protocol of the scenario S1 and its failure rules.

Package overview (``nmi/scenario/__init__.py`` stays empty; the entry is ``scripts/run_scenario.py``, the data adapter ``nmi.data.triviaqa``):
``prompts`` and ``render`` (``configs/prompts/scenario_s1.yaml``; the messages through a model's chat template, the tool injected by family), ``adapter`` (what counts as a call, which call is executed,
the three outcomes), ``scoring`` (normalisation, first line, aliases and the three sensitivity variants), ``readout`` (the digit readout of the confidence follow-up), ``backend`` (the generation back end),
this module, ``calibration`` (the two-parameter calibration and its three gates: the calibration gate of the development pool), ``analysis`` (what any batch's tables allow without fitting anything: the paired item sets,
the quality gate of a batch, z from the development fit), ``summary`` (counts and rates of a phase table), ``model_files`` (the fingerprint of the model directory that the manifest binds), ``config`` (the run configuration), ``task`` (the task adapter).

Phases (one unit of work is a set of items, and for ``decide`` one condition):

* ``expert``: the expert (Qwen3-32B-AWQ, no thinking) answers each item once. Greedy, at most 32 tokens, the first line is scored.
* ``answer``: the end-side model's P-answer (same settings; no tool, no expert, no deduction in the prompt), then the confidence follow-up about that complete answer, read as the exact log-probabilities of
  the ten digit tokens (``logprob_token_ids``, ``max_tokens=1``): ``digit_mass`` (not normalised), ``c`` (conditional expectation / 9), ``digit_gate`` (mass >= 0.9).
* ``decide``: P-decide. The system prompt of one condition with the tool schema injected by family, the question as the user message, greedy, at most 96 tokens, special tokens kept. The reply goes through the
  adapter (``nmi.scenario.adapter``): outcome (offload, self_answer, format_failure), syntax, multi-call, whether the generated question equals the original one, the native-writing reading; a self-answer is scored
  like P-answer (the first line of the cleaned reply).

Failures. A generation that raised or did not come back is a failed unit element; it is run once more with the same settings, and when it fails again it is recorded as MISSING in the table
of its phase (``status = "missing"``, every derived field null): the expert's result for that item is missing; the answer, its correctness and ``c`` are missing (so is ``z``); the decision of that item under
that condition is missing and is NOT a format failure. A reply whose CONTENT is unusable (an empty reply, no valid call, a truncated call) is not a failure of this kind. If the follow-up of an item fails after
its answer succeeded, the answer-phase result of the item is missing as a whole (answer correctness and ``c`` together).

Missing flags. Every row of every phase table carries the boolean ``missing`` (never null; True exactly when ``status == "missing"``): ``expert`` and ``answer`` one row per
registered item, ``decide`` one row per registered item AND condition, the row keeping its ``condition``, ``group``, ``level``, ``delay_s`` and ``points`` when the decision is missing. The confirmatory paired
differences use only the items that are not missing under every condition of the contrast; descriptions of one condition use that condition's own non-missing items.
The flag says that no result exists; an item whose answer was read but whose digit distribution has no mass is NOT missing (its ``c`` is null).

Only derived fields go into the tables. The generated texts are returned separately (``UnitOutput.raw``) for the file that only code reads; the entry writes it and nothing here prints or logs it. Exceptions that a
back end raises are counted by class name only.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa

from nmi.scenario import adapter, readout, scoring
from nmi.scenario.backend import GenerationBackend, GenResult, GenSpec
from nmi.scenario.prompts import Condition, ScenarioPromptBook
from nmi.scenario.render import INJECTION_TOOLS_ARG, render

PHASES = ("expert", "answer", "decide")
OK, MISSING = "ok", "missing"
MISSING_REASON = "generation_failed_twice"
RETRIES = 1  # a failed element is run once more with the original settings


@dataclass(frozen=True)
class PhaseSettings:
    """Numeric settings of the phases; they go into the manifest."""

    expert_max_tokens: int = 32
    answer_max_tokens: int = 32
    follow_up_max_tokens: int = 1
    decide_max_tokens: int = 96

    def as_dict(self) -> dict[str, int]:
        return {"expert_max_tokens": self.expert_max_tokens, "answer_max_tokens": self.answer_max_tokens, "follow_up_max_tokens": self.follow_up_max_tokens, "decide_max_tokens": self.decide_max_tokens}


# ---------------------------------------------------------------------------------------
# the table schemas
# ---------------------------------------------------------------------------------------

_COMMON = [("item_key", pa.string()), ("status", pa.string()), ("missing", pa.bool_()), ("missing_reason", pa.string())]
_SCORES = [("correct", pa.bool_()), ("correct_full_alias", pa.bool_()), ("correct_negation", pa.bool_()), ("correct_short_boundary", pa.bool_())]
_SELF_SCORES = [("self_correct", pa.bool_()), ("self_correct_full_alias", pa.bool_()), ("self_correct_negation", pa.bool_()), ("self_correct_short_boundary", pa.bool_())]
_GEN = [("empty", pa.bool_()), ("gen_tokens", pa.int64()), ("truncated", pa.bool_()), ("finish_reason", pa.string()), ("service_s", pa.float64()), ("unit_seconds", pa.float64())]
EXPERT_SCHEMA = pa.schema([*_COMMON, *_SCORES, *_GEN])
ANSWER_SCHEMA = pa.schema([
    *_COMMON, *_SCORES, *_GEN, ("c", pa.float64()), ("digit_mass", pa.float64()), ("digit_gate", pa.bool_()), ("digit_logprobs", pa.list_(pa.float64())), ("followup_service_s", pa.float64()),
])
DECIDE_SCHEMA = pa.schema([
    *_COMMON, ("condition", pa.string()), ("group", pa.string()), ("level", pa.int64()), ("delay_s", pa.int64()), ("points", pa.float64()),
    ("outcome", pa.string()), ("reply_class", pa.string()), ("offload", pa.bool_()), ("structure", pa.string()), ("syntax", pa.string()), ("first_valid_tagged", pa.bool_()),
    ("n_valid", pa.int64()), ("n_invalid", pa.int64()), ("multi_call", pa.bool_()), ("executed_index", pa.int64()), ("arg_equals_question", pa.bool_()), ("text_before", pa.bool_()), ("text_after", pa.bool_()),
    ("executed_native", pa.bool_()), ("native_offload", pa.bool_()), ("native_syntax", pa.string()), ("native_n_valid", pa.int64()), ("shape", pa.string()), *_SELF_SCORES, *_GEN,
])
SCHEMAS = {"expert": EXPERT_SCHEMA, "answer": ANSWER_SCHEMA, "decide": DECIDE_SCHEMA}


def build_table(schema: pa.Schema, columns: Mapping[str, Sequence[Any]]) -> pa.Table:
    """A table of the schema from lists of values (None is null); every column of the schema is needed and none else."""
    if set(columns) != set(schema.names):
        raise ValueError(f"columns {sorted(set(columns) ^ set(schema.names))} do not match the schema")
    return pa.table({f.name: pa.array(list(columns[f.name]), type=f.type) for f in schema}, schema=schema)


# ---------------------------------------------------------------------------------------
# rendering for one model
# ---------------------------------------------------------------------------------------

@dataclass
class Renderer:
    """A model's chat template with its settings: the prompts as the model reads them. ``family`` is the tool-call family (the native writing of the adapter's sensitivity analysis)."""

    tokenizer: Any
    chat_kwargs: Mapping[str, Any]
    family: str
    injection: str = INJECTION_TOOLS_ARG
    tools: Sequence[Mapping[str, Any]] = ()

    def plain(self, messages: Sequence[Mapping[str, str]]) -> str:
        return render(self.tokenizer, messages, self.chat_kwargs)

    def with_tools(self, messages: Sequence[Mapping[str, str]]) -> str:
        return render(self.tokenizer, messages, self.chat_kwargs, tools=self.tools, injection=self.injection)


# ---------------------------------------------------------------------------------------
# generation with the per-element failure rule
# ---------------------------------------------------------------------------------------

@dataclass
class GenStats:
    n: int = 0
    batch_errors: int = 0  # calls that raised or returned the wrong number of results
    retried: int = 0  # elements that were run a second time
    missing: int = 0  # elements without a result after the second run
    error_classes: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"n": self.n, "batch_errors": self.batch_errors, "retried": self.retried, "missing": self.missing, "error_classes": dict(sorted(self.error_classes.items()))}

    def merge(self, other: "GenStats") -> "GenStats":
        classes = Counter(self.error_classes)
        classes.update(other.error_classes)
        return GenStats(self.n + other.n, self.batch_errors + other.batch_errors, self.retried + other.retried, self.missing + other.missing, dict(classes))


def generate_with_retry(backend: GenerationBackend, prompts: Sequence[str], spec: GenSpec, *, retries: int = RETRIES) -> tuple[list[GenResult | None], GenStats]:
    """Run the prompts; an element whose generation raised or did not come back is run again, with the same settings, ``retries`` times (once); what is still without a result is ``None``.

    A call that raises, or returns another number of results than prompts, fails all of its elements. Exceptions are counted by class name; their messages (which may hold prompt text) are dropped.
    ``KeyboardInterrupt`` and ``SystemExit`` are not caught."""
    results: list[GenResult | None] = [None] * len(prompts)
    stats = GenStats(n=len(prompts))
    pending = list(range(len(prompts)))
    for attempt in range(retries + 1):
        if not pending:
            break
        if attempt > 0:
            stats.retried += len(pending)
        try:
            got = backend.generate([prompts[i] for i in pending], spec)
            if len(got) != len(pending):
                raise ValueError("wrong number of results")
        except Exception as error:  # noqa: BLE001 - a failed batch of any kind is a failed element; the class name is all that is kept
            stats.batch_errors += 1
            stats.error_classes[type(error).__name__] = stats.error_classes.get(type(error).__name__, 0) + 1
            continue
        still: list[int] = []
        for i, result in zip(pending, got, strict=True):
            if result is None:
                still.append(i)
            else:
                results[i] = result
        pending = still
    stats.missing = len(pending)
    return results, stats


# ---------------------------------------------------------------------------------------
# units of work
# ---------------------------------------------------------------------------------------

@dataclass
class UnitOutput:
    """The result of one unit: its table columns (one value per item, in item order), the raw lines for the file that only code reads, and the generation statistics."""

    columns: dict[str, list[Any]]
    raw: list[dict[str, Any]]
    stats: GenStats
    seconds: float

    def table(self, schema: pa.Schema) -> pa.Table:
        return build_table(schema, self.columns)


def _empty_columns(schema: pa.Schema, n: int) -> dict[str, list[Any]]:
    return {name: [None] * n for name in schema.names}


def _fill_missing(columns: dict[str, list[Any]], k: int, item_key: str) -> None:
    columns["item_key"][k] = item_key
    columns["status"][k] = MISSING
    columns["missing"][k] = True
    columns["missing_reason"][k] = MISSING_REASON


def _fill_ok(columns: dict[str, list[Any]], k: int, item_key: str) -> None:
    columns["item_key"][k] = item_key
    columns["status"][k] = OK
    columns["missing"][k] = False


def _fill_generation(columns: dict[str, list[Any]], k: int, result: GenResult, unit_seconds: float) -> None:
    columns["empty"][k] = adapter.clean(result.text) == ""
    columns["gen_tokens"][k] = len(result.token_ids)
    columns["truncated"][k] = result.finish_reason == "length"
    columns["finish_reason"][k] = result.finish_reason
    columns["service_s"][k] = result.service_s
    columns["unit_seconds"][k] = unit_seconds


def _fill_scores(columns: dict[str, list[Any]], k: int, scores: scoring.Scores, prefix: str) -> None:
    columns[prefix][k] = scores.primary
    columns[f"{prefix}_full_alias"][k] = scores.full_alias
    columns[f"{prefix}_negation"][k] = scores.negation
    columns[f"{prefix}_short_boundary"][k] = scores.short_boundary


def expert_unit(backend: GenerationBackend, renderer: Renderer, book: ScenarioPromptBook, task: Any, items: Sequence[Any], settings: PhaseSettings, *, unit_id: str) -> UnitOutput:
    """The expert answers each item once (greedy, at most ``expert_max_tokens`` tokens, no thinking); the first line of the cleaned reply is scored."""
    t0 = time.perf_counter()
    prompts = [renderer.plain(book.expert_messages(item.question)) for item in items]
    results, stats = generate_with_retry(backend, prompts, GenSpec(max_tokens=settings.expert_max_tokens))
    seconds = time.perf_counter() - t0
    columns = _empty_columns(EXPERT_SCHEMA, len(items))
    raw: list[dict[str, Any]] = []
    for k, (item, result) in enumerate(zip(items, results, strict=True)):
        if result is None:
            _fill_missing(columns, k, item.item_key)
            continue
        _fill_ok(columns, k, item.item_key)
        _fill_scores(columns, k, task.score(result.text, item), "correct")
        _fill_generation(columns, k, result, seconds)
        raw.append({"phase": "expert", "unit": unit_id, "item_key": item.item_key, "finish_reason": result.finish_reason, "text": result.text})
    return UnitOutput(columns, raw, stats, seconds)


def answer_unit(
    backend: GenerationBackend, renderer: Renderer, book: ScenarioPromptBook, task: Any, items: Sequence[Any], settings: PhaseSettings, digit_ids: Sequence[int], *, unit_id: str,
) -> UnitOutput:
    """P-answer and the confidence follow-up about that complete answer. An item whose answer or follow-up failed twice has its whole answer-phase result missing."""
    t0 = time.perf_counter()
    prompts = [renderer.plain(book.answer_messages(item.question)) for item in items]
    answers, stats_a = generate_with_retry(backend, prompts, GenSpec(max_tokens=settings.answer_max_tokens))
    have = [k for k, r in enumerate(answers) if r is not None]
    follow_prompts = [renderer.plain(book.follow_up_messages(items[k].question, adapter.clean(answers[k].text))) for k in have]  # type: ignore[union-attr]
    follow, stats_f = generate_with_retry(backend, follow_prompts, GenSpec(max_tokens=settings.follow_up_max_tokens, logprob_token_ids=tuple(digit_ids)))
    seconds = time.perf_counter() - t0
    follow_of: dict[int, GenResult | None] = dict(zip(have, follow, strict=True))
    columns = _empty_columns(ANSWER_SCHEMA, len(items))
    raw: list[dict[str, Any]] = []
    unreadable = 0
    for k, item in enumerate(items):
        answer, second = answers[k], follow_of.get(k)
        confidence = None
        if answer is not None and second is not None and second.logprobs is not None:
            try:
                confidence = readout.confidence_from_logprobs([second.logprobs[i] for i in digit_ids])
            except (readout.ReadoutValueError, KeyError):
                unreadable += 1
        if answer is None or second is None or confidence is None:
            _fill_missing(columns, k, item.item_key)
            continue
        _fill_ok(columns, k, item.item_key)
        _fill_scores(columns, k, task.score(answer.text, item), "correct")
        _fill_generation(columns, k, answer, seconds)
        columns["c"][k] = None if confidence.c != confidence.c else confidence.c
        columns["digit_mass"][k], columns["digit_gate"][k] = confidence.mass, confidence.gate_pass
        columns["digit_logprobs"][k] = list(confidence.logprobs)
        columns["followup_service_s"][k] = second.service_s
        raw.append({"phase": "answer", "unit": unit_id, "item_key": item.item_key, "finish_reason": answer.finish_reason, "text": answer.text, "follow_up_text": second.text})
    stats = stats_a.merge(stats_f)
    if unreadable:
        stats.error_classes["ReadoutValueError"] = stats.error_classes.get("ReadoutValueError", 0) + unreadable
    return UnitOutput(columns, raw, stats, seconds)


def decide_unit(
    backend: GenerationBackend, renderer: Renderer, book: ScenarioPromptBook, task: Any, items: Sequence[Any], cond: Condition, a_pct: int, settings: PhaseSettings, *, unit_id: str,
) -> UnitOutput:
    """P-decide under one condition. The reply goes through the adapter; the derived fields never hold the reply."""
    t0 = time.perf_counter()
    prompts = [renderer.with_tools(book.decide_messages(cond, a_pct, item.question)) for item in items]
    results, stats = generate_with_retry(backend, prompts, GenSpec(max_tokens=settings.decide_max_tokens, skip_special_tokens=False))
    seconds = time.perf_counter() - t0
    columns = _empty_columns(DECIDE_SCHEMA, len(items))
    raw: list[dict[str, Any]] = []
    for k, (item, result) in enumerate(zip(items, results, strict=True)):
        columns["condition"][k], columns["group"][k], columns["level"][k] = cond.label, cond.group, cond.level
        columns["delay_s"][k], columns["points"][k] = cond.delay_s, float(cond.points)
        if result is None:
            _fill_missing(columns, k, item.item_key)
            continue
        _fill_ok(columns, k, item.item_key)
        row = adapter.analyze(result.text, item.question, renderer.family)
        columns["outcome"][k], columns["reply_class"][k], columns["offload"][k] = row["outcome"], row["class"], row["offload"]
        for name in ("structure", "syntax", "first_valid_tagged", "n_valid", "n_invalid", "multi_call", "executed_index", "arg_equals_question", "text_before", "text_after", "executed_native", "native_offload", "native_syntax", "native_n_valid", "shape"):
            columns[name][k] = row[name]
        if row["outcome"] == "self_answer":
            _fill_scores(columns, k, task.score(result.text, item), "self_correct")
        _fill_generation(columns, k, result, seconds)
        raw.append({"phase": "decide", "unit": unit_id, "item_key": item.item_key, "condition": cond.label, "finish_reason": result.finish_reason, "text": result.text})
    return UnitOutput(columns, raw, stats, seconds)
