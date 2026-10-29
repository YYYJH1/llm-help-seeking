#!/usr/bin/env python
"""Real scenario S1: an end-side question-answering assistant that answers a TriviaQA question itself or hands it to an edge expert.
Prompts: configs/prompts/scenario_s1.yaml; run configuration: configs/scenario_s1_v1.yaml; data: nmi/data/triviaqa.py (data/splits/s1_triviaqa_v1); adapter: nmi/scenario/adapter.py.

Three phases, one model per run (``--phase expert|answer|decide``, ``--model``):

``expert``  the expert (Qwen3-32B-AWQ, no thinking) answers every item of the development pool once: greedy, at most 32 tokens, the first line is scored (primary score and the three sensitivity variants).
``answer``  the end-side model's P-answer (same settings; no tool, no expert, no deduction in the prompt), then the confidence follow-up about that complete answer: the exact log-probabilities of the ten
            digit tokens (vLLM ``SamplingParams(logprob_token_ids=<the ten ids>, max_tokens=1)``, no top-k) give the digit mass and the raw confidence c. At the end the cross-fitted two-parameter calibration and
            its three gates (the CALIBRATION GATE of the development pool, ``nmi.scenario.calibration``, block ``calibration``) are computed and written to the summary (``answer_summary.json``), and next to it,
            as a separate block ``quality_gate``, the QUALITY GATE that any batch gets (``nmi.scenario.analysis``): the digit quality over ALL items of the pool (an item without a usable digit distribution
            does not pass) and the answer-phase missing share. A confirmatory batch would get only the second one, with no refit; this entry reads the development pool only.
``decide``  P-decide: the system prompt of one of the 18 conditions (S-exp, S-nat, S-loc at D = 0, 10, 20, 45, 80, 150) with the tool ``ask_edge_expert`` injected by family (``tools=``; Phi-4-mini: the
            ``tools`` field of the system message; after rendering the tool name must be in the prompt), greedy, at most 96 tokens, special tokens kept. The reply goes through the adapter: outcome (offload,
            self_answer, format_failure), syntax, multi-call, whether the generated question equals the original one (the server forwards the original question and executes the first valid call only), the
            native-writing reading. ``--conditions`` takes condition labels (S-exp_D010, S-loc_x045) or group names (S-exp, S-nat, S-loc); the default is all 18. The expert accuracy ``a_pct`` that the prompts
            state comes from the run configuration and must be set (the phase refuses ``null``).

Only the development pool is accepted (``--pools triviaqa_s1_dev``, the default); the sealed pool, any other pool name or several pools exit 2 before anything is read. ``--limit-items N`` takes the first N
items of the pool (a smoke test); it is part of the identity of a run.

Failures. A generation that raised or did not come back is run once more with the same settings; when it fails again the item (for ``decide``: the item under that condition) is recorded as
MISSING in the table of its phase (``status = "missing"`` and the boolean ``missing``, one flag per item and, for decide, per condition), which is not a format failure. Each summary gives the missing share of
the phase; a decide condition with more than 1 % missing is marked untestable, and a model whose answer phase is missing for more than 1 % of the items has an invalid z. A paired analysis uses the items
that are not missing under every condition of the contrast.

Output (``results/scenario/<time>_<tag>/``): ``<phase>.parquet`` (derived fields only: item key, ``status`` and the boolean ``missing`` of every item and, for decide, of every item under every condition, condition, the three outcomes, syntax, tagged or not, native reading, truncated, multi-call, generated question equal
to the original, correctness under the four scores, digit distribution, token counts, seconds) and ``<phase>_summary.json``; ``raw_outputs.jsonl.gz`` holds the generated texts for the code that re-parses them and
for audits (its md5 is in the manifest); nothing the entry prints or logs contains a question or a generated text; ``llm/model.json``; ``manifest.json``.

Manifest. Besides the entry fields of every run (git state, the hash of the whole ``nmi/`` source, the package versions) it binds, under ``input_md5``, the md5 of the run configuration
(``config``), the prompt file (``prompts``), this file (``entry``), the code that decides what a result is (``adapter``, ``scoring``, ``readout``, ``protocol``, ``calibration``,
``measure_calibration``, ``summary``, ``analysis``) and the pool registry (``split.*``); under ``numeric`` the numeric settings of the model and the back end (versions of torch, transformers and
vllm, the engine arguments, what the engine reports); and under ``scenario_run`` the identity of the run: the model with its settings and the fingerprint of its directory (``model_files``; the sha256
of the weights themselves is in ``llm/model.json``), the md5 of the chat template and of every rendered prompt of a probe question, the phase, the items, the conditions, the frozen ``a_pct``, the token
limits, the unit size, the tool injection, the digit token ids, the thresholds of the two gates and the hash of the source file. ``exploratory: true`` and a ``development_reason`` are written to the
manifest; the stage is ``scenario``.

Resume. ``--resume RUN_DIR`` continues an interrupted run: the identity of the call is compared with the recorded one BEFORE any finished unit is reused (the unit names hold the phase, the condition and the
chunk number only); a difference exits 2 and touches nothing, and the md5 values of the inputs, the source hash of ``nmi/`` and the numeric settings must also be those of the first start of the run.
``--dry-run`` checks the configuration, the prompts, the pool, the source file hash, the model directory (and shows its fingerprint), the chat template of the model (tool injection, digits, rendering) and
the plan, and loads no model.

Typical use (from a clean checkout; environment: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_WORKER_MULTIPROC_METHOD=spawn, one card per run):
    python -I scripts/run_scenario.py --phase expert --model qwen3-32b-awq --tag s1_expert --gpu-mem-util 0.85 --results-root results --dry-run     (then without --dry-run)
    then set a_pct in configs/scenario_s1_v1.yaml to ``a_pct_suggested`` of results/scenario/<run>/expert_summary.json (the md5 of the file changes; the decide phase refuses ``null``);
    python -I scripts/run_scenario.py --phase answer --model qwen3-4b --tag s1_answer_qwen3-4b --gpu-mem-util 0.45 --limit-items 20 --results-root ...     (a smoke test; drop --limit-items for the run)
    python -I scripts/run_scenario.py --phase decide --model qwen3-4b --tag s1_decide_qwen3-4b --gpu-mem-util 0.45 --conditions S-exp_D010,S-loc_x045 --limit-items 20 --results-root ...

Exit codes: 0 done; 1 a dry run found a problem, or the calibration of an answer run raised; 2 unusable input or arguments (nothing was computed); 4 the source guard stopped the dispatch (``nmi/`` changed while
the batch ran).
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import io
import json
import math
import os
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
for _p in (HERE.parent, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import _c2_common as common  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402

from nmi import paths  # noqa: E402
from nmi.data import triviaqa  # noqa: E402
from nmi.data.holdout import HoldoutLedgerError, HoldoutSealedError  # noqa: E402
from nmi.data.registry import InputHashMismatchError, SplitError  # noqa: E402
from nmi.prompts import PromptError  # noqa: E402
from nmi.provenance import RunRecord, SourceHashGuard, begin_run, file_md5, iso, now_local  # noqa: E402
from nmi.readout.digits import DigitTokenError  # noqa: E402
from nmi.readout.hf_backend import ReadoutError  # noqa: E402
from nmi.measure import calibration as measure_calibration  # noqa: E402
from nmi.scenario import adapter, analysis, calibration, model_files, protocol, readout, render, scoring, summary  # noqa: E402
from nmi.scenario.backend import GenSpec, VLLMGenerationBackend  # noqa: E402
from nmi.scenario.config import DEFAULT_SCENARIO_CONFIG, ModelEntry, ScenarioConfig, ScenarioConfigError, load_scenario_config  # noqa: E402
from nmi.scenario.prompts import GROUPS, Condition, ScenarioPromptBook  # noqa: E402
from nmi.scenario.task import TASKS  # noqa: E402

SCRIPT = "run_scenario"
STAGE = "scenario"
PROBE_QUESTION = "<question>"  # the rendering fingerprints are made on this placeholder: no real question is rendered for the manifest
RAW_FILE = "raw_outputs.jsonl.gz"
RAW_PARTS = "raw_parts"
UNIT_STATS = "unit_stats"
EXIT_FAILED_CHECK = 1
USER_ERRORS = (
    common.EntryError, SplitError, paths.PathPolicyError, ScenarioConfigError, PromptError, DigitTokenError, ReadoutError, InputHashMismatchError, HoldoutSealedError, HoldoutLedgerError,
)


@dataclass
class ScenarioHooks:
    """Replaceable parts, for the tests: how the tokenizer is loaded, how the back end is opened (it gets the tokenizer), and the model-hash switch."""

    tokenizer_loader: Callable[[ModelEntry], Any] | None = None
    opener: Callable[["ScenarioContext", ModelEntry, Any], common.BackendHandle] | None = None
    model_files: bool = True


@dataclass
class ScenarioContext:
    args: argparse.Namespace
    cfg: ScenarioConfig
    book: ScenarioPromptBook
    task: Any
    split: triviaqa.TriviaSplit
    results_root: Path
    source: Path
    script: str
    stage: str
    argv: list[str]

    @property
    def tag(self) -> str | None:
        return getattr(self.args, "tag", None)


# ---------------------------------------------------------------------------------------
# arguments and inputs
# ---------------------------------------------------------------------------------------

def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Real scenario S1: the phases expert, answer and decide of one model on the development pool of TriviaQA.")
    p.add_argument("--phase", required=True, choices=protocol.PHASES, help="expert (the expert answers), answer (P-answer and the confidence follow-up) or decide (P-decide under the chosen conditions)")
    p.add_argument("--model", required=True, help="model id of the run configuration: the expert for --phase expert, an end-side model for the others")
    p.add_argument("--config", default=str(DEFAULT_SCENARIO_CONFIG), help="run configuration (default: configs/scenario_s1_v1.yaml)")
    p.add_argument("--expect-config-md5", default=None, help="stop unless the configuration has exactly this md5 (the frozen value)")
    p.add_argument("--results-root", default=None, help="default: paths.results_root of the configuration, relative to the repository")
    p.add_argument("--split-dir", default=None, help="registry directory of the pools (default: data.split_dir of the configuration)")
    p.add_argument("--source", default=None, help="the TriviaQA validation parquet (default: data.source of the configuration; its sha256 must be the one the registry was sealed on)")
    p.add_argument("--tag", default=None, help="run tag: results/scenario/<time>_<tag>/ (required unless --dry-run; an existing tag is an error)")
    p.add_argument("--resume", default=None, metavar="RUN_DIR", help="continue an interrupted run in its own directory; finished units are skipped")
    p.add_argument("--dry-run", action="store_true", help="check the configuration, the prompts, the pool, the model directory and the chat template; print the plan; load no model")
    p.add_argument("--pools", default=None, help=f"only {triviaqa.DEV_POOL} (the default); the sealed pool and any other name exit 2")
    p.add_argument("--conditions", default=None, help="decide only: comma list of condition labels (S-exp_D010, S-nat_D000, S-loc_x045) or group names (S-exp, S-nat, S-loc); default: all 18")
    p.add_argument("--limit-items", type=int, default=None, help="smoke test: the first N items of the pool (part of the identity of the run)")
    p.add_argument("--backend", choices=("vllm_offline",), default="vllm_offline")
    p.add_argument("--gpu-mem-util", type=float, default=None, help="vLLM gpu_memory_utilization (offline); set it from the free memory measured before the run")
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--llm-kwargs", default=None, help="JSON object merged last into the keyword arguments of vllm.LLM (after the model's llm_kwargs of the configuration)")
    return p


def build_context(args: argparse.Namespace, *, argv: Sequence[str]) -> ScenarioContext:
    try:
        cfg = load_scenario_config(args.config, expected_md5=args.expect_config_md5)
    except (ScenarioConfigError, OSError) as error:
        raise common.EntryError(f"configuration: {error}") from error
    try:
        book = ScenarioPromptBook.load(cfg.prompts_path)
    except (PromptError, OSError) as error:
        raise common.EntryError(f"prompts: {error}") from error
    task = TASKS[cfg.task]
    try:
        split = task.load_split(args.split_dir or cfg.split_dir)
    except (SplitError, OSError, ValueError, KeyError) as error:
        raise common.EntryError(f"pool registry: {error}") from error
    return ScenarioContext(
        args=args, cfg=cfg, book=book, task=task, split=split, results_root=paths.results_root(args.results_root or cfg.results_root), source=Path(args.source) if args.source else cfg.source,
        script=SCRIPT, stage=STAGE, argv=[f"scripts/{SCRIPT}.py", *argv],
    )


def pool_of(args: argparse.Namespace) -> str:
    """The development pool, the one pool a run reads. The sealed pool, another name or several pools exit 2 before anything is read."""
    requested = common.csv_list(args.pools) or [triviaqa.DEV_POOL]
    if triviaqa.SEALED_POOL in requested:
        raise common.EntryError(f"the pool {triviaqa.SEALED_POOL} is sealed: no entry of the scenario reads it before the confirmatory phase (nmi.data.holdout)")
    if requested != [triviaqa.DEV_POOL]:
        raise common.EntryError(f"the scenario runs on the one pool {triviaqa.DEV_POOL} (development evidence); --pools {args.pools} was given: another pool, several pools or a repeated pool are refused")
    return triviaqa.DEV_POOL


def condition_list(args: argparse.Namespace, book: ScenarioPromptBook) -> list[Condition]:
    """The conditions of a decide run, in the canonical order. Only decide takes ``--conditions``; names are condition labels or group names; unknown and repeated names exit 2."""
    requested = common.csv_list(args.conditions)
    if args.phase != "decide":
        if requested:
            raise common.EntryError(f"--conditions belongs to --phase decide; the phase {args.phase} has no conditions")
        return []
    everything = book.conditions()
    if not requested:
        return everything
    labels = {c.label for c in everything}
    unknown = [r for r in requested if r not in labels and r not in GROUPS]
    if unknown:
        raise common.EntryError(f"unknown condition(s) {unknown}; known: {sorted(labels)} and the groups {list(GROUPS)}")
    picked: list[str] = []
    for name in requested:
        for label in [name] if name in labels else [c.label for c in everything if c.group == name]:
            if label in picked:
                raise common.EntryError(f"the condition {label} is named twice in --conditions {args.conditions} (also through its group)")
            picked.append(label)
    return [c for c in everything if c.label in picked]


def chunks(n: int, size: int) -> list[range]:
    return [range(i, min(i + size, n)) for i in range(0, n, size)]


@dataclass(frozen=True)
class UnitSpec:
    unit_id: str
    rows: range
    condition: Condition | None


def plan_units(phase: str, n_items: int, conds: Sequence[Condition], size: int) -> list[UnitSpec]:
    """Units of work: chunks of ``size`` items; for decide, the chunks of one condition together, the conditions in the canonical order."""
    cuts = chunks(n_items, size)
    if phase != "decide":
        return [UnitSpec(f"{phase}__c{ci:03d}", rows, None) for ci, rows in enumerate(cuts)]
    return [UnitSpec(f"decide__{cond.label}__c{ci:03d}", rows, cond) for cond in conds for ci, rows in enumerate(cuts)]


def renderer_of(ctx: ScenarioContext, entry: ModelEntry, tokenizer: Any) -> protocol.Renderer:
    injection = render.resolve_injection(tokenizer, entry.tool_injection)
    return protocol.Renderer(tokenizer=tokenizer, chat_kwargs=dict(entry.chat_template_kwargs), family=entry.family, injection=injection, tools=ctx.book.tools())


def rendering_fingerprints(ctx: ScenarioContext, renderer: protocol.Renderer, phase: str, conds: Sequence[Condition], a_pct: int | None) -> dict[str, str]:
    """md5 of the prompt of every rendering the phase uses, on a placeholder question (the manifest records these, never a prompt): this freezes how the model's template renders the system prompt of each
    condition. The decide renderings are also checked for the tool: its schema must reach the prompt, not only its name (the system prompt names it too)."""
    book = ctx.book
    if phase == "expert":
        return {"expert": render.fingerprint(renderer.plain(book.expert_messages(PROBE_QUESTION)))}
    if phase == "answer":
        return {
            "answer": render.fingerprint(renderer.plain(book.answer_messages(PROBE_QUESTION))),
            "follow_up": render.fingerprint(renderer.plain(book.follow_up_messages(PROBE_QUESTION, "<answer>"))),
        }
    out: dict[str, str] = {}
    for cond in conds:
        messages = book.decide_messages(cond, a_pct or 50, PROBE_QUESTION)
        render.verify_injection(renderer.tokenizer, messages, renderer.chat_kwargs, renderer.tools, renderer.injection)
        out[cond.label] = render.fingerprint(renderer.with_tools(messages))
    return out


def load_tokenizer_for(hooks: ScenarioHooks, entry: ModelEntry) -> Any:
    try:
        return (hooks.tokenizer_loader or (lambda e: common.load_tokenizer(e.path)))(entry)
    except (OSError, ValueError, ImportError) as error:
        raise common.EntryError(f"model {entry.id!r}: the tokenizer cannot be loaded ({type(error).__name__}): {error}") from error


def vllm_kwargs(ctx: ScenarioContext, entry: ModelEntry) -> dict[str, Any]:
    """Keyword arguments of ``vllm.LLM``: the settings of the configuration, then the model's ``llm_kwargs``, then ``--llm-kwargs`` of the call."""
    args = ctx.args
    kwargs: dict[str, Any] = {"dtype": ctx.cfg.backend["dtype"], "max_model_len": args.max_model_len, "seed": ctx.cfg.backend["seed"], "gpu_memory_utilization": args.gpu_mem_util}
    kwargs.update(dict(entry.llm_kwargs))
    if args.llm_kwargs:
        try:
            extra = json.loads(args.llm_kwargs)
        except ValueError as error:
            raise common.EntryError(f"--llm-kwargs is not valid JSON: {error}") from error
        if not isinstance(extra, dict) or "logprobs_mode" in extra:
            raise common.EntryError("--llm-kwargs must be a JSON object without logprobs_mode")
        kwargs.update(extra)
    return kwargs


def backend_plan(ctx: ScenarioContext, entry: ModelEntry, unit: Mapping[str, Any]) -> dict[str, Any]:
    """The planned numeric record of the back end (the memory fraction changes no number and is left out), keyed like ``common.numeric_plan`` wants it."""
    kwargs = {k: v for k, v in vllm_kwargs(ctx, entry).items() if k != "gpu_memory_utilization"}
    return {"vllm_offline": {
        "kind": "vllm_offline", "dtype": kwargs.get("dtype"), "device": "cuda", "batch_size": None, "batch_plan": {"rule": common.VLLM_BATCH_RULE, "engine_kwargs": kwargs, "unit": dict(unit)}, "numerics": "engine",
    }}


def open_vllm(ctx: ScenarioContext, entry: ModelEntry, tokenizer: Any) -> common.BackendHandle:
    common.require_gpu_memory_fraction(ctx.args)
    kwargs = vllm_kwargs(ctx, entry)
    model_dir = Path(entry.path)
    backend = VLLMGenerationBackend.from_model(str(model_dir), tokenizer=tokenizer, **kwargs)

    def close() -> None:
        common.shutdown_offline_engine(backend.engine)

    return common.BackendHandle(backend, tokenizer, model_dir, "vllm_offline", {"llm_kwargs": kwargs}, closer=close)


# ---------------------------------------------------------------------------------------
# identity of a run
# ---------------------------------------------------------------------------------------

def md5_of_keys(keys: Sequence[str]) -> str:
    return hashlib.md5("\n".join(keys).encode("ascii"), usedforsecurity=False).hexdigest()


def model_fingerprint_of(entry: ModelEntry) -> dict[str, Any]:
    """The fingerprint of the model directory (``nmi.scenario.model_files``); an unreadable directory is a user error (exit 2) and not a crash."""
    try:
        return model_files.model_fingerprint(entry.path)
    except OSError as error:
        raise common.EntryError(f"model {entry.id!r}: the model directory cannot be read ({type(error).__name__}): {error}") from error


def bound_files(cfg: ScenarioConfig) -> dict[str, Path]:
    """The files whose md5 the manifest binds (``input_md5``) and a resumed run must still have: the run configuration and the prompt file, this entry, and the code that decides what a result is: the adapter
    (what counts as a call), the scoring, the digit readout, the three-phase protocol with its tables, the calibration and the AUROC and ECE it uses, the summaries, and the table-level analysis (the quality
    gate, the pairing, z from the frozen fit). The whole ``nmi/`` source is bound as well (``nmi_source_sha256``); these md5 values say which file changed."""

    def code(module: Any) -> Path:
        return Path(module.__file__).resolve()

    return {
        "config": cfg.path, "prompts": cfg.prompts_path, "entry": Path(__file__).resolve(), "adapter": code(adapter), "scoring": code(scoring), "readout": code(readout), "protocol": code(protocol),
        "calibration": code(calibration), "measure_calibration": code(measure_calibration), "summary": code(summary), "analysis": code(analysis),
    }


def scenario_identity(
    ctx: ScenarioContext, entry: ModelEntry, phase: str, pool: str, item_keys: Sequence[str], conds: Sequence[Condition], units: Sequence[UnitSpec], renderer: protocol.Renderer,
    fingerprints: Mapping[str, str], digit_ids: Sequence[int] | None, source_sha256: str, model_fp: Mapping[str, Any],
) -> dict[str, Any]:
    """What a run of the scenario is, beyond what the md5 values of the manifest hold. The manifest keeps it (``scenario_run``) and a resumed run must have the same (``check_resume_identity``): the unit
    names hold the phase, the condition and the chunk number only. ``model_fp`` is the fingerprint of the model directory (``model_fingerprint_of``)."""
    cfg = ctx.cfg
    return {
        "phase": phase, "model": entry.as_dict(), "model_files": dict(model_fp), "task": cfg.task, "pool": pool, "items": {"n": len(item_keys), "keys_md5": md5_of_keys(item_keys), "limit_items": ctx.args.limit_items},
        "conditions": [{"label": c.label, "group": c.group, "level": c.level, "delay_s": c.delay_s, "points": float(c.points)} for c in conds],
        "a_pct": cfg.a_pct if phase == "decide" else None, "settings": cfg.settings.as_dict(), "unit_items": cfg.unit_items, "retries": protocol.RETRIES, "n_units": len(units),
        "prompts": ctx.book.describe(), "tool_injection": renderer.injection, "digit_token_ids": None if digit_ids is None else [int(i) for i in digit_ids], "rendering_md5": dict(fingerprints),
        "chat_template_md5": hashlib.md5(render.template_text(renderer.tokenizer).encode("utf-8"), usedforsecurity=False).hexdigest(),
        "system_role": render.system_role(renderer.tokenizer, renderer.chat_kwargs),
        "sampling": sampling_record(cfg, digit_ids),
        "split": {"version": ctx.split.version, "pool_md5": ctx.split.pool_md5(pool), **ctx.split.file_md5s}, "source": {"file": triviaqa.SOURCE_FILE, "sha256": source_sha256},
        "backend": {"kind": cfg.backend["kind"], "dtype": cfg.backend["dtype"], "seed": cfg.backend["seed"], "max_model_len": ctx.args.max_model_len},
        "adapter": {"rules_version": adapter.RULES_VERSION, "md5": file_md5(Path(adapter.__file__))},
        "calibration": {"folds": calibration.N_FOLDS, "clip": calibration.CLIP, "digit_mass_min": calibration.DIGIT_MASS_MIN, "digit_pass_rate_min": calibration.DIGIT_PASS_RATE_MIN,
                        "auroc_resamples": calibration.AUROC_RESAMPLES, "auroc_seed": calibration.AUROC_SEED, "ece_bins": calibration.ECE_BINS, "ece_max": calibration.ECE_MAX} if phase == "answer" else None,
        "quality_gate": {"digit_mass_min": calibration.DIGIT_MASS_MIN, "digit_pass_rate_min": calibration.DIGIT_PASS_RATE_MIN, "digit_quality_denominator": "all registered items",
                         "answer_missing_max": analysis.ANSWER_MISSING_MAX, "refit": False} if phase == "answer" else None,
    }


def sampling_record(cfg: ScenarioConfig, digit_ids: Sequence[int] | None) -> dict[str, Any]:
    """The generation settings of the three phases as the manifest freezes them: greedy, the token limits, the special tokens kept for decide, the ten digit ids of the follow-up; the stop tokens are the
    model's own (the engine takes them from the generation configuration of the model, no stop string is set)."""
    s = cfg.settings
    return {
        "temperature": 0.0, "stop": "the model's own stop tokens (engine default from its generation config); no stop string",
        "expert": GenSpec(max_tokens=s.expert_max_tokens).as_dict(), "answer": GenSpec(max_tokens=s.answer_max_tokens).as_dict(),
        "follow_up": None if digit_ids is None else GenSpec(max_tokens=s.follow_up_max_tokens, logprob_token_ids=tuple(digit_ids)).as_dict(),
        "decide": GenSpec(max_tokens=s.decide_max_tokens, skip_special_tokens=False).as_dict(),
    }


def identity_differences(recorded: Mapping[str, Any], identity: Mapping[str, Any]) -> list[str]:
    """The ways in which two identities differ, one line each (the lists of conditions and fingerprints are summarised)."""
    before, after = dict(recorded), json.loads(json.dumps(identity))
    out: list[str] = []
    for key in ("conditions", "rendering_md5"):
        b, a = before.pop(key, None), after.pop(key, None)
        if b != a:
            out.append(f"{key}: {len(b or [])} recorded, {len(a or [])} now, they differ")
    return out + common.numeric_differences(before, after)


def check_resume_identity(args: argparse.Namespace, identity: Mapping[str, Any]) -> None:
    """Compare the identity of the call with that of the run being resumed BEFORE any finished unit is reused (EntryError, exit 2). Nothing is checked without --resume or when that run has no manifest
    (``open_scenario_run`` says so itself)."""
    resume = getattr(args, "resume", None)
    if not resume:
        return
    path = Path(resume) / common.MANIFEST_FILE
    if not path.is_file():
        return
    manifest = common.read_json(path)
    recorded = manifest.get("scenario_run") if isinstance(manifest, dict) else None
    if not isinstance(recorded, dict):
        raise common.EntryError(f"--resume {resume}: that is not a run of {SCRIPT} (its manifest has no scenario_run, the record of the model, phase, items and conditions it was started with)")
    differences = identity_differences(recorded, identity)
    if differences:
        raise common.EntryError(
            f"--resume {resume}: the phase, model, items, conditions, a_pct, settings, prompts, pool, source and back end of the run must be those it was started with (a finished unit of another "
            "setting would be reused as if it were this run's): " + "; ".join(differences)
        )


# ---------------------------------------------------------------------------------------
# the run lifecycle
# ---------------------------------------------------------------------------------------

def _resume_record(ctx: ScenarioContext, run_dir: Path, files: Mapping[str, Path], input_md5: Mapping[str, str], numeric: Mapping[str, Any]) -> RunRecord:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise common.EntryError(f"--resume {run_dir}: no manifest.json there")
    manifest = common.read_json(manifest_path)
    if manifest.get("stage") != STAGE:
        raise common.EntryError(f"--resume {run_dir}: the run is stage {manifest.get('stage')!r}, this entry writes {STAGE!r}")
    if manifest.get("exit_status") == "ok":
        raise common.EntryError(f"--resume {run_dir}: the run finished normally; nothing to resume")
    current = {name: file_md5(path) for name, path in files.items()}
    current.update(input_md5)
    recorded = manifest.get("input_md5", {})
    changed = sorted(k for k, v in current.items() if recorded.get(k) != v)
    if changed:
        raise common.EntryError(f"--resume {run_dir}: inputs changed since the run started: {changed}")
    if manifest.get("nmi_source_sha256") != common.source_hash():
        raise common.EntryError(f"--resume {run_dir}: the nmi/ source changed since the run started (code must be identical)")
    recorded_numeric = manifest.get("numeric")
    if not isinstance(recorded_numeric, dict) or "planned" not in recorded_numeric:
        raise common.EntryError(f"--resume {run_dir}: the manifest has no numeric record, so the numeric settings cannot be compared; start the run again under a new --tag")
    differences = common.numeric_differences(recorded_numeric["planned"], numeric)
    if differences:
        raise common.EntryError(f"--resume {run_dir}: the numeric settings changed since the run started: " + "; ".join(differences))
    start = now_local()
    manifest.update(exit_status="running", end_time=None, exit_code=None, error=None)
    manifest.setdefault("resumes", []).append({"start_time": iso(start), "command_line": list(ctx.argv)})
    aborted = run_dir / "aborted.json"
    if aborted.is_file():
        aborted.rename(run_dir / f"aborted.{len(manifest['resumes'])}.json")
    record = RunRecord(run_dir=run_dir, manifest=manifest, _start=start)
    record.write_manifest()
    return record


@contextlib.contextmanager
def open_scenario_run(
    ctx: ScenarioContext, *, files: Mapping[str, Path], input_md5: Mapping[str, str], numeric: Mapping[str, Any], extra_manifest: Mapping[str, Any],
) -> Iterator[common.RunHandle]:
    """Open (or resume) the run directory and guard the batch (the lifecycle of ``common.open_run``, with the inputs of the scenario): the manifest gets the md5 of the configuration, the prompt file, this
    entry and the adapter, the pool registry, the numeric block and the extra keys; SIGTERM writes aborted.json; the ``with`` block closes the record."""
    planned = common.json_safe(numeric)
    resumed = bool(ctx.args.resume)
    if resumed:
        record = _resume_record(ctx, Path(ctx.args.resume), files, input_md5, planned)
    else:
        if not ctx.tag:
            raise common.EntryError("--tag is required for a real run")
        record = begin_run(
            stage=STAGE, tag=ctx.tag, protocol_version=ctx.cfg.protocol_version, config_revision=ctx.cfg.revision, results=ctx.results_root, argv=ctx.argv, input_files=files, input_md5=input_md5,
            seed_keys=(), parents=(),
        )
        record.manifest["numeric"] = {"planned": planned, "actual": {}}
        record.write_manifest()
    record.manifest.update(extra_manifest)
    record.write_manifest()
    restore = record.install_sigterm_handler()
    handle = common.RunHandle(
        record=record, guard=SourceHashGuard(), parts=common.PartStore(record.run_dir / "parts"), resumed=resumed, run_dir=record.run_dir, ctx=ctx,  # type: ignore[arg-type]
    )
    try:
        with record:
            yield handle
    finally:
        restore()


# ---------------------------------------------------------------------------------------
# raw outputs
# ---------------------------------------------------------------------------------------

def gzip_bytes(lines: Sequence[Mapping[str, Any]]) -> bytes:
    """The raw lines as one gzip member with a fixed header (no time stamp), so that the same lines always give the same bytes."""
    payload = "".join(json.dumps(line, sort_keys=True, ensure_ascii=True) + "\n" for line in lines).encode("ascii")  # ASCII escapes: a reply with a lone surrogate or a control character still writes
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0, compresslevel=6) as handle:
        handle.write(payload)
    return buffer.getvalue()


def write_raw_part(run_dir: Path, unit_id: str, lines: Sequence[Mapping[str, Any]]) -> Path:
    target = paths.assert_off_root_partition(run_dir / RAW_PARTS / f"{paths.validate_name('unit', unit_id)}.jsonl.gz")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_bytes(gzip_bytes(lines))
    os.replace(tmp, target)
    return target


def assemble_raw(run_dir: Path, unit_ids: Sequence[str]) -> dict[str, Any]:
    """``raw_outputs.jsonl.gz``: the gzip members of the units, in unit order (a concatenation of members is a gzip file). Returns its md5 and the number of lines."""
    data = b"".join((run_dir / RAW_PARTS / f"{u}.jsonl.gz").read_bytes() for u in unit_ids)
    target = run_dir / RAW_FILE
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, target)
    n_lines = 0
    with gzip.open(target, "rb") as handle:
        for _ in handle:
            n_lines += 1
    return {"file": RAW_FILE, "md5": hashlib.md5(data, usedforsecurity=False).hexdigest(), "bytes": len(data), "n_lines": n_lines}


# ---------------------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------------------

def run_unit(ctx: ScenarioContext, bh: common.BackendHandle, renderer: protocol.Renderer, spec: UnitSpec, items: Sequence[Any], digit_ids: Sequence[int] | None) -> protocol.UnitOutput:
    rows = [items[i] for i in spec.rows]
    phase, cfg = ctx.args.phase, ctx.cfg
    if phase == "expert":
        return protocol.expert_unit(bh.backend, renderer, ctx.book, ctx.task, rows, cfg.settings, unit_id=spec.unit_id)
    if phase == "answer":
        assert digit_ids is not None
        return protocol.answer_unit(bh.backend, renderer, ctx.book, ctx.task, rows, cfg.settings, digit_ids, unit_id=spec.unit_id)
    assert spec.condition is not None and cfg.a_pct is not None
    return protocol.decide_unit(bh.backend, renderer, ctx.book, ctx.task, rows, spec.condition, cfg.a_pct, cfg.settings, unit_id=spec.unit_id)


def finish_tables(
    ctx: ScenarioContext, run: common.RunHandle, phase: str, unit_ids: Sequence[str], conds: Sequence[Condition], n_items: int,
) -> tuple[pa.Table, dict[str, Any]]:
    """Assemble the units into the table of the phase, write it with its summary, and (answer) run the calibration. Returns the table and the summary."""
    schema = protocol.SCHEMAS[phase]
    table = run.parts.concat(unit_ids).select(schema.names).cast(schema)
    expected = n_items * (len(conds) if phase == "decide" else 1)
    if table.num_rows != expected:
        raise common.EntryError(f"the {phase} table has {table.num_rows} rows, expected {expected}")
    if phase == "expert":
        result = summary.expert_summary(table)
    elif phase == "answer":
        ok = np.array([s == protocol.OK for s in table.column("status").to_pylist()])
        keys = [k for k, o in zip(table.column("item_key").to_pylist(), ok, strict=True) if o]
        c = np.array([math.nan if v is None else v for v, o in zip(table.column("c").to_pylist(), ok, strict=True) if o], dtype=np.float64)
        correct = np.array([bool(v) for v, o in zip(table.column("correct").to_pylist(), ok, strict=True) if o])
        mass = np.array([v for v, o in zip(table.column("digit_mass").to_pylist(), ok, strict=True) if o], dtype=np.float64)
        calibration_result, error = None, None
        try:
            calibration_result = calibration.calibrate(keys, c, correct, mass, n_registered=table.num_rows)
        # the table is written either way; the summary says that the calibration did not run, and the exit code is 1
        except Exception as exc:  # noqa: BLE001
            error = type(exc).__name__
        fold = [None] * table.num_rows
        z = [None] * table.num_rows
        if calibration_result is not None:
            positions = np.flatnonzero(ok)
            for pos, i in enumerate(positions):
                fold[int(i)] = int(calibration_result.fold_index[pos]) if calibration_result.fold_index[pos] >= 0 else None
                zi = calibration_result.z_oof[pos]
                z[int(i)] = float(zi) if np.isfinite(zi) else None
        table = table.append_column("fold", pa.array(fold, type=pa.int64())).append_column("z_oof", pa.array(z, type=pa.float64()))
        result = summary.answer_summary(table.select(schema.names), calibration_result, calibration_error=error)
    else:
        result = summary.decide_summary(table, [c.label for c in conds])
    common.write_parquet(run.run_dir / f"{phase}.parquet", table)
    return table, result


def execute(ctx: ScenarioContext, hooks: ScenarioHooks) -> dict[str, Any]:
    args, cfg = ctx.args, ctx.cfg
    phase = args.phase
    entry = cfg.model(args.model, phase)
    pool = pool_of(args)
    conds = condition_list(args, ctx.book)
    if phase == "decide" and cfg.a_pct is None:
        raise common.EntryError(f"the decide phase states the expert's accuracy in its prompts and a_pct is null in {cfg.path.name}: run the expert phase and freeze the integer percent in the configuration first")
    if args.limit_items is not None and args.limit_items < 1:
        raise common.EntryError("--limit-items must be positive")
    source_sha = ctx.split.verify_input(triviaqa.INPUT_NAME, ctx.source)["sha256"]
    items = ctx.task.load_pool(ctx.split, pool, ctx.source, results_root=ctx.results_root)
    if args.limit_items:
        items = items[: args.limit_items]
    ctx.split.assert_no_sealed([i.item_key for i in items], context=pool)
    tokenizer = load_tokenizer_for(hooks, entry)
    renderer = renderer_of(ctx, entry, tokenizer)
    digit_ids = readout.resolve_digit_ids(tokenizer) if phase == "answer" else None
    fingerprints = rendering_fingerprints(ctx, renderer, phase, conds, cfg.a_pct)
    units = plan_units(phase, len(items), conds, cfg.unit_items)
    identity = scenario_identity(ctx, entry, phase, pool, [i.item_key for i in items], conds, units, renderer, fingerprints, digit_ids, source_sha, model_fingerprint_of(entry))
    check_resume_identity(args, identity)  # exit 2 BEFORE any finished unit of the run being resumed is reused
    files = bound_files(cfg)
    input_md5 = ctx.split.manifest_md5s([pool])
    extra_manifest = {"exploratory": True, "development_reason": cfg.development_reason, "scenario_run": identity, "scenario_config": cfg.as_dict(), "source_sha256": source_sha}
    opener = hooks.opener or open_vllm
    if hooks.opener is None:
        common.require_gpu_memory_fraction(args)  # before a run directory is made, not inside the run
    plan = common.numeric_plan(backend_plan(ctx, entry, {"unit_items": cfg.unit_items, "phase": phase}))
    with open_scenario_run(ctx, files=files, input_md5=input_md5, numeric=plan, extra_manifest=extra_manifest) as run:
        bh = opener(ctx, entry, tokenizer)
        try:
            run.confirm_numeric(args.backend, bh.numeric_actual())
            if hooks.model_files and not run.resumed:
                common.write_model_files(run, {"id": entry.id, "path": entry.path, "chat_template_kwargs": dict(entry.chat_template_kwargs)}, bh)
            done = skipped = 0
            t0 = time.time()
            for spec in units:
                if run.parts.done(spec.unit_id):
                    skipped += 1
                    continue
                run.check_source()
                out = run_unit(ctx, bh, renderer, spec, items, digit_ids)
                write_raw_part(run.run_dir, spec.unit_id, [{**line} for line in out.raw])
                common.write_json(run.run_dir / UNIT_STATS / f"{spec.unit_id}.json", {"unit": spec.unit_id, "seconds": out.seconds, "generation": out.stats.as_dict()})
                run.parts.write(spec.unit_id, out.table(protocol.SCHEMAS[phase]))  # the part is written last: a unit is done when its table is there
                done += 1
                n_missing = sum(s == protocol.MISSING for s in out.columns["status"])
                common.log(f"{spec.unit_id}: {len(spec.rows)} items, {n_missing} missing, {out.seconds:.1f} s")
            wall = time.time() - t0
        finally:
            bh.close()
        unit_ids = [u.unit_id for u in units]
        totals = protocol.GenStats()
        for unit_id in unit_ids:  # all units of the run, also those written before a resume
            totals = totals.merge(protocol.GenStats(**{k: v for k, v in common.read_json(run.run_dir / UNIT_STATS / f"{unit_id}.json")["generation"].items()}))
        table, result = finish_tables(ctx, run, phase, unit_ids, conds, len(items))
        raw = assemble_raw(run.run_dir, unit_ids)
        result.update({
            "phase": phase, "model": entry.id, "pool": pool, "task": cfg.task, "n_items": len(items), "condition_labels": [c.label for c in conds], "a_pct": identity["a_pct"], "backend": args.backend,
            "settings": cfg.settings.as_dict(), "units": {"n": len(units), "done": done, "skipped": skipped, "wall_s": wall}, "generation": totals.as_dict(), "raw_outputs": raw,
            "exploratory": True, "development_reason": cfg.development_reason,
        })
        result = summary.jsonable(result)
        common.write_json(run.run_dir / f"{phase}_summary.json", result)
        run.record.manifest["raw_outputs"] = raw
        run.record.manifest["generation"] = totals.as_dict()
        run.record.manifest["phase_summary"] = {k: result[k] for k in ("n_registered", "n_missing", "missing_rate") if k in result}
        run.record.write_manifest()
        for unit_id in unit_ids:  # the parts of the raw file are inside it now
            with contextlib.suppress(OSError):
                (run.run_dir / RAW_PARTS / f"{unit_id}.jsonl.gz").unlink()
        with contextlib.suppress(OSError):
            (run.run_dir / RAW_PARTS).rmdir()
        report = {
            "run_dir": str(run.run_dir), "phase": phase, "model": entry.id, "n_rows": int(table.num_rows), "n_missing": result.get("n_missing"), "missing_rate": result.get("missing_rate"),
            "generation": totals.as_dict(), "raw_outputs": {k: raw[k] for k in ("md5", "n_lines")},
        }
        if phase == "answer":
            report.update({
                "z_valid": result["z_valid"], "z_invalid_reasons": result["z_invalid_reasons"], "calibration_error": result["calibration_error"], "quality_gate_pass": result["quality_gate"]["pass"],
            })
        if phase == "decide":
            report["n_untestable"] = result["n_untestable"]
        return report


def dry_run(ctx: ScenarioContext, hooks: ScenarioHooks) -> int:
    dry = common.DryRun(ctx)  # type: ignore[arg-type]
    args, cfg = ctx.args, ctx.cfg
    try:
        phase = args.phase
        entry = cfg.model(args.model, phase)
        pool = pool_of(args)
        conds = condition_list(args, ctx.book)
        if args.limit_items is not None and args.limit_items < 1:
            raise common.EntryError("--limit-items must be positive")
    except USER_ERRORS as error:
        dry.check("arguments and inputs", False, str(error))
        return dry.emit()
    dry.check("configuration", True, {"path": str(cfg.path), "md5": cfg.md5, "version": cfg.version, "revision": cfg.revision, "expected_md5": args.expect_config_md5})
    dry.check("prompt file", True, ctx.book.describe())
    dry.check("pool registry", True, {"dir": str(ctx.split.root), "version": ctx.split.version, "file_md5": ctx.split.file_md5s, "pool": {"name": pool, "n": len(ctx.split.keys(pool)), "md5": ctx.split.pool_md5(pool)}})
    try:
        record = ctx.split.verify_input(triviaqa.INPUT_NAME, ctx.source)
        dry.check("source file hash", True, {"path": str(ctx.source), "sha256": record["sha256"], "n_rows": record["n_rows"]})
        source_sha = record["sha256"]
    except (InputHashMismatchError, OSError) as error:
        dry.check("source file hash", False, str(error))
        source_sha = "unverified"
    dry.check("sealed pool", pool == triviaqa.DEV_POOL, {"pool": pool, "refused": [triviaqa.SEALED_POOL]})
    dry.check("a_pct (decide only)", phase != "decide" or cfg.a_pct is not None, {"a_pct": cfg.a_pct, "needed_for": "decide"})
    model_dir = Path(entry.path)
    detail: dict[str, Any] = {"id": entry.id, "path": str(model_dir), "family": entry.family}
    readable = True
    if model_dir.is_dir():
        try:
            fp = model_files.model_fingerprint(model_dir)
            detail.update(fingerprint_md5=fp["md5"], declared=fp["declared"], n_weight_files=len(fp["weight_files"]), n_tokenizer_files=len(fp["tokenizer_files"]))
        except OSError as error:
            readable = False
            detail["fingerprint_error"] = f"{type(error).__name__}: {error}"
    dry.check("model directory", readable and model_dir.is_dir() and (model_dir / "config.json").is_file(), detail)
    if args.backend == "vllm_offline" and hooks.opener is None:
        try:
            common.require_gpu_memory_fraction(args)
            dry.check("offline vLLM memory fraction", True, {"gpu_mem_util": args.gpu_mem_util, "max_model_len": args.max_model_len})
        except common.EntryError as error:
            dry.check("offline vLLM memory fraction", False, str(error))
    items_n = len(ctx.split.keys(pool))
    if args.limit_items:
        items_n = min(items_n, args.limit_items)
    units = plan_units(phase, items_n, conds, cfg.unit_items)
    try:
        tokenizer = load_tokenizer_for(hooks, entry)
        renderer = renderer_of(ctx, entry, tokenizer)
        digit_ids = readout.resolve_digit_ids(tokenizer) if phase == "answer" else None
        fingerprints = rendering_fingerprints(ctx, renderer, phase, conds, cfg.a_pct)
        dry.check("chat template", True, {
            "tool_injection": renderer.injection, "system_role": render.system_role(tokenizer, renderer.chat_kwargs), "digit_token_ids": None if digit_ids is None else list(digit_ids),
            "n_renderings": len(fingerprints), "rendering_md5": fingerprints,
        })
    except (*USER_ERRORS, KeyError, TypeError) as error:
        dry.check("chat template", False, f"{type(error).__name__}: {error}")
    try:
        paths.assert_off_root_partition(ctx.results_root)
        off_root = True
    except paths.PathPolicyError as error:
        off_root = False
        dry.check("output directory (off the root partition)", False, str(error))
    if not off_root:
        pass
    elif args.resume:
        used = Path(args.resume)
        dry.check("resume directory", (used / "manifest.json").is_file(), str(used))
    else:
        used_tag = common.tag_in_use(ctx.results_root, STAGE, ctx.tag or "dryrun-tag")
        dry.check("output directory", used_tag is None, {"results_root": str(ctx.results_root), "stage": STAGE, "tag": ctx.tag, "tag_already_used_by": used_tag})
    n_prompts = items_n * (len(conds) if phase == "decide" else (2 if phase == "answer" else 1))
    dry.note("workload", {
        "phase": phase, "model": entry.id, "pool": pool, "items": items_n, "conditions": [c.label for c in conds], "prompts": n_prompts, "units": len(units), "unit_items": cfg.unit_items,
        "settings": cfg.settings.as_dict(), "backend": args.backend, "stage": STAGE,
    })
    dry.note("exploratory", {"exploratory": True, "development_reason": cfg.development_reason})
    dry.note("numeric_plan", common.numeric_plan(backend_plan(ctx, entry, {"unit_items": cfg.unit_items, "phase": phase})))
    dry.note("source_sha256", source_sha)
    return dry.emit()


def main(argv: Sequence[str] | None = None, hooks: ScenarioHooks | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    hooks = hooks or ScenarioHooks()
    args = make_parser().parse_args(argv)
    t0 = time.time()
    try:
        pool_of(args)  # the sealed pool and any other name exit 2 before the configuration, the registry or a model is read, also for --dry-run
        ctx = build_context(args, argv=argv)
        if args.dry_run:
            return dry_run(ctx, hooks)
        outcome = execute(ctx, hooks)
    except USER_ERRORS as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except common.SourceChangedError as error:
        return common.stopped_by_source_change(error)
    outcome["wall_s"] = time.time() - t0
    print(json.dumps(common.json_safe(outcome), indent=2, sort_keys=True))
    return EXIT_FAILED_CHECK if outcome.get("calibration_error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
