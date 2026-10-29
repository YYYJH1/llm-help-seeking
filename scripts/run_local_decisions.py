#!/usr/bin/env python
"""The decision-only arm for one local model: a greedy generated reply instead of digit probabilities.

The P-ask prompt of ``run_decisions`` (the same functions: ``rd.condition_job`` -> ``common.ask_texts`` -> ``render_chat`` with the ``chat_template_kwargs`` of the model entry, the layout
of the prompt file, no assistant prefix), as token ids, goes to an offline vLLM engine that decodes greedily (temperature 0) at most four new tokens, stops where vLLM stops that model
and returns no probability. The reply is read by ``nmi.decision_only.read_ids``: the tokenizer object of the entry decodes the WHOLE generated sequence with the tokens it marks as special
skipped, ``str.strip()``, first character: "1".."4" an answer of the model itself, "5" a request for help, anything else a format failure. The text that the engine returns is never used.
The probability readout of ``run_decisions`` is not touched.

Only the development pool is read. ``--exploratory`` is the door of ``run_decisions`` and the only one this entry has: with it ``--pools mmlu_redux_cal`` is required and is the only
pool taken, ``--grids`` must be given (psychometric, f1, screening; not operating) and ``--layout`` may be ``block_position``. Every holdout pool (mmlu_redux_holdout is the pool of the
2,291 questions of the first batch) is refused with an error, whatever markers exist; this entry has no way to release one. That path belongs to another entry.

Diagnostic and smoke runs (another grid, layout or paraphrase, ``--limit-items``) write the same completion files and end ``ok``. The full specified run is told from them by
``protocol_run`` in the summary: true only if the arguments are the specified ones (pool, grid psychometric, paraphrase a00, standard layout, no pilot subset) AND the table itself holds exactly
the whole pool (the fixed key list, 1,000 items) under the 54 conditions and no other trial (``protocol_run_checks``, computed from the table before the summary is written).
A run is complete when its manifest says ``ok``, its summary is there and ``protocol_run`` is true.

Output: ``results/decide/<time>_<tag>/`` with ``manifest.json`` (stage ``decide``), ``parts/*.parquet`` (one file per unit of ``--conditions-per-unit`` conditions, written atomically,
a resumed run skips the finished units), ``local_decisions.parquet`` and ``local_decisions_summary.json`` (the last two are the completion record: they are written only after every unit
is on disk and verified, and the manifest says ``ok`` after them). Table columns: item_key, model, pool, paraphrase, layout, framing, a_H, c0, w_displayed, kappa, D_hat_s (as in
``decisions.parquet``), then category (answer, ask, format_failure), first_char (only "1".."5"), ask (bool), self_choice (1..4 for an answer, else null), generated_token_ids,
generated_text (the whole reply, decoded, special tokens skipped, not stripped), n_generated, finish_reason, first_special_position (index of the first special token, null if none),
non_special_after_special (a token that is not special follows the first special token; false without one), read_text (generated_text stripped: what the reading looked at), then
gt_position and grid. No question text and no prompt is stored.

The manifest records, besides what ``run_decisions`` records: ``input_md5.decision_arm_protocol`` (the md5 of the decision-arm protocol file), ``decide_run`` (the identity of the run and the constants
of ``nmi.decision_only``), ``decide_tokenization`` (how the prompt is encoded and the reply decoded; the size and md5 of the set of special token ids, as the tokenizer marks them and as
``all_special_ids`` alone lists them) and ``decide_engine`` (the complete sampling parameters, and the end-of-text ids of the engine as far as it reports them). A resumed run must match
the first call in the identity, the special token sets and the sampling parameters. ``decide_model_files`` holds the sha256 of every file at the top level of the model directory
(weights, tokenizer files, chat template, configuration): a resumed run is refused, before a unit is reused or made, if one of them is new, gone or different. The tokens that end a
generation must have been read from the engine: if a source cannot be read, the run ends as an error before the first dispatch. ``trust_remote_code`` in ``--llm-kwargs`` is refused.

Tokenizer, encoding and decoding go through ONE place, ``open_prompt_codec`` (see its docstring).

    python scripts/run_local_decisions.py --config CONFIG --results-root RR --exploratory --pools mmlu_redux_cal --paraphrases a00 --grids psychometric --model MODEL --gpu-mem-util F --tag TAG

Exit codes as in ``run_decisions``: 0 done, 1 a dry run found a problem, 2 unusable input or arguments (nothing was computed), 4 the source guard stopped the dispatch.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
for _p in (HERE.parent, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import _c2_common as common  # noqa: E402
import pyarrow as pa  # noqa: E402
import run_decisions as rd  # noqa: E402

from nmi import decision_only, paths  # noqa: E402
from nmi.data.registry import SplitError, is_holdout_pool  # noqa: E402
from nmi.prompts import quantize_w  # noqa: E402
from nmi.provenance import LabelExistsError, file_sha256, note_skipped  # noqa: E402
from nmi.readout.hf_backend import ReadoutError, encode_prompts  # noqa: E402

SCRIPT = "run_local_decisions"
STAGE = "decide"
TABLE_FILE = "local_decisions.parquet"  # not decide.parquet: run_scenario writes that name in its own stage
SUMMARY_FILE = "local_decisions_summary.json"
DEVELOPMENT_POOLS = (rd.EXPLORATORY_POOL,)  # the pools run_decisions --exploratory can read
DEVELOPMENT_REASON = "development run of the decision-only arm"
# the leading columns of the table: those of decisions.parquet that identify the trial and the condition, then the reading; whatever else the unit builder returns follows (gt_position, grid)
FIRST_COLUMNS = (
    "item_key", "model", "pool", "paraphrase", "layout", "framing", "a_H", "c0", "w_displayed", "kappa", "D_hat_s",
    "category", "first_char", "ask", "self_choice", "generated_token_ids", "generated_text", "n_generated", "finish_reason", "first_special_position", "non_special_after_special", "read_text",
)
TRIAL_KEY = ("item_key", "pool", "paraphrase", "layout", "framing", "a_H", "c0", "w_displayed", "kappa", "D_hat_s")  # one trial per value of these
CONDITION_KEY = TRIAL_KEY[1:]
FORBIDDEN_LLM_KWARGS = ("trust_remote_code",)  # every model is loaded without it, whatever the value
MODEL_FILES_CACHE_MIN_BYTES = 8 << 20  # a file smaller than this is hashed every time; the shared hash cache (path, size, mtime) is for the weight shards
MODEL_FILES_RECIPE = "sha256 of every regular file at the top level of the model directory, by file name (weights, tokenizer files, chat template, configuration, anything else)"


@dataclass
class LocalDecisionHooks:
    """Test seams, as ``DecisionHooks`` of ``run_decisions``: ``opener(ctx, model_id)`` returns the ``BackendHandle`` (default ``open_engine``); ``model_files=False`` skips ``llm/model.json``."""

    opener: Callable[[common.Context, str], common.BackendHandle] | None = None
    model_files: bool = True


# ---------------------------------------------------------------------------------------
# the tokenizer, the encoding of the prompts and the decoding of the replies: ONE place
# ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class PromptCodec:
    """What the entry needs from tokenizers. ``chat`` renders the chat template (``apply_chat_template``): it is what the prompts are made with and what ``BackendHandle.tokenizer`` is.
    ``ids`` turns the rendered text into token ids AND the generated token ids back into text (the reading decodes the whole reply with it, special tokens skipped), and its
    special tokens (``decision_only.special_id_set``) are the ones that the manifest and two columns of the table refer to. By default they are the same object; a model whose prompts are encoded with its
    ``tokenizer.json`` gets another object in ``ids`` and nothing else changes."""

    chat: Any
    ids: Any
    special_ids: frozenset[int]
    description: Mapping[str, Any]

    def encode(self, texts: Sequence[str]) -> list[list[int]]:
        return encode_prompts(self.ids, texts)

    def read(self, token_ids: Sequence[int]) -> decision_only.Reading:
        return decision_only.read_ids(token_ids, self.ids, special_ids=self.special_ids)


def _size(tokenizer: Any) -> int | None:
    try:
        return int(len(tokenizer))
    except Exception:  # noqa: BLE001
        return None


def codec_from_tokenizer(chat: Any, ids: Any = None, *, source: str) -> PromptCodec:
    """A codec from a tokenizer object (``ids`` defaults to ``chat``). ``source`` says where the object comes from; it goes into the manifest."""
    ids = chat if ids is None else ids
    special = decision_only.special_id_set(ids)
    description = {
        "source": source,
        "rendering": {"class": type(chat).__name__},
        "ids": {"class": type(ids).__name__, "same_object_as_rendering": ids is chat, "vocabulary_size": _size(ids)},
        "encoding": "nmi.readout.hf_backend.encode_prompts: tokenizer(text, add_special_tokens=False)['input_ids']",
        "decoding": {"whole_generated_sequence": True, **dict(decision_only.DECODE_KWARGS)},
        "special_token_ids": decision_only.special_ids_record(special),
        "all_special_ids_attribute": decision_only.special_ids_record(decision_only.attribute_special_ids(ids)),
    }
    return PromptCodec(chat, ids, special, description)


def open_prompt_codec(model_dir: Path) -> PromptCodec:
    """The ONE place where this entry gets a tokenizer, and so where the prompts are encoded and the replies are decoded. At present ``common.load_tokenizer(model_dir)`` (what
    ``open_backend`` does for the probability readout) renders the chat template, encodes with ``encode_prompts`` (``add_special_tokens=False``) and decodes the generated ids.
    For a model whose prompts are rendered with the chat template but encoded with the ``tokenizer.json`` of the model directory, return
    ``codec_from_tokenizer(rendering_tokenizer, ids=fast_tokenizer, source=...)``: the encoding, the decoding of the reply, the special token ids and the description in
    the manifest then all follow ``ids``, and the rendering stays on ``chat``."""
    tokenizer = common.load_tokenizer(model_dir)
    return codec_from_tokenizer(tokenizer, source="common.load_tokenizer(model_dir): AutoTokenizer.from_pretrained(model_dir, local_files_only=True)")


# ---------------------------------------------------------------------------------------
# the engine
# ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Generated:
    """What the engine returned for one prompt."""

    token_ids: tuple[int, ...]
    finish_reason: str | None


def _repr(value: Any) -> str:
    """``repr`` without memory addresses, so that the same object gives the same text in two processes (a resumed run compares them)."""
    return re.sub(r"0x[0-9a-fA-F]+", "0x", repr(value))


def _plain(value: Any) -> Any:
    """A JSON value for whatever a sampling parameter or a generation config holds (sets and frozensets become sorted lists, enums their name)."""
    if isinstance(value, Enum):  # (before the numbers: an IntEnum is an int)
        return value.name
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        try:
            return [_plain(v) for v in sorted(value)]
        except TypeError:
            return sorted(_repr(v) for v in value)
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    names = getattr(value, "__struct_fields__", None)
    if names:
        return {name: _plain(getattr(value, name, None)) for name in names}
    return _repr(value)


def describe_sampling_params(params: Any) -> dict[str, Any]:
    """Every field of the sampling parameters (``vllm.SamplingParams`` is a ``msgspec.Struct``; a stand-in is read by its attributes), as plain JSON values."""
    names = getattr(params, "__struct_fields__", None)
    if not names:
        names = tuple(vars(params)) if hasattr(params, "__dict__") else ()
    out: dict[str, Any] = {}
    for name in names:
        try:
            out[str(name)] = _plain(getattr(params, name))
        except Exception as error:  # noqa: BLE001
            out[str(name)] = f"<unreadable: {type(error).__name__}>"
    return out or {"repr": _repr(params)}


def _probe(errors: dict[str, str], name: str, *getters: Callable[[], Any]) -> Any:
    """The value of the first getter that works; when none does, the error of the last one is left in ``errors`` and the value is None."""
    last = None
    for getter in getters:
        try:
            return getter()
        except Exception as error:  # noqa: BLE001
            last = f"{type(error).__name__}: {error}"
    if last is not None:
        errors[name] = last
    return None


def _ids_of(value: Any) -> list[int]:
    if isinstance(value, int) and not isinstance(value, bool):
        return [int(value)]
    if isinstance(value, (list, tuple, set, frozenset)):
        return [int(v) for v in value if isinstance(v, int) and not isinstance(v, bool)]
    return []


def engine_stop_report(engine: Any, sampling_params: Any, special_ids: frozenset[int] | None = None) -> dict[str, Any]:
    """The tokens that end a generation, as far as the engine reports them. vLLM 0.30.0 (v1/engine/input_processor.py, SamplingParams.update_from_generation_config) stops a request at the
    end-of-text id of the tokenizer and at the ``eos_token_id`` (an int or a list) of ``generation_config.json``, plus the ``stop_token_ids`` and ``stop`` strings of the request; the entry
    gives neither. Each source is read on its own, and what cannot be read is left out and named in ``errors`` (nothing here can stop a run). ``request`` is vLLM's own computation
    applied to a clone of the sampling parameters (the request is cloned inside the engine, so the original does not show it). ``effective_stop_token_ids`` is None unless both
    sources were read: a source that could not be read is not an empty one. ``effective_stop_ids_outside_special_set`` lists the stop ids that the tokenizer does not mark as special (decode does not skip such an id, so it stays in the text); None when the stop set or the special set is not known."""
    errors: dict[str, str] = {}
    eos = _probe(
        errors, "tokenizer_eos_token_id",
        lambda: engine.llm_engine.renderer.get_eos_token_id(), lambda: engine.renderer.get_eos_token_id(), lambda: engine.get_tokenizer().eos_token_id,
    )
    fields = _probe(
        errors, "generation_config_fields",
        lambda: engine.llm_engine.input_processor.generation_config_fields, lambda: engine.input_processor.generation_config_fields, lambda: engine.model_config.try_get_generation_config(),
    )
    fields = dict(fields) if isinstance(fields, Mapping) else None
    config_eos = None if fields is None else fields.get("eos_token_id")
    from_sources = sorted(set(_ids_of(eos)) | set(_ids_of(config_eos)))
    both_read = "tokenizer_eos_token_id" not in errors and "generation_config_fields" not in errors  # an unread source is not an empty one
    request = None
    if both_read:
        try:
            probe = sampling_params.clone()
            probe.update_from_generation_config(fields or {}, eos)
            request = {"eos_token_id": probe.eos_token_id, "stop_token_ids": sorted(_ids_of(probe.stop_token_ids)), "all_stop_token_ids": sorted(_ids_of(probe.all_stop_token_ids))}
        except Exception as error:  # noqa: BLE001
            errors["request"] = f"{type(error).__name__}: {error}"
    effective = (request["all_stop_token_ids"] if request is not None else from_sources) if both_read else None
    return {
        "rule": "vLLM 0.30.0: a request stops at the tokenizer's end-of-text id and at the eos_token_id of generation_config.json; the entry sets no stop string and no stop token id",
        "tokenizer_eos_token_id": eos,
        "generation_config_eos_token_id": config_eos,
        "generation_config_fields": None if fields is None else _plain(fields),
        "request": request,
        "effective_stop_token_ids": effective,
        "stop_set_known": effective is not None,
        "effective_stop_ids_outside_special_set": None if effective is None or special_ids is None else sorted(set(effective) - set(special_ids)),
        "errors": errors,
    }


def engine_report(engine: Any, sampling_params: Any, special_ids: frozenset[int] | None = None) -> dict[str, Any]:
    return {"sampling_params": describe_sampling_params(sampling_params), "stop": engine_stop_report(engine, sampling_params, special_ids)}


class GreedyBackend:
    """An offline vLLM engine (or anything with the same ``generate``) asked for greedy replies to token-id prompts. ``codec`` encodes the rendered texts; the engine's own text is never used."""

    def __init__(self, engine: Any, codec: PromptCodec, sampling_params: Any, *, use_tqdm: bool = False) -> None:
        self.engine = engine
        self.codec = codec
        self.sampling_params = sampling_params
        self.use_tqdm = use_tqdm
        self.report = engine_report(engine, sampling_params, codec.special_ids)

    def generate(self, texts: Sequence[str]) -> list[Generated]:
        """The replies in input order: one generation per text, as many token ids as the engine returned (at most ``MAX_NEW_TOKENS``) and the finish reason."""
        if len(texts) == 0:
            return []
        id_lists = self.codec.encode(texts)
        outputs = self.engine.generate([{"prompt_token_ids": ids} for ids in id_lists], self.sampling_params, use_tqdm=self.use_tqdm)
        if len(outputs) != len(id_lists):
            raise ReadoutError(f"engine returned {len(outputs)} outputs for {len(id_lists)} prompts")
        replies = []
        for sent, output in zip(id_lists, outputs, strict=True):
            echoed = getattr(output, "prompt_token_ids", None)
            if echoed is not None and [int(t) for t in echoed] != sent:
                raise ReadoutError("the engine returned an output for a different prompt (order or content changed)")
            completions = getattr(output, "outputs", None)
            if not completions:
                raise ReadoutError("RequestOutput has no outputs")
            completion = completions[0]
            generated = getattr(completion, "token_ids", None)
            if generated is None or len(generated) < 1:
                raise ReadoutError("outputs[0].token_ids is empty: no token was generated")
            if len(generated) > decision_only.MAX_NEW_TOKENS:
                raise ReadoutError(f"the engine returned {len(generated)} tokens for a request of at most {decision_only.MAX_NEW_TOKENS}")
            replies.append(Generated(tuple(int(t) for t in generated), getattr(completion, "finish_reason", None)))
        return replies


def build_handle(
    model_dir: str | Path, codec: PromptCodec, engine: Any, sampling_params: Any, *, serve_params: Mapping[str, Any] | None = None, close_engine: Callable[[Any], None] | None = None,
) -> common.BackendHandle:
    """The handle of the entry around an engine: ``tokenizer`` is the rendering tokenizer (what ``rd.condition_job`` renders with), ``backend`` the ``GreedyBackend``."""
    backend = GreedyBackend(engine, codec, sampling_params)
    closer = None if close_engine is None else (lambda: close_engine(backend.engine))
    return common.BackendHandle(backend, codec.chat, Path(model_dir), "vllm_offline", dict(serve_params or {}), closer=closer)


def open_engine(ctx: common.Context, model_id: str) -> common.BackendHandle:
    """The default opener: the codec, then the offline engine with the keyword arguments of ``common.vllm_llm_kwargs`` (no ``trust_remote_code``) and the sampling parameters of ``decision_only.sampling_kwargs``."""
    args = ctx.args
    entry = common.device_entry(ctx.cfg, model_id)
    model_dir = Path(entry["path"])
    common.require_gpu_memory_fraction(args)  # before anything is read
    check_llm_kwargs(args)
    codec = open_prompt_codec(model_dir)
    llm_kwargs = common.vllm_llm_kwargs(args)
    reject_forbidden_llm_kwargs(llm_kwargs)  # (the merged dictionary again: nothing that reaches vllm.LLM names trust_remote_code)
    from vllm import LLM, SamplingParams

    engine = LLM(model=str(model_dir), **llm_kwargs)
    return build_handle(
        model_dir, codec, engine, SamplingParams(**decision_only.sampling_kwargs()), serve_params={"llm_kwargs": llm_kwargs}, close_engine=lambda e: common.shutdown_offline_engine(e),
    )


# ---------------------------------------------------------------------------------------
# arguments and guards
# ---------------------------------------------------------------------------------------

def require_exploratory(args: argparse.Namespace) -> None:
    if not rd.is_exploratory(args):
        raise common.EntryError(
            "the decision-only arm has no registered run yet: its development runs go through --exploratory, the door of run_decisions "
            f"(--exploratory --pools {DEVELOPMENT_POOLS[0]} --grids psychometric); the run on the 2,291 questions of the first batch belongs to another entry"
        )


def pool_list(args: argparse.Namespace) -> list[str]:
    """The pools of the run: the development pool and nothing else. A holdout pool is refused first, with its own message, whatever markers exist and whether or not --exploratory is given."""
    requested = common.csv_list(args.pools)
    held = [p for p in requested if is_holdout_pool(p)]
    if held:
        raise common.EntryError(
            f"pool(s) {held} are holdout pools (mmlu_redux_holdout holds the 2,291 questions of the first batch): this entry never reads them. Reading them belongs to another entry, "
            "which is guarded by a release marker that is created by hand; this entry has no way to release them. "
            f"It reads the development pool {DEVELOPMENT_POOLS[0]} only"
        )
    require_exploratory(args)
    if requested != list(DEVELOPMENT_POOLS):
        given = "no --pools was given" if not requested else f"--pools {args.pools} was given"
        raise common.EntryError(
            f"the decision-only arm reads the one development pool {DEVELOPMENT_POOLS[0]} and needs --pools {DEVELOPMENT_POOLS[0]}; {given}: "
            "an evaluation pool, another pool or several pools are refused"
        )
    return common.validate_pools(requested)


def assert_development_pools(pools: Iterable[str]) -> None:
    """The guard again, right before a pool is loaded (the arguments were checked by ``pool_list``): nothing but a development pool reaches ``load_pool_items``."""
    for pool in pools:
        if is_holdout_pool(pool) or pool not in DEVELOPMENT_POOLS:
            raise common.EntryError(f"pool {pool!r} is not a development pool of the decision-only arm ({list(DEVELOPMENT_POOLS)}); it is not loaded")


def protocol_binding(args: argparse.Namespace) -> dict[str, str | None]:
    """The protocol file of the repository and its md5; ``--expect-protocol-md5`` stops the run unless it is exactly that."""
    path = paths.REPO_ROOT / decision_only.PROTOCOL_FILE
    if not path.is_file():  # the protocol text is not part of this release
        note_skipped(path, "protocol")
        return {"file": decision_only.PROTOCOL_FILE, "md5": None}
    md5 = common.file_md5(path)
    expected = getattr(args, "expect_protocol_md5", None)
    if expected and expected != md5:
        raise common.EntryError(f"{decision_only.PROTOCOL_FILE} has md5 {md5}, not the expected {expected}")
    return {"file": decision_only.PROTOCOL_FILE, "md5": md5}


def reject_forbidden_llm_kwargs(kwargs: Mapping[str, Any]) -> None:
    forbidden = sorted(str(k) for k in kwargs if k in FORBIDDEN_LLM_KWARGS)
    if forbidden:
        raise common.EntryError(
            f"{forbidden} in the keyword arguments of vllm.LLM: the decision-only arm loads every model without trust_remote_code, true or false; a key of that name is refused"
        )


def check_llm_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """``--llm-kwargs`` as a dictionary. It must be a JSON object, and it must not name ``trust_remote_code`` (any value: the admission is "loads without it", and the key is the way round it)."""
    text = getattr(args, "llm_kwargs", None)
    if not text:
        return {}
    try:
        value = json.loads(text)
    except ValueError as error:
        raise common.EntryError(f"--llm-kwargs is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise common.EntryError("--llm-kwargs must be a JSON object")
    reject_forbidden_llm_kwargs(value)
    return value


def model_files_fingerprint(model_dir: Path) -> dict[str, Any]:
    """The model input files by name and sha256: every regular file at the top level of the model directory (a link is followed: the files of a Hugging Face snapshot are links to blobs). Weight
    shards and other big files go through the shared hash cache of the entries (path, size, mtime), so that an unchanged shard is not read again; a small file is always hashed."""
    directory = Path(model_dir)
    if not directory.is_dir():
        raise common.EntryError(f"the model directory {directory} does not exist")
    cache_path = common.HASH_CACHE_PATH
    try:
        cache = common.read_json(cache_path) if cache_path.is_file() else {}
    except (OSError, ValueError):
        cache = {}
    changed = False
    files: dict[str, str] = {}
    for path in sorted(directory.iterdir(), key=lambda p: p.name):
        if not path.is_file():
            continue
        stat = path.stat()
        entry = cache.get(str(path)) if stat.st_size >= MODEL_FILES_CACHE_MIN_BYTES else None
        if entry and entry.get("size") == stat.st_size and entry.get("mtime_ns") == stat.st_mtime_ns:
            files[path.name] = entry["sha256"]
            continue
        files[path.name] = file_sha256(path)
        if stat.st_size >= MODEL_FILES_CACHE_MIN_BYTES:
            cache[str(path)] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": files[path.name]}
            changed = True
    if changed:
        common.write_json(cache_path, cache)
    return {"recipe": MODEL_FILES_RECIPE, "files": files}


def model_files_differences(recorded: Mapping[str, str], now: Mapping[str, str]) -> list[str]:
    """One line per file that is new, gone or different (the first twelve hex digits of the hashes)."""
    out = []
    for name in sorted(set(recorded) | set(now)):
        if name not in recorded:
            out.append(f"{name}: a file that was not there in the first session")
        elif name not in now:
            out.append(f"{name}: a file of the first session that is gone")
        elif recorded[name] != now[name]:
            out.append(f"{name}: content changed ({str(recorded[name])[:12]} -> {str(now[name])[:12]})")
    return out


def check_resume_model_files(args: argparse.Namespace, fingerprint: Mapping[str, Any]) -> None:
    """Before any finished unit is reused or a new one is made: the files of the model directory must be those of the first call (the chat template, the tokenizer files that encode the
    prompts and decode the replies, the model configuration, the weights). A unit made with other files would be reused as if it were made with these. ``EntryError`` (exit 2), and nothing of
    the run is touched. A run whose manifest holds no record of the files is resumed only if it has no unit on disk; the record is made then."""
    resume = getattr(args, "resume", None)
    if not resume:
        return
    path = Path(resume) / common.MANIFEST_FILE
    if not path.is_file():
        return  # open_run says so itself
    manifest = common.read_json(path)
    recorded = manifest.get("decide_model_files") if isinstance(manifest, dict) else None
    if not isinstance(recorded, dict) or not isinstance(recorded.get("files"), dict):
        parts = Path(resume) / "parts"
        if parts.is_dir() and any(parts.glob("*.parquet")):
            raise common.EntryError(
                f"--resume {resume}: the manifest of that run holds no record of the model files (decide_model_files) although units are on disk; that they were made with the files of the model "
                "directory as they are now cannot be shown, so the run is not resumed"
            )
        return
    differences = model_files_differences(recorded["files"], fingerprint["files"])
    if differences:
        shown = "; ".join(differences[:12]) + (f"; and {len(differences) - 12} more" if len(differences) > 12 else "")
        raise common.EntryError(
            f"--resume {resume}: the files of the model directory differ from those of the first session (the finished units were made with the old ones, a new unit would be made with the new ones): {shown}"
        )


def require_known_stop_set(report: Mapping[str, Any]) -> None:
    """The tokens that end a generation must have been read from the engine: an unread source is not an empty one. Raised before the first
    dispatch, so the run ends as an error and writes no completion record."""
    stop = report.get("stop") or {}
    if stop.get("effective_stop_token_ids") is None:
        errors = stop.get("errors") or {}
        unread = "; ".join(f"{name}: {text}" for name, text in sorted(errors.items())) or "no source could be read"
        raise common.EntryError(f"the tokens that end a generation could not be read from the engine, so no trial is generated and no completion record is written ({unread})")


def decide_identity(args: argparse.Namespace, pools: Sequence[str], grids: Sequence[str], paraphrases: Sequence[str]) -> dict[str, Any]:
    """What a run is beyond the layout and the conditions of ``rd.exploratory_identity``: the unit names hold the pool, the paraphrase and the chunk number only, so a resumed run is compared with this."""
    return {"model": args.model, "pools": list(pools), "grids": list(grids), "paraphrases": list(paraphrases), "limit_items": args.limit_items, "pilot_rep": args.pilot_rep}


def check_resume_identity(args: argparse.Namespace, shared: dict[str, Any], own: dict[str, Any]) -> None:
    """Before any finished unit of the run being resumed is reused: the mode, layout and conditions (``rd.check_resume_identity``) and the identity of this entry must be those of the first call."""
    rd.check_resume_identity(args, shared)
    resume = getattr(args, "resume", None)
    if not resume:
        return
    path = Path(resume) / common.MANIFEST_FILE
    if not path.is_file():
        return  # open_run says so itself
    manifest = common.read_json(path)
    recorded = manifest.get("decide_run") if isinstance(manifest, dict) else None
    if not isinstance(recorded, dict):
        raise common.EntryError(f"--resume {resume}: the manifest of that run does not record decide_run (model, pools, grids, paraphrases, pilot subset): it is not a run of this entry")
    now = json.loads(json.dumps(own))
    differences = [f"{key}: {recorded.get(key)!r} -> {now[key]!r}" for key in now if recorded.get(key) != now[key]]
    if differences:
        raise common.EntryError(f"--resume {resume}: the run must be the one it was started as (a finished unit of another model, pool, grid or paraphrase would be reused as if it were this run's): " + "; ".join(differences))


def confirm_record(run: common.RunHandle, key: str, record: Mapping[str, Any], must_match: Sequence[str]) -> None:
    """Write ``record`` into the manifest under ``key``; a resumed run finds the record of its first call there and ``must_match`` (dotted paths) must be equal, else ``EntryError``."""
    manifest = run.record.manifest
    now = common.json_safe(dict(record))
    recorded = manifest.get(key)
    if recorded is None:
        manifest[key] = now
        run.record.write_manifest()
        return

    def dig(doc: Any, dotted: str) -> Any:
        for part in dotted.split("."):
            doc = doc.get(part) if isinstance(doc, dict) else None
        return doc

    differences = [f"{key}.{dotted}: {dig(recorded, dotted)!r} -> {dig(now, dotted)!r}" for dotted in must_match if dig(recorded, dotted) != dig(now, dotted)]
    if differences:
        raise common.EntryError(f"--resume {run.run_dir}: what the run reads with differs from the first session: " + "; ".join(differences))


# ---------------------------------------------------------------------------------------
# the table
# ---------------------------------------------------------------------------------------

def reading_columns(replies: Sequence[Generated], codec: PromptCodec) -> dict[str, pa.Array]:
    """The columns of the replies of one unit. The reading depends on the ids only, so a reply that occurs again is not decoded again."""
    cache: dict[tuple[int, ...], decision_only.Reading] = {}
    readings: list[decision_only.Reading] = []
    for reply in replies:
        reading = cache.get(reply.token_ids)
        if reading is None:
            reading = cache[reply.token_ids] = codec.read(reply.token_ids)
        readings.append(reading)
    return {
        "category": pa.array([r.category for r in readings], type=pa.string()),
        "first_char": pa.array([r.first_char for r in readings], type=pa.string()),
        "ask": pa.array([r.ask for r in readings], type=pa.bool_()),
        "self_choice": pa.array([r.self_choice for r in readings], type=pa.int64()),
        "generated_token_ids": pa.array([list(g.token_ids) for g in replies], type=pa.list_(pa.int64())),
        "generated_text": pa.array([r.decoded for r in readings], type=pa.string()),
        "n_generated": pa.array([len(g.token_ids) for g in replies], type=pa.int64()),
        "finish_reason": pa.array([g.finish_reason for g in replies], type=pa.string()),
        "first_special_position": pa.array([r.first_special_position for r in readings], type=pa.int64()),
        "non_special_after_special": pa.array([r.non_special_after_special for r in readings], type=pa.bool_()),
        "read_text": pa.array([r.text for r in readings], type=pa.string()),
    }


def unit_table(meta: Mapping[str, Any], reading: Mapping[str, pa.Array]) -> pa.Table:
    columns: dict[str, pa.Array] = {name: common.to_arrow(values) for name, values in meta.items()}
    columns.update(reading)
    names = [c for c in FIRST_COLUMNS if c in columns] + [c for c in columns if c not in FIRST_COLUMNS]
    return pa.table({name: columns[name] for name in names})


def run_jobs(run: common.RunHandle, backend: GreedyBackend, jobs: Iterable[common.Job], *, label: str = "decide ") -> dict[str, Any]:
    """``common.run_jobs`` for this arm: the units that are not on disk yet, the source guard before every dispatch, one atomic part per unit."""
    t0 = time.time()
    done = skipped = rows = 0
    for job in jobs:
        if run.parts.done(job.unit_id):
            skipped += 1
            continue
        run.check_source()
        started = time.time()
        texts, meta = job.build()
        replies = backend.generate(texts)
        if len(replies) != len(texts):
            raise common.EntryError(f"{job.unit_id}: the back end returned {len(replies)} replies for {len(texts)} prompts")
        run.parts.write(job.unit_id, unit_table(meta, reading_columns(replies, backend.codec)))
        done += 1
        rows += len(texts)
        common.log(f"{label}{job.unit_id}: {len(texts)} prompts in {time.time() - started:.1f} s")
    return {"units_done": done, "units_skipped": skipped, "rows": rows, "wall_s": time.time() - t0}


def verify_table(table: pa.Table, expected_rows: int) -> None:
    """Every trial exactly once: the units hold the rows the run should hold, and no (item, condition) occurs twice. Raised before the table or the summary is written."""
    if table.num_rows != expected_rows:
        raise common.EntryError(f"the units hold {table.num_rows} trials but the run should hold {expected_rows}: a unit of another configuration was reused, or a unit is missing")
    keys = list(zip(*[table.column(name).to_pylist() for name in TRIAL_KEY], strict=True))
    if len(set(keys)) != len(keys):
        raise common.EntryError("a trial occurs twice in the table (one generation per item and condition)")


def protocol_run_checks(
    table: pa.Table, pool_keys: Sequence[str], rule: Any, *, pools: Sequence[str], grids: Sequence[str], paraphrases: Sequence[str], layout: str, limit_items: int | None, pilot_rep: int | None,
) -> dict[str, Any]:
    """Is this the full specified development run, as opposed to a diagnostic or a smoke run? The arguments must be the specified ones (the pool, the grid psychometric, the paraphrase a00, the
    standard layout, no pilot subset), and the table itself, not the arguments, must hold exactly the trials of the whole pool (``pool_keys``: the fixed key list of the pool, of the specified size)
    under the 54 conditions: every (item key, pool, paraphrase, layout, framing, a_H, c0, displayed w, kappa, D_hat, grid) of the specified run and no other. Computed from the table, before
    the summary is written."""
    def row(key, pool, para, lay, framing, a_H, c0, w, kappa, d_hat, grid):
        rounded = [None if x is None else round(x, 9) for x in (a_H, c0, w)]  # (a null would be a table that is not the specified one; it must not stop the run here)
        return (key, pool, para, lay, framing, *rounded, kappa, d_hat, grid)

    got = {row(*values) for values in zip(*[table.column(name).to_pylist() for name in (*TRIAL_KEY, "grid")], strict=True)}
    expected = {
        row(key, decision_only.PROTOCOL_POOL, decision_only.PROTOCOL_PARAPHRASES[0], decision_only.PROTOCOL_LAYOUT, decision_only.PROTOCOL_FRAMING, a_H, c0, float(quantize_w(w, rule)), None, None,
            decision_only.PROTOCOL_GRIDS[0])
        for key in pool_keys for a_H, c0, w in decision_only.protocol_conditions()
    }
    checks = {
        "pools_argument": list(pools) == [decision_only.PROTOCOL_POOL],
        "grids_argument": list(grids) == list(decision_only.PROTOCOL_GRIDS),
        "paraphrases_argument": list(paraphrases) == list(decision_only.PROTOCOL_PARAPHRASES),
        "layout_argument": layout == decision_only.PROTOCOL_LAYOUT,
        "no_pilot_subset": limit_items is None and pilot_rep is None,
        "pool_has_the_size_of_the_protocol": len(pool_keys) == decision_only.PROTOCOL_POOL_ITEMS,
        "table_holds_the_whole_pool_under_the_54_conditions_and_nothing_else": table.num_rows == len(got) and got == expected,
    }
    counts = {
        "n_rows": table.num_rows, "n_distinct_trials": len(got), "n_expected_trials": len(expected), "n_pool_items": len(pool_keys),
        "n_items_in_table": len({r[0] for r in got}), "n_conditions_in_table": len({r[4:] for r in got}),
    }
    return {"protocol_run": all(checks.values()), "checks": checks, "counts": counts}


def _rates(n: int, counts: Mapping[str, int]) -> dict[str, Any]:
    valid = counts.get(decision_only.ANSWER, 0) + counts.get(decision_only.ASK, 0)
    asks = counts.get(decision_only.ASK, 0)
    return {
        "n": n, "n_valid": valid, "valid_fraction": valid / n if n else None,
        "ask_rate_of_all_trials": asks / n if n else None, "ask_rate_of_valid_trials": asks / valid if valid else None,
    }


def summarize(table: pa.Table) -> dict[str, Any]:
    """A diagnostic of the run, not the verdict on the specified run: the categories, the valid share (a format failure stays in the denominator), the ask rate on both denominators, how the
    replies ended, how many replies had a special token and in how many a token that is not special came after it (reported per model; it says only that the
    model went on generating after a special token, not where the decision comes from), and the valid share of every condition with the flag of the 99% rule."""
    n = table.num_rows
    category = table.column("category").to_pylist()
    first_special = table.column("first_special_position").to_pylist()
    counts = Counter(category)
    pool, para, framing = table.column("pool").to_pylist(), table.column("paraphrase").to_pylist(), table.column("framing").to_pylist()
    by_group: dict[tuple[str, str, str], Counter] = {}
    for p, q, f, c in zip(pool, para, framing, category, strict=True):
        by_group.setdefault((p, q, f), Counter())[c] += 1
    key_columns = [table.column(name).to_pylist() for name in CONDITION_KEY]
    by_condition: dict[tuple[Any, ...], Counter] = {}
    for values, c in zip(zip(*key_columns, strict=True), category, strict=True):
        by_condition.setdefault(values, Counter())[c] += 1
    conditions = []
    for values, c in by_condition.items():
        record = dict(zip(CONDITION_KEY, values, strict=True))
        record.update(_rates(sum(c.values()), c))
        conditions.append(record)
    below = [c for c in conditions if c["valid_fraction"] is not None and c["valid_fraction"] < decision_only.MIN_VALID_FRACTION]
    return {
        "n_rows": n,
        "category_counts": {name: counts.get(name, 0) for name in decision_only.CATEGORIES},
        **_rates(n, counts),
        "finish_reasons": dict(Counter("none" if r is None else r for r in table.column("finish_reason").to_pylist())),
        "n_generated": {str(k): v for k, v in sorted(Counter(table.column("n_generated").to_pylist()).items())},
        "first_special_positions": {("none" if k is None else str(k)): v for k, v in sorted(Counter(first_special).items(), key=lambda kv: (kv[0] is None, kv[0] or 0))},
        "n_with_special_token": sum(1 for k in first_special if k is not None),
        "n_non_special_after_special": sum(1 for flag in table.column("non_special_after_special").to_pylist() if flag),
        "definitions": {
            "n_with_special_token": "trials whose generated sequence holds a special token",
            "n_non_special_after_special": "trials in which a token that is not special comes after the first special token (nothing else is claimed: not where the decision comes from)",
        },
        "groups": {f"{p}/{q}/{f}": _rates(sum(c.values()), c) for (p, q, f), c in sorted(by_group.items())},
        "conditions": conditions,
        "min_valid_fraction": decision_only.MIN_VALID_FRACTION,
        "conditions_below_min_valid_fraction": below,
    }


# ---------------------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------------------

def execute(ctx: common.Context, args: argparse.Namespace, hooks: LocalDecisionHooks) -> dict[str, Any]:
    cfg = ctx.cfg
    pools = pool_list(args)
    check_llm_kwargs(args)  # exit 2 before anything is read or made: no trust_remote_code, and a JSON object
    entry = common.device_entry(cfg, args.model)
    grids = rd.grid_list(args)
    layout = rd.layout_of(args)
    paraphrases = rd.paraphrases_of(ctx, args)
    binding = protocol_binding(args)
    chat_kwargs = common.chat_kwargs_of(entry)
    assert_development_pools(pools)
    fingerprint = model_files_fingerprint(Path(entry["path"]))
    items_by_pool = {p: common.load_pool_items(ctx, p, limit=args.limit_items, pilot_rep=args.pilot_rep) for p in pools}
    conds_by_pool = {p: rd.grid_conditions(cfg, grids) for p in pools}
    shared = rd.exploratory_identity(layout, conds_by_pool, pools)
    own = decide_identity(args, pools, grids, paraphrases)
    check_resume_identity(args, shared, own)  # exit 2 BEFORE any finished unit of the run being resumed is reused
    check_resume_model_files(args, fingerprint)  # the same for the model files: the template, the tokenizer files, the configuration, the weights are those of the first call
    keys = [ctx.ns.development_pilot_key(args.pilot_rep)] if args.limit_items else []
    extra_md5 = {"decision_arm_protocol": binding["md5"]}
    parents = common.parent_run_ids([("screening run", args.screen_run)]) if args.screen_run else []  # exit 2 (before a run directory is made) when the screening run has no manifest with a run id
    if args.screen_run:
        extra_md5["input.screening"] = common.file_md5(Path(args.screen_run) / common.SCREENING)
    unit = {"conditions_per_unit": args.conditions_per_unit, **decision_only.sampling_kwargs()}
    plan = common.numeric_plan(common.readout_backend_plan(args, "vllm_offline", unit=unit))
    extra_manifest: dict[str, Any] = {
        "exploratory": True, "development_reason": DEVELOPMENT_REASON, "exploratory_run": shared,
        "decide_run": {**own, "protocol": binding, "constants": decision_only.protocol_constants()},
        "decide_model_files": fingerprint,
    }
    expected_rows = sum(len(items_by_pool[p]) * len(conds_by_pool[p]) * len(paraphrases) for p in pools)
    opener = hooks.opener or open_engine
    with common.open_run(ctx, pools=pools, seed_keys=keys, parents=parents, extra_md5=extra_md5, numeric=plan, extra_manifest=extra_manifest) as run:
        bh = opener(ctx, args.model)
        try:
            run.confirm_numeric("vllm_offline", bh.numeric_actual())
            report = bh.backend.report
            confirm_record(run, "decide_tokenization", bh.backend.codec.description, ["special_token_ids", "all_special_ids_attribute", "encoding", "decoding"])
            confirm_record(run, "decide_engine", report, ["sampling_params", "stop.effective_stop_token_ids"])
            require_known_stop_set(report)  # before the first dispatch: an unread stop set ends the run as an error, with the report in the manifest
            if hooks.model_files and not run.resumed:
                common.write_model_files(run, entry, bh)
            jobs, unit_ids = [], []
            for pool in pools:
                for para in paraphrases:
                    for gi, rows in enumerate(rd.chunks(len(conds_by_pool[pool]), args.conditions_per_unit)):
                        unit_id = f"{pool}__{para}__g{gi:02d}"
                        unit_ids.append(unit_id)
                        jobs.append(common.Job(unit_id, rd.condition_job(ctx, bh, items_by_pool[pool], pool, para, [conds_by_pool[pool][i] for i in rows], chat_kwargs, args.model, layout)))
            stats = run_jobs(run, bh.backend, jobs)
        finally:
            bh.close()
        # the completion record: only now that every unit is on disk (concat reads them all) and the table is verified
        table = run.parts.concat(unit_ids)
        verify_table(table, expected_rows)
        table = table.select([c for c in FIRST_COLUMNS if c in table.column_names] + [c for c in table.column_names if c not in FIRST_COLUMNS])
        verdict = protocol_run_checks(
            table, ctx.registry().keys(decision_only.PROTOCOL_POOL), ctx.book.rule, pools=pools, grids=grids, paraphrases=paraphrases, layout=layout, limit_items=args.limit_items, pilot_rep=args.pilot_rep,
        )  # from the table itself, before the summary is written
        common.write_parquet(run.run_dir / TABLE_FILE, table)
        summary = summarize(table)
        stop = report["stop"]
        summary.update({
            "protocol_run": verdict["protocol_run"], "protocol_run_checks": verdict["checks"], "protocol_run_counts": verdict["counts"],
            "model": args.model, "stage": STAGE, "backend": "vllm_offline", "pools": pools, "grids": grids, "paraphrases": paraphrases, "layout": layout,
            "stop": {key: stop.get(key) for key in ("tokenizer_eos_token_id", "generation_config_eos_token_id", "effective_stop_token_ids", "stop_set_known", "effective_stop_ids_outside_special_set")},
            "exploratory": True, "development_reason": DEVELOPMENT_REASON, "protocol": binding, "constants": decision_only.protocol_constants(), "jobs": stats,
            "conditions_per_pool": rd.conditions_record(conds_by_pool, pools),
        })
        if args.limit_items:
            summary["pilot"] = {"limit_items": args.limit_items, "pilot_rep": args.pilot_rep}
        common.write_json(run.run_dir / SUMMARY_FILE, summary)
        return {
            "run_dir": str(run.run_dir), "model": args.model, "n_rows": int(table.num_rows), "protocol_run": summary["protocol_run"], "valid_fraction": summary["valid_fraction"],
            "ask_rate_of_valid_trials": summary["ask_rate_of_valid_trials"],
        }


def dry_run(ctx: common.Context, args: argparse.Namespace) -> int:
    cfg = ctx.cfg
    dry = common.DryRun(ctx)
    try:
        pools = pool_list(args)
        check_llm_kwargs(args)
        grids = rd.grid_list(args)
        layout = rd.layout_of(args)
        paraphrases = rd.paraphrases_of(ctx, args)
        binding = protocol_binding(args)
    except common.EntryError as error:
        dry.check("arguments and inputs", False, str(error))
        return dry.emit()
    keys = [ctx.ns.development_pilot_key(args.pilot_rep)] if args.limit_items and args.pilot_rep else []
    common.dry_run_basics(dry, pools=pools, models=[("device", args.model)], seed_keys=keys)
    dry.check("decision-only protocol file", True, binding)
    if args.screen_run:
        try:
            parents = common.parent_run_ids([("screening run", args.screen_run)])
            dry.check("parent runs: the input runs have manifests with a run id", True, parents)
            dry.note("parent_run_ids", parents)
        except common.EntryError as error:
            dry.check("parent runs: the input runs have manifests with a run id", False, str(error))
    sizes = common.pool_sizes(ctx, pools, args.limit_items)
    per_pool = {p: rd.grid_conditions(cfg, grids) for p in pools}
    prompts = sum(sizes[p] * len(per_pool[p]) * len(paraphrases) for p in pools)
    dry.note("workload", {
        "pools": sizes, "grids": grids, "paraphrases": paraphrases, "conditions": {p: len(per_pool[p]) for p in pools}, "decision_prompts": prompts,
        "units": sum(len(rd.chunks(len(per_pool[p]), args.conditions_per_unit)) * len(paraphrases) for p in pools), "backend": "vllm_offline",
        "note": "one generation per item and condition; a condition that two grids share is run once",
    })
    dry.note("generation", {**decision_only.sampling_kwargs(), "constants": decision_only.protocol_constants()})
    shared = rd.exploratory_identity(layout, per_pool, pools)
    dry.note("exploratory", {"exploratory": True, "layout": layout, "development_reason": DEVELOPMENT_REASON, "pools": pools, "grids": grids, "conditions": {p: len(per_pool[p]) for p in pools}})
    if getattr(args, "resume", None):
        try:
            check_resume_identity(args, shared, decide_identity(args, pools, grids, paraphrases))
            dry.check("resume: the run being resumed has this mode, layout, conditions and identity", True, {"resume": args.resume})
        except common.EntryError as error:
            dry.check("resume: the run being resumed has this mode, layout, conditions and identity", False, str(error))
        try:
            check_resume_model_files(args, model_files_fingerprint(Path(common.device_entry(cfg, args.model)["path"])))
            dry.check("resume: the files of the model directory are those of the first session", True, {"resume": args.resume})
        except common.EntryError as error:
            dry.check("resume: the files of the model directory are those of the first session", False, str(error))
    return dry.emit()


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="The decision-only arm of one local model: the P-ask prompt of run_decisions, a greedy reply of at most four tokens, the first character read (development pool only)."
    )
    common.add_common_arguments(p)
    group = p.add_argument_group("engine (offline vLLM only)")
    group.add_argument("--model", required=True, help="end-side model id of the configuration (models.devices[].id)")
    group.add_argument("--backend", choices=("vllm_offline",), default="vllm_offline", help="the arm generates, so it needs the offline vLLM engine (the other back ends of run_decisions read digit probabilities)")
    group.add_argument("--gpu-mem-util", type=float, default=None, help="vLLM gpu_memory_utilization (offline); set it from the free memory measured before the run")
    group.add_argument("--max-model-len", type=int, default=4096)
    group.add_argument("--llm-kwargs", default=None, help="JSON object merged into the keyword arguments of vllm.LLM (offline)")
    group.add_argument("--limit-items", type=int, default=None, help="pilot: a seeded subset of this many items per pool")
    group.add_argument("--pilot-rep", type=int, default=None, help="registered development pilot seed (9001-9006) of the subset")
    p.add_argument("--screen-run", default=None, help="prompt-screening run directory: its retained paraphrases are the ones run")
    p.add_argument("--paraphrases", default=None, help="comma list of paraphrase ids (overrides --screen-run)")
    p.add_argument("--pools", default=None, help="comma list of pools: only mmlu_redux_cal, the development pool (required); a holdout pool is refused")
    p.add_argument("--grids", default=None, help="comma list of psychometric, f1, screening (required); the operating grid is not available")
    p.add_argument("--conditions-per-unit", type=int, default=6)
    p.add_argument("--exploratory", action="store_true", help="required: the door of run_decisions for development runs; the manifest and the summary say exploratory: true; --resume must match the mode, the layout and the conditions")
    p.add_argument("--layout", default=None, help="standard (default) or block_position, the variant layout of the prompt file (the conditions block before the question)")
    p.add_argument("--expect-protocol-md5", default=None, help="stop unless the protocol file (nmi.decision_only.PROTOCOL_FILE) has exactly this md5 (the frozen value)")
    return p


def main(argv: Sequence[str] | None = None, hooks: LocalDecisionHooks | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    hooks = hooks or LocalDecisionHooks()
    args = make_parser().parse_args(argv)
    t0 = time.time()
    try:
        ctx = common.build_context(args, script=SCRIPT, stage=STAGE, argv=argv)
        if args.dry_run:
            return dry_run(ctx, args)
        outcome = execute(ctx, args, hooks)
    except (common.EntryError, SplitError, paths.PathPolicyError, LabelExistsError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except common.SourceChangedError as error:
        return common.stopped_by_source_change(error)
    outcome["wall_s"] = time.time() - t0
    print(json.dumps(common.json_safe(outcome), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
