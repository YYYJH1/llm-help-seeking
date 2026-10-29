"""Shared layer of the confirmatory entries (held-out pool, pre-registered protocol).

Shared by ``scripts/run_c1_confirm.py``, ``run_mechanism_confirm.py`` and ``run_scenario_confirm.py``. The holdout rules of ``nmi.data.holdout`` are unchanged (a marker created
by hand, a permanent consumption ledger, a key consumed by a batch is development data afterwards). This module adds, in this order, the steps that the ledger does not do for an entry:

1. Batch manifest (``load_batch_manifest``): one JSON file per batch with the batch id, the pool, the models, the arms, the md5 of every bound entry, configuration, prompt and code file, the md5 of
   the frozen protocol and, optionally, the numeric settings of each model. Its md5 is the ``description`` of the batch in the ledger, and the md5 of the frozen protocol is its ``protocol_md5``.
2. Batch identity (``check_batch_identity``), before the pool is loaded: the batch id belongs to the pool, ``protocol_md5`` is the md5 of the frozen protocol as it is now, ``description`` is the md5
   of the batch manifest, and the model and the arm are in the manifest. Any difference raises ``ConfirmError`` (the entry exits 2) and nothing has been consumed.
3. Preflight (``run_preflight``) and the load that cannot skip it (``load_confirm_items``): label, model directory, batch identity, md5 of the bound files, numeric settings, the unique run of the
   (batch, model, arm) and, for ``--resume``, the identity of the run that is continued. The pool is loaded only with a passing report for the same (batch, model, arm).
4. Access (``open_confirm_access``): the ledger records the models and arms of the FIRST access that consumes the keys of a pool, so the access is built from the batch manifest and carries the
   full lists whichever model the run is. The scenario batch needs a ``TriviaSplit`` (the plain ``SplitRegistry`` has no key list of the TriviaQA pools).
5. Unique run and attempts (``bind_run``, ``find_binding``, ``record_attempt``, ``attempts``, ``state_of_records``) in ``<results_root>/confirm_runs.jsonl``: append-only, every change made under an
   exclusive lock on ``confirm_runs.jsonl.lock`` (a read takes no lock). A (batch, model, arm) is bound to one run directory for good; a unit has at most two attempts (the first and the one after a
   resume); the first attempt that completes is the result of the unit, and a complete unit is never run again. The records are ``{"kind": "bind", batch_id, model, arm, run_dir, utc, epoch, pid}``
   and ``{"kind": "attempt", batch_id, model, arm, run_dir, unit_id, attempt, outcome, utc, epoch, pid}`` (outcome ``"started"``, then ``"complete"``, ``"call_failure"`` or ``"missing_rows"``).
6. Manifest block of a confirmatory run (``confirm_manifest_block``): no exploratory marker; the ``prereg`` key of ``input_md5`` binds the frozen protocol, and the batch manifest is bound too.

All steps use ``canonical_results_root(...)`` (absolute, symbolic links resolved, anchored at the repository when relative, never a default; an entry names the root with ``--results-root``), so the
marker, the ledger and ``confirm_runs.jsonl`` are the same files for every step. Rehearsal (``--rehearsal`` of every entry): the same code on the development pool of the bank (``REHEARSAL_POOLS``),
marked exploratory, with no ledger, no ``confirm_runs.jsonl`` entry and no batch manifest. No item key, question or answer is logged, printed or put into an error message: messages carry counts,
file names and md5 values.

Sequence of an entry (the pool is read once, last; ``--dry-run`` stops after the report and loads nothing)::

    batch = load_batch_manifest(args.batch_manifest)
    report = run_preflight(root, batch, model=m, arm=a, pool=batch.pool, stage=STAGE, tag=args.tag, model_dir=entry["path"], gpu_mem_util=args.gpu_mem_util,
                           max_model_len=args.max_model_len, llm_kwargs=args.llm_kwargs, resume_dir=args.resume)      # all checks, none loads
    report.require()                                                                                                    # exit 2 here, nothing consumed
    block = confirm_manifest_block(batch, prereg_frozen_path=FROZEN_PREREG_PATH)
    with common.open_run(ctx, **open_run_kwargs(block, extra_md5=..., extra_manifest=...)) as run:                      # _c2_common.open_run
        binding = bind_run(root, batch, model=m, arm=a, run_dir=run.run_dir)                                            # before the pool is read; a run that reads several arms
                                                                                                                        # (C1: answer and ask) binds every one of them here
        items = load_confirm_items(root, batch, registry, model=m, arm=a, loader=..., preflight=report)
        # per unit: record_attempt(root, binding, unit, "started"), then "complete", "call_failure" or "missing_rows"
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
for _p in (REPO_ROOT, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from nmi import paths  # noqa: E402
from nmi.data.holdout import (  # noqa: E402
    HoldoutAccess,
    HoldoutLedger,
    HoldoutLedgerError,
    HoldoutSealedError,
    open_holdout,
)
from nmi.data.registry import (  # noqa: E402
    MMLU_CAL,
    MMLU_HOLDOUT,
    SCENARIO_DEV_POOL,
    SCENARIO_SEALED_POOL,
    SplitRegistry,
)
from nmi.data.triviaqa import TriviaSplit  # noqa: E402
from nmi.provenance import bytes_md5, file_md5, note_skipped  # noqa: E402

# ---------------------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------------------

# The two batches and the pool of each. The table is in the code on purpose: a batch manifest that names a pool other than the one its batch id stands for is refused.
BATCH_MMLU = "v2_mmlu_01"
BATCH_SCENARIO = "v2_s1_01"
BATCH_POOLS: dict[str, str] = {BATCH_MMLU: MMLU_HOLDOUT, BATCH_SCENARIO: SCENARIO_SEALED_POOL}
# The arms of a batch: the C1 read-outs and the mechanism group share the MMLU batch; the scenario batch has the expert and two phases of the end-side models.
ARMS_BY_POOL: dict[str, tuple[str, ...]] = {MMLU_HOLDOUT: ("answer", "ask", "mech"), SCENARIO_SEALED_POOL: ("expert", "answer", "decide")}
# The rehearsal runs the confirmatory code on the development pool of the same bank (names as in nmi.data.registry).
REHEARSAL_POOLS: dict[str, str] = {MMLU_HOLDOUT: MMLU_CAL, SCENARIO_SEALED_POOL: SCENARIO_DEV_POOL}

# The frozen copy of the pre-registered protocol; its md5 is recorded in the batch manifest and in the ledger. It is read at call time, so a test can point it elsewhere.
FROZEN_PREREG_PATH = paths.PROTOCOL_DIR / "preregistration_v2.md"

CONFIRM_RUNS_NAME = "confirm_runs.jsonl"
ATTEMPTS_COPY_NAME = "attempts.jsonl"  # the copy in a run directory; the registry in the results root is the record
MAX_ATTEMPTS = 2  # the first attempt and the one after a resume
OUTCOMES = ("started", "complete", "call_failure", "missing_rows")
MANIFEST_FORMAT = 1

# What an output file or a run label must not say ("*question*", "*holdout*", "*benchmark*"); checked without regard to case.
FORBIDDEN_NAME_PARTS = ("holdout", "question", "benchmark")

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")
_MD5_RE = re.compile(r"^[0-9a-f]{32}$")
_RUN_DIR_RE = re.compile(r"\d{8}_\d{6}_(.+)")
_MANIFEST_KEYS = frozenset({"format", "batch_id", "pool", "models", "arms", "prereg_md5", "files_md5"})
_MANIFEST_OPTIONAL = frozenset({"numeric", "comment"})
_NUMERIC_KEYS = frozenset({"gpu_mem_util", "max_model_len", "llm_kwargs"})


class ConfirmError(RuntimeError):
    """A confirmatory entry must stop: it prints the message and exits 2. Raised before the pool is loaded, it has consumed no item."""


class AttemptLimitError(ConfirmError):
    """A unit asked for a third attempt (at most ``MAX_ATTEMPTS`` are allowed)."""


# ---------------------------------------------------------------------------------------
# the results root
# ---------------------------------------------------------------------------------------

def canonical_results_root(results_root: str | Path | None) -> Path:
    """The one form of the results root that registration, loader, access and resumed run share: anchored at the repository when relative (as ``nmi.paths.results_root``), absolute,
    symbolic links resolved, and not under a directory that ``nmi.paths`` refuses for writing. ``None`` is refused: a confirmatory step never falls back to the default ``results`` of the checkout."""
    if results_root is None:
        raise ConfirmError("the results root must be given explicitly (--results-root); the confirmatory steps do not fall back to a default")
    try:
        return paths.assert_off_root_partition(paths.results_root(results_root))
    except paths.PathPolicyError as error:
        raise ConfirmError(str(error)) from error


def _ledger(root: Path) -> HoldoutLedger:
    return HoldoutLedger(paths.holdout_ledger_path(root))


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------------------
# the batch manifest
# ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class BatchManifest:
    """A loaded batch manifest. ``md5`` is the md5 of the file (the ``description`` of the batch in the ledger), ``path`` the absolute path it was read from, ``prereg_md5`` the md5 of the
    frozen pre-registration that it names, ``files_md5`` the md5 of every bound file by path relative to the repository.

    ``numeric`` is optional and empty by default: model -> the numeric settings that the batch protocol fixes for it
    (``gpu_mem_util``, ``max_model_len``, ``llm_kwargs``), checked by ``check_numeric_settings`` before the pool is loaded."""

    batch_id: str
    pool: str
    models: tuple[str, ...]
    arms: tuple[str, ...]
    files_md5: dict[str, str]
    prereg_md5: str
    path: str
    md5: str
    numeric: dict[str, dict[str, Any]] = field(default_factory=dict)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate key {key!r}")
        out[key] = value
    return out


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _text_list(where: str, name: str, value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ConfirmError(f"{where}: {name} must be a non-empty list")
    if not all(isinstance(v, str) and _ID_RE.fullmatch(v) for v in value):
        raise ConfirmError(f"{where}: every entry of {name} must be a plain name ([A-Za-z0-9][A-Za-z0-9_.-]*)")
    if len(set(value)) != len(value):
        raise ConfirmError(f"{where}: {name} lists a name twice")
    return tuple(value)


def _validated_numeric(where: str, value: Any, models: Sequence[str]) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict) or set(value) != set(models):
        raise ConfirmError(f"{where}: numeric must give an entry for every model of the batch, and for no other")
    out: dict[str, dict[str, Any]] = {}
    for model, entry in value.items():
        label = f"{where}: numeric[{model!r}]"
        if not isinstance(entry, dict) or "gpu_mem_util" not in entry or set(entry) - _NUMERIC_KEYS:
            raise ConfirmError(f"{label} must be a mapping with gpu_mem_util and, optionally, max_model_len and llm_kwargs")
        util = entry["gpu_mem_util"]
        if not _is_number(util) or not 0.0 < float(util) <= 1.0:
            raise ConfirmError(f"{label}: gpu_mem_util must be a number in (0, 1]")
        if "max_model_len" in entry and (isinstance(entry["max_model_len"], bool) or not isinstance(entry["max_model_len"], int) or entry["max_model_len"] < 1):
            raise ConfirmError(f"{label}: max_model_len must be a positive integer")
        if "llm_kwargs" in entry and not isinstance(entry["llm_kwargs"], dict):
            raise ConfirmError(f"{label}: llm_kwargs must be a mapping")
        out[model] = dict(entry)
    return out


def load_batch_manifest(path: str | Path) -> BatchManifest:
    """Read and validate a batch manifest. A repeated key, an unknown key, an unknown batch id, a pool other than the one of the id, arms other than those of the pool, a
    malformed md5 or a bound path that leaves the repository raises ``ConfirmError``. The file is one JSON object::

        {"format": 1, "batch_id": "v2_mmlu_01", "pool": "mmlu_redux_holdout", "models": ["qwen3-1.7b", ...], "arms": ["answer", "ask", "mech"],
         "prereg_md5": "<md5 of the frozen protocol file>", "files_md5": {"<path relative to the repository>": "<md5>", ...},
         "numeric": {"<model>": {"gpu_mem_util": 0.8, "max_model_len": 4096, "llm_kwargs": {"quantization": "fp8"}}, ...},      (optional: an entry for every model, or none)
         "comment": "..."}                                                                                                     (optional)
    """
    manifest_path = Path(path)
    try:
        raw = manifest_path.read_bytes()
    except OSError as error:
        raise ConfirmError(f"batch manifest {manifest_path}: cannot be read ({type(error).__name__})") from error
    where = f"batch manifest {manifest_path.name}"
    try:
        doc = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, ValueError) as error:
        raise ConfirmError(f"{where}: not valid JSON ({error})") from error
    if not isinstance(doc, dict):
        raise ConfirmError(f"{where}: the top level must be a mapping")
    missing, unknown = sorted(_MANIFEST_KEYS - set(doc)), sorted(set(doc) - _MANIFEST_KEYS - _MANIFEST_OPTIONAL)
    if missing or unknown:
        raise ConfirmError(f"{where}: missing keys {missing}, unknown keys {unknown}")
    if doc["format"] != MANIFEST_FORMAT:
        raise ConfirmError(f"{where}: unsupported format {doc['format']!r}")
    batch_id, pool = doc["batch_id"], doc["pool"]
    if batch_id not in BATCH_POOLS:
        raise ConfirmError(f"{where}: batch id {batch_id!r} is not a known batch {sorted(BATCH_POOLS)}")
    if pool != BATCH_POOLS[batch_id]:
        raise ConfirmError(f"{where}: batch {batch_id!r} reads the pool {BATCH_POOLS[batch_id]!r}, the manifest says {pool!r}")
    models = _text_list(where, "models", doc["models"])
    arms = _text_list(where, "arms", doc["arms"])
    if set(arms) != set(ARMS_BY_POOL[pool]):
        raise ConfirmError(f"{where}: the arms of {pool!r} are {list(ARMS_BY_POOL[pool])}, the manifest lists {list(arms)}: the first access must carry them all")
    if not isinstance(doc["prereg_md5"], str) or not _MD5_RE.fullmatch(doc["prereg_md5"]):
        raise ConfirmError(f"{where}: prereg_md5 must be 32 lower-case hex digits")
    files = doc["files_md5"]
    if not isinstance(files, dict) or not files:
        raise ConfirmError(f"{where}: files_md5 must be a non-empty mapping of repository paths to md5 values")
    for rel, digest in files.items():
        if not isinstance(rel, str) or not rel or Path(rel).is_absolute() or ".." in Path(rel).parts or "\\" in rel:
            raise ConfirmError(f"{where}: files_md5 key {rel!r} must be a relative path inside the repository")
        if not isinstance(digest, str) or not _MD5_RE.fullmatch(digest):
            raise ConfirmError(f"{where}: files_md5[{rel!r}] must be 32 lower-case hex digits")
    numeric = _validated_numeric(where, doc["numeric"], models) if "numeric" in doc else {}
    return BatchManifest(
        batch_id=batch_id, pool=pool, models=models, arms=arms, files_md5=dict(files), prereg_md5=doc["prereg_md5"], path=str(manifest_path.resolve()), md5=bytes_md5(raw), numeric=numeric,
    )


# ---------------------------------------------------------------------------------------
# the check of the batch identity
# ---------------------------------------------------------------------------------------

def _frozen_prereg_md5(named_by_batch: str) -> str:
    """The md5 of the frozen pre-registration as it is now. The document is not part of this release: when it is absent, one line is printed, the file check is skipped and the md5 that the
    batch manifest names is taken as the reference (the ledger is still compared with it)."""
    path = Path(FROZEN_PREREG_PATH)
    if not path.is_file():
        note_skipped(path, "frozen pre-registration")
        return named_by_batch
    return file_md5(path)


def _require_member(batch: BatchManifest, model: str, arm: str) -> None:
    if model not in batch.models:
        raise ConfirmError(f"model {model!r} is not in the batch manifest of {batch.batch_id!r} (models: {list(batch.models)})")
    if arm not in batch.arms:
        raise ConfirmError(f"arm {arm!r} is not in the batch manifest of {batch.batch_id!r} (arms: {list(batch.arms)})")


def _check_registration(root: Path, batch: BatchManifest) -> dict[str, Any]:
    """The first three items of the check of the batch identity (pool, ``protocol_md5``, ``description``); returns the ledger record of the batch."""
    expected_pool = BATCH_POOLS.get(batch.batch_id)
    if expected_pool is None or batch.pool != expected_pool:
        raise ConfirmError(f"pool: batch {batch.batch_id!r} reads the pool {expected_pool!r}, this manifest says {batch.pool!r}")
    ledger = paths.holdout_ledger_path(root)
    try:
        record = _ledger(root).batches().get(batch.batch_id)
    except HoldoutLedgerError as error:
        raise ConfirmError(f"the consumption ledger {ledger} cannot be read: {error}") from error
    if record is None:
        raise ConfirmError(f"batch {batch.batch_id!r} is not registered in {ledger}: batches are registered before the marker is created")
    frozen = _frozen_prereg_md5(batch.prereg_md5)
    if record.get("protocol_md5") != frozen:
        raise ConfirmError(
            f"protocol_md5: the ledger holds {str(record.get('protocol_md5'))[:12]} for batch {batch.batch_id!r}, the frozen pre-registration {FROZEN_PREREG_PATH} has {frozen[:12]}"
        )
    if batch.prereg_md5 != frozen:
        raise ConfirmError(f"protocol_md5: the batch manifest names the pre-registration {batch.prereg_md5[:12]}, the frozen file has {frozen[:12]}")
    if record.get("description") != batch.md5:
        raise ConfirmError(f"description: the ledger holds {str(record.get('description'))[:12]} for batch {batch.batch_id!r}, the batch manifest has md5 {batch.md5[:12]}")
    on_disk = file_md5(batch.path) if Path(batch.path).is_file() else None
    if on_disk != batch.md5:
        raise ConfirmError(f"description: the batch manifest {batch.path} is not the file that was loaded (md5 {str(on_disk)[:12]}, loaded {batch.md5[:12]})")
    return record


def check_batch_identity(results_root: str | Path, batch: BatchManifest, *, model: str, arm: str) -> None:
    """The check of the batch identity, before the pool is loaded; any difference raises ``ConfirmError`` and nothing is consumed:

    * the batch id belongs to the pool of the manifest (``BATCH_POOLS``) and is registered in the ledger of ``results_root``;
    * ``protocol_md5`` of the registered batch is the md5 of the frozen pre-registration as it is now (and the one the manifest names);
    * ``description`` of the registered batch is the md5 of the batch manifest (and of the file it was loaded from);
    * ``model`` and ``arm`` are in the manifest."""
    root = canonical_results_root(results_root)
    _check_registration(root, batch)
    _require_member(batch, model, arm)


# ---------------------------------------------------------------------------------------
# the access
# ---------------------------------------------------------------------------------------

def _require_registry_for(batch: BatchManifest, registry: Any) -> None:
    if batch.pool == SCENARIO_SEALED_POOL:
        if not isinstance(registry, TriviaSplit):
            raise ConfirmError(f"the scenario batch {batch.batch_id!r} needs a TriviaSplit (the plain SplitRegistry has no key list of the TriviaQA pools), got {type(registry).__name__}")
    elif not isinstance(registry, SplitRegistry):
        raise ConfirmError(f"the batch {batch.batch_id!r} reads {batch.pool!r} and needs a SplitRegistry, got {type(registry).__name__}")


def open_confirm_access(results_root: str | Path, batch: BatchManifest, registry: SplitRegistry | TriviaSplit) -> HoldoutAccess:
    """The ``HoldoutAccess`` of the batch. It carries ALL the models and arms of the manifest, because the ledger keeps the access of the first read that consumes a key. It is built on the
    canonical results root (use ``access.results_root`` as the ``results_root`` of the loader) and refuses a batch that fails the registration checks of ``check_batch_identity``. Marker missing
    or batch unregistered raises ``ConfirmError`` (``open_holdout`` raises the ledger's own errors)."""
    root = canonical_results_root(results_root)
    _check_registration(root, batch)
    _require_registry_for(batch, registry)
    try:
        return open_holdout(registry=registry, batch_id=batch.batch_id, models=batch.models, arms=batch.arms, results_root=root)  # type: ignore[arg-type]
    except (HoldoutSealedError, HoldoutLedgerError) as error:
        raise ConfirmError(str(error)) from error


# ---------------------------------------------------------------------------------------
# the manifest block of a confirmatory run
# ---------------------------------------------------------------------------------------

def confirm_manifest_block(batch: BatchManifest, *, prereg_frozen_path: str | Path) -> dict[str, Any]:
    """What a confirmatory run adds to its manifest, as two parts:

    * ``"confirmatory"``: the batch, pool, models, arms, the md5 of the batch manifest, of the frozen pre-registration and of every bound file (``extra_manifest={"confirmatory": ...}`` of
      ``_c2_common.open_run``). There is no ``exploratory`` key anywhere: this is not a development run;
    * ``"input_md5"``: ``prereg`` is the md5 of the FROZEN pre-registration (it replaces the md5 that ``open_run`` binds under that name by default; ``begin_run`` lets ``input_md5``
      override the md5 of an input file), and ``input.confirm_batch_manifest`` is the md5 of the batch manifest (``extra_md5=``; a resumed run must have the same).

    The pre-registration file must be the one the batch manifest names."""
    path = Path(prereg_frozen_path)
    if not path.is_file():  # the document is not part of this release: the md5 the batch manifest names is recorded in its place
        note_skipped(path, "frozen pre-registration")
        prereg = batch.prereg_md5
    else:
        prereg = file_md5(path)
        if prereg != batch.prereg_md5:
            raise ConfirmError(f"the pre-registration {path} has md5 {prereg[:12]}, the batch manifest names {batch.prereg_md5[:12]}")
    return {
        "confirmatory": {
            "batch_id": batch.batch_id,
            "pool": batch.pool,
            "models": list(batch.models),
            "arms": list(batch.arms),
            "batch_manifest": {"path": batch.path, "md5": batch.md5},
            "prereg_frozen": {"path": str(path.resolve()), "md5": prereg},
            "files_md5": dict(batch.files_md5),
        },
        "input_md5": {"prereg": prereg, "input.confirm_batch_manifest": batch.md5},
    }


def open_run_kwargs(block: Mapping[str, Any], *, extra_md5: Mapping[str, str] | None = None, extra_manifest: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The ``extra_md5`` and ``extra_manifest`` arguments of ``_c2_common.open_run`` for a confirmatory run: the block of ``confirm_manifest_block`` merged with what the entry adds itself.
    (The block is not an ``extra_manifest`` as it is: its ``input_md5`` would replace the whole ``input_md5`` of the manifest.)"""
    return {
        "extra_md5": {**block["input_md5"], **(extra_md5 or {})},
        "extra_manifest": {"confirmatory": block["confirmatory"], **(extra_manifest or {})},
    }


def _stable_identity(block: Mapping[str, Any]) -> dict[str, Any]:
    """The part of a ``confirmatory`` block that must be the same in a resumed run (paths are left out: a checkout may have moved)."""
    return {
        "batch_id": block.get("batch_id"), "pool": block.get("pool"), "models": block.get("models"), "arms": block.get("arms"),
        "batch_manifest_md5": (block.get("batch_manifest") or {}).get("md5"), "prereg_frozen_md5": (block.get("prereg_frozen") or {}).get("md5"), "files_md5": block.get("files_md5"),
    }


# ---------------------------------------------------------------------------------------
# the registry of confirmatory runs and attempts
# ---------------------------------------------------------------------------------------

class _Runs:
    """``<results_root>/confirm_runs.jsonl``: append-only; whoever changes it holds an exclusive lock on ``<path>.lock`` while it reads and appends (as ``HoldoutLedger``); a plain read takes none."""

    def __init__(self, root: Path) -> None:
        self.path = root / CONFIRM_RUNS_NAME

    @contextmanager
    def locked(self) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path.with_name(self.path.name + ".lock"), "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def read(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        records: list[dict[str, Any]] = []
        with open(self.path, encoding="utf-8") as fh:
            for number, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ConfirmError(f"{self.path}:{number} is not valid JSON") from error
                if not isinstance(record, dict):
                    raise ConfirmError(f"{self.path}:{number} is not a JSON object")
                records.append(record)
        return records

    def append(self, record: Mapping[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())


def read_confirm_runs(results_root: str | Path) -> list[dict[str, Any]]:
    """Every record of ``confirm_runs.jsonl`` (bindings and attempts), in file order. A read takes no lock and creates nothing (a record is one appended line); the decisions that change the
    registry (``bind_run``, ``record_attempt``) read again under the lock."""
    return _Runs(canonical_results_root(results_root)).read()


@dataclass(frozen=True)
class RunBinding:
    batch_id: str
    model: str
    arm: str
    run_dir: str


def _canonical_run_dir(root: Path, run_dir: str | Path) -> Path:
    directory = Path(run_dir).resolve()
    if directory == root or not directory.is_relative_to(root):
        raise ConfirmError(f"the run directory {directory} is not inside the results root {root}")
    return directory


def _binding_record(records: Sequence[Mapping[str, Any]], batch_id: str, model: str, arm: str) -> Mapping[str, Any] | None:
    for record in records:
        if record.get("kind") == "bind" and (record.get("batch_id"), record.get("model"), record.get("arm")) == (batch_id, model, arm):
            return record
    return None


def find_binding(results_root: str | Path, batch: BatchManifest, *, model: str, arm: str) -> RunBinding | None:
    """The run directory that (batch, model, arm) is bound to, or None."""
    record = _binding_record(read_confirm_runs(results_root), batch.batch_id, model, arm)
    return None if record is None else RunBinding(batch.batch_id, model, arm, str(record["run_dir"]))


def bind_run(results_root: str | Path, batch: BatchManifest, *, model: str, arm: str, run_dir: str | Path) -> RunBinding:
    """Bind (batch, model, arm) to a run directory, once. Binding the same directory again returns the binding (a resumed run); another directory raises ``ConfirmError``: the confirmatory run
    of a (batch, model, arm) is unique, a new tag or a new directory does not start again. Call it as soon as the run directory exists and before the pool is loaded."""
    root = canonical_results_root(results_root)
    _require_member(batch, model, arm)
    directory = _canonical_run_dir(root, run_dir)
    runs = _Runs(root)
    with runs.locked():
        existing = _binding_record(runs.read(), batch.batch_id, model, arm)
        if existing is not None:
            if existing["run_dir"] != str(directory):
                raise ConfirmError(
                    f"({batch.batch_id}, {model}, {arm}) is already bound to the run {existing['run_dir']}; the confirmatory run exists once, "
                    f"a new label or directory ({directory}) does not start it again: continue the bound run with --resume"
                )
        else:
            runs.append({
                "kind": "bind", "batch_id": batch.batch_id, "model": model, "arm": arm, "run_dir": str(directory),
                "utc": _now_iso(), "epoch": time.time(), "pid": os.getpid(),
            })
    return RunBinding(batch.batch_id, model, arm, str(directory))


def _attempt_records(records: Sequence[Mapping[str, Any]], binding: RunBinding, unit_id: str) -> list[dict[str, Any]]:
    return [
        dict(r) for r in records
        if r.get("kind") == "attempt" and (r.get("batch_id"), r.get("model"), r.get("arm"), r.get("unit_id")) == (binding.batch_id, binding.model, binding.arm, unit_id)
    ]


def _check_bound(records: Sequence[Mapping[str, Any]], binding: RunBinding) -> None:
    bound = _binding_record(records, binding.batch_id, binding.model, binding.arm)
    if bound is None or bound["run_dir"] != str(Path(binding.run_dir).resolve()):
        raise ConfirmError(f"({binding.batch_id}, {binding.model}, {binding.arm}) is not bound to the run {binding.run_dir}")


def record_attempt(results_root: str | Path, binding: RunBinding, unit_id: str, outcome: str) -> int:
    """Record the start or the result of an attempt of a unit; returns the number of the attempt (1 or 2).

    * ``"started"`` opens the next attempt. It raises ``AttemptLimitError`` when two attempts have been started already (an attempt that was started and never closed, because the process died, counts)
      and ``ConfirmError`` when the unit is complete: the first complete attempt is the result of the unit and is never run again.
    * ``"complete"``, ``"call_failure"`` (the back end raised, or returned fewer read-outs than prompts) and ``"missing_rows"`` (a finished unit lacks rows of the expected set) close the open
      attempt; with no open attempt they raise ``ConfirmError``.

    The count lives in ``confirm_runs.jsonl`` of the results root, across processes and labels. The record is copied to ``attempts.jsonl`` in the run directory (a copy; the registry decides)."""
    if outcome not in OUTCOMES:
        raise ConfirmError(f"unknown outcome {outcome!r}; known: {list(OUTCOMES)}")
    try:
        paths.validate_name("unit", unit_id)
    except paths.PathPolicyError as error:
        raise ConfirmError(str(error)) from error
    check_output_name(unit_id)  # a unit is a file of the parts directory
    root = canonical_results_root(results_root)
    runs = _Runs(root)
    with runs.locked():
        records = runs.read()
        _check_bound(records, binding)
        unit = _attempt_records(records, binding, unit_id)
        started = [r for r in unit if r["outcome"] == "started"]
        if outcome == "started":
            if any(r["outcome"] == "complete" for r in unit):
                raise ConfirmError(f"unit {unit_id!r} of ({binding.batch_id}, {binding.model}, {binding.arm}) is complete; a complete unit is not run again")
            number = len(started) + 1
            if number > MAX_ATTEMPTS:
                raise AttemptLimitError(
                    f"unit {unit_id!r} of ({binding.batch_id}, {binding.model}, {binding.arm}) has had {len(started)} attempts; at most {MAX_ATTEMPTS} are allowed "
                    "(the unit is missing and the model is not testable on it)"
                )
        else:
            if not unit or unit[-1]["outcome"] != "started":
                raise ConfirmError(f"unit {unit_id!r} of ({binding.batch_id}, {binding.model}, {binding.arm}) has no attempt in progress to close with {outcome!r}")
            number = int(unit[-1]["attempt"])
        record = {
            "kind": "attempt", "batch_id": binding.batch_id, "model": binding.model, "arm": binding.arm, "run_dir": binding.run_dir, "unit_id": unit_id,
            "attempt": number, "outcome": outcome, "utc": _now_iso(), "epoch": time.time(), "pid": os.getpid(),
        }
        runs.append(record)
    copy_dir = Path(binding.run_dir)
    if copy_dir.is_dir():
        try:
            with open(copy_dir / ATTEMPTS_COPY_NAME, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, sort_keys=True) + "\n")
        except OSError:
            pass  # only a copy; the registry in the results root has the record
    return number


def attempts(results_root: str | Path, binding: RunBinding, unit_id: str) -> list[dict[str, Any]]:
    """The attempt records of a unit (starts and outcomes), in the order they were written."""
    return _attempt_records(read_confirm_runs(results_root), binding, unit_id)


def state_of_records(records: Sequence[Mapping[str, Any]]) -> str:
    """The state of a unit from its attempt records, in the order written: ``"new"`` (no attempt), ``"in_progress"`` (the last attempt was started and not closed: a running or a dead process),
    ``"retry"`` (one failed attempt, one more is allowed), ``"exhausted"`` (two attempts failed: the unit is missing) or ``"complete"``."""
    if any(r["outcome"] == "complete" for r in records):
        return "complete"
    n_started = sum(r["outcome"] == "started" for r in records)
    if n_started == 0:
        return "new"
    if records[-1]["outcome"] == "started":
        return "in_progress"
    return "exhausted" if n_started >= MAX_ATTEMPTS else "retry"


# ---------------------------------------------------------------------------------------
# the preflight
# ---------------------------------------------------------------------------------------

def check_output_name(name: str) -> None:
    """A file or directory name that a confirmatory run writes must not contain a forbidden word (``FORBIDDEN_NAME_PARTS``): a name
    like that must not give away what a file holds."""
    lowered = name.lower()
    hits = [part for part in FORBIDDEN_NAME_PARTS if part in lowered]
    if hits:
        raise ConfirmError(f"the name {name!r} contains {hits}, which no output of a confirmatory run may")


def check_label(results_root: str | Path, stage: str, tag: str | None, *, prefix: str | None = None, resume_dir: str | Path | None = None) -> None:
    """The label of a new run: a plain name without a forbidden word, with the required prefix (the scenario uses ``s1seal_``), not used in the stage yet. A resumed run keeps its own label."""
    root = canonical_results_root(results_root)
    try:
        paths.validate_name("stage", stage)
        check_output_name(stage)
        if resume_dir is None:
            if tag is None:
                raise ConfirmError("--tag is required for a new run")
            paths.validate_name("tag", tag)
            check_output_name(tag)
    except paths.PathPolicyError as error:
        raise ConfirmError(str(error)) from error
    if resume_dir is not None:
        if not (Path(resume_dir) / "manifest.json").is_file():
            raise ConfirmError(f"--resume {resume_dir}: no manifest.json there")
        return
    if prefix is not None and not tag.startswith(prefix):  # type: ignore[union-attr]
        raise ConfirmError(f"the label {tag!r} must start with {prefix!r}")
    stage_dir = root / stage
    if stage_dir.is_dir():
        for existing in sorted(stage_dir.iterdir()):
            match = _RUN_DIR_RE.fullmatch(existing.name)
            if match and match.group(1) == tag:
                raise ConfirmError(f"the label {tag!r} is already used in the stage {stage!r} by {existing.name}")


def check_model_dir(model_dir: str | Path) -> None:
    """The model directory exists and holds a configuration, a tokenizer and weight files (nothing is read)."""
    directory = Path(model_dir)
    if not directory.is_dir():
        raise ConfirmError(f"the model directory {directory} does not exist")
    problems = []
    if not (directory / "config.json").is_file():
        problems.append("config.json")
    if not ((directory / "tokenizer.json").is_file() or (directory / "tokenizer.model").is_file()):
        problems.append("tokenizer.json or tokenizer.model")
    if not (any(directory.glob("*.safetensors")) or any(directory.glob("*.bin"))):
        problems.append("weight files (*.safetensors or *.bin)")
    if problems:
        raise ConfirmError(f"the model directory {directory} lacks {', '.join(problems)}")


def check_bound_files(batch: BatchManifest, repo_root: str | Path = REPO_ROOT) -> None:
    """Every file the batch manifest binds still has its md5. The message lists the paths that differ (names and short md5 values only)."""
    problems = []
    for rel, expected in sorted(batch.files_md5.items()):
        path = Path(repo_root) / rel
        if not path.is_file():
            problems.append(f"{rel}: missing")
            continue
        actual = file_md5(path)
        if actual != expected:
            problems.append(f"{rel}: md5 {actual[:12]}, frozen {expected[:12]}")
    if problems:
        raise ConfirmError(f"{len(problems)} of the {len(batch.files_md5)} files bound by the batch manifest differ: " + "; ".join(problems))


def check_numeric_settings(
    batch: BatchManifest, model: str, *, gpu_mem_util: float | None, max_model_len: int | None = None, llm_kwargs: Mapping[str, Any] | str | None = None,
) -> None:
    """The memory fraction is a number in (0, 1] (the offline engine has no default that works on a shared card); when the batch manifest froze numeric settings for the model, the run's
    ``gpu_mem_util``, ``max_model_len`` (if the manifest has it) and ``llm_kwargs`` (a JSON text or a mapping; the manifest's, or none) are exactly those (for example 0.80 with online
    FP8 weights for the 14B model and 0.85 for the 32B model)."""
    if gpu_mem_util is None or not _is_number(gpu_mem_util) or not 0.0 < float(gpu_mem_util) <= 1.0:
        raise ConfirmError(f"--gpu-mem-util must be a number in (0, 1], got {gpu_mem_util!r}")
    if model not in batch.models:
        raise ConfirmError(f"model {model!r} is not in the batch manifest of {batch.batch_id!r}")
    frozen = batch.numeric.get(model)
    if frozen is None:
        return
    if isinstance(llm_kwargs, str):
        try:
            llm_kwargs = json.loads(llm_kwargs)
        except ValueError as error:
            raise ConfirmError(f"--llm-kwargs is not valid JSON: {error}") from error
    given = dict(llm_kwargs or {})
    differences = []
    if abs(float(gpu_mem_util) - float(frozen["gpu_mem_util"])) > 1e-9:
        differences.append(f"gpu_mem_util {gpu_mem_util} (frozen {frozen['gpu_mem_util']})")
    if "max_model_len" in frozen and max_model_len != frozen["max_model_len"]:
        differences.append(f"max_model_len {max_model_len} (frozen {frozen['max_model_len']})")
    if given != dict(frozen.get("llm_kwargs") or {}):
        differences.append(f"llm_kwargs {given} (frozen {dict(frozen.get('llm_kwargs') or {})})")
    if differences:
        raise ConfirmError(f"the numeric settings of {model!r} differ from those the batch manifest froze: " + "; ".join(differences))


def check_start_allowed(results_root: str | Path, batch: BatchManifest, *, model: str, arm: str, resume_dir: str | Path | None = None) -> None:
    """The confirmatory run of (batch, model, arm) exists once. A new run needs no binding yet; a run with ``--resume`` needs the binding to be exactly that directory. Checked before the
    pool is loaded (a new label for a bound (batch, model, arm) exits 2 here)."""
    binding = find_binding(results_root, batch, model=model, arm=arm)
    if resume_dir is None:
        if binding is not None:
            raise ConfirmError(
                f"({batch.batch_id}, {model}, {arm}) is already bound to the run {binding.run_dir}: the confirmatory run exists once and a new label does not start it again; "
                "continue it with --resume"
            )
        return
    if binding is None:
        raise ConfirmError(f"--resume {resume_dir}: ({batch.batch_id}, {model}, {arm}) has no bound run, so there is nothing to continue")
    if binding.run_dir != str(Path(resume_dir).resolve()):
        raise ConfirmError(f"--resume {resume_dir}: ({batch.batch_id}, {model}, {arm}) is bound to the run {binding.run_dir}, a resume continues that directory only")


def check_resume_identity(
    results_root: str | Path, batch: BatchManifest, *, model: str, arm: str, run_dir: str | Path, prereg_frozen_path: str | Path | None = None,
) -> None:
    """The run in ``run_dir`` is the bound run of (batch, model, arm) and its manifest was written for this batch: same batch id, pool, models, arms, batch manifest, frozen
    pre-registration and bound-file md5 values as now, no exploratory marker. A difference raises ``ConfirmError`` naming the keys that differ."""
    check_start_allowed(results_root, batch, model=model, arm=arm, resume_dir=run_dir)
    manifest_path = Path(run_dir) / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ConfirmError(f"--resume {run_dir}: manifest.json cannot be read ({type(error).__name__})") from error
    if not isinstance(manifest, dict) or not isinstance(manifest.get("confirmatory"), dict):
        raise ConfirmError(f"--resume {run_dir}: the manifest has no confirmatory block, so it is not a confirmatory run")
    if "exploratory" in manifest:
        raise ConfirmError(f"--resume {run_dir}: the manifest carries an exploratory marker, so it is not a confirmatory run")
    block = confirm_manifest_block(batch, prereg_frozen_path=FROZEN_PREREG_PATH if prereg_frozen_path is None else prereg_frozen_path)
    recorded, current = _stable_identity(manifest["confirmatory"]), _stable_identity(block["confirmatory"])
    differing = sorted(k for k in current if recorded.get(k) != current[k])
    recorded_md5 = manifest.get("input_md5", {})
    differing += sorted(f"input_md5.{k}" for k, v in block["input_md5"].items() if recorded_md5.get(k) != v)
    if differing:
        raise ConfirmError(f"--resume {run_dir}: the identity of the run differs from the batch in {differing}")


PREFLIGHT_CHECKS = ("pool of the batch", "label", "model directory", "batch identity", "bound files", "numeric settings")  # then "unique run" (a new run) or "resume identity" (--resume)


@dataclass(frozen=True)
class PreflightReport:
    """The result of ``run_preflight`` for one (batch, model, arm): every check with its outcome. ``ok`` needs every check to be there and to have passed (a report made by hand, with
    fewer checks, is not ``ok``); ``require()`` raises when it is not."""

    batch_id: str
    model: str
    arm: str
    checks: tuple[Mapping[str, Any], ...]

    @property
    def complete(self) -> bool:
        names = [c["name"] for c in self.checks]
        return names[: len(PREFLIGHT_CHECKS)] == list(PREFLIGHT_CHECKS) and names[len(PREFLIGHT_CHECKS):] in (["unique run"], ["resume identity"])

    @property
    def ok(self) -> bool:
        return self.complete and all(c["ok"] for c in self.checks)

    def failures(self) -> list[Mapping[str, Any]]:
        return [c for c in self.checks if not c["ok"]]

    def require(self) -> None:
        if not self.complete:
            raise ConfirmError(f"preflight incomplete, nothing was loaded or consumed: the checks are {[c['name'] for c in self.checks]}, run_preflight makes {list(PREFLIGHT_CHECKS)} and one more")
        if not self.ok:
            raise ConfirmError("preflight failed, nothing was loaded or consumed: " + "; ".join(f"{c['name']}: {c['detail']}" for c in self.failures()))


def run_preflight(
    results_root: str | Path,
    batch: BatchManifest,
    *,
    model: str,
    arm: str,
    pool: str,
    stage: str,
    tag: str | None,
    model_dir: str | Path,
    gpu_mem_util: float | None,
    max_model_len: int | None = None,
    llm_kwargs: Mapping[str, Any] | str | None = None,
    resume_dir: str | Path | None = None,
    repo_root: str | Path = REPO_ROOT,
    tag_prefix: str | None = None,
) -> PreflightReport:
    """Every check that comes before the pool is loaded, each reported with its outcome (none stops the others): the pool the entry reads is the pool of the batch, the label,
    the model directory, the batch identity, the md5 of the bound files, the numeric settings, the unique run of (batch, model, arm) and, with ``resume_dir``, the identity of the run that is
    continued. ``load_confirm_items`` loads only with a passing report."""
    checks: list[dict[str, Any]] = []

    def run(name: str, check: Callable[[], Any]) -> None:
        try:
            detail = check()
            checks.append({"name": name, "ok": True, "detail": "" if detail is None else detail})
        except (ConfirmError, OSError, ValueError) as error:
            checks.append({"name": name, "ok": False, "detail": str(error)})

    def entry_pool() -> None:
        if batch.pool != pool:
            raise ConfirmError(f"the entry reads {pool!r} but batch {batch.batch_id!r} is the batch of {batch.pool!r}")

    run("pool of the batch", entry_pool)
    run("label", lambda: check_label(results_root, stage, tag, prefix=tag_prefix, resume_dir=resume_dir))
    run("model directory", lambda: check_model_dir(model_dir))
    run("batch identity", lambda: check_batch_identity(results_root, batch, model=model, arm=arm))
    run("bound files", lambda: check_bound_files(batch, repo_root))
    run("numeric settings", lambda: check_numeric_settings(batch, model, gpu_mem_util=gpu_mem_util, max_model_len=max_model_len, llm_kwargs=llm_kwargs))
    if resume_dir is None:
        run("unique run", lambda: check_start_allowed(results_root, batch, model=model, arm=arm))
    else:
        run("resume identity", lambda: check_resume_identity(results_root, batch, model=model, arm=arm, run_dir=resume_dir))
    return PreflightReport(batch_id=batch.batch_id, model=model, arm=arm, checks=tuple(checks))


def load_confirm_items(
    results_root: str | Path,
    batch: BatchManifest,
    registry: SplitRegistry | TriviaSplit,
    *,
    model: str,
    arm: str,
    loader: Callable[[HoldoutAccess], Sequence[Any]],
    preflight: PreflightReport,
) -> list[Any]:
    """Load the pool of the batch under its access, after the preflight. ``loader(access)`` reads the pool with the existing loader (``load_items_from_config`` for MMLU-Redux,
    ``triviaqa.load_pool`` for the scenario) and must pass ``results_root=access.results_root`` and ``holdout_access=access``. The preflight must be a passing report for the same
    (batch, model, arm); the identity is checked once more, and the access is built from the manifest (full models and arms). The loader's items must be the registered key list of the pool, in
    order. Ledger and marker errors become ``ConfirmError``."""
    if not isinstance(preflight, PreflightReport) or (preflight.batch_id, preflight.model, preflight.arm) != (batch.batch_id, model, arm):
        raise ConfirmError(f"the pool is loaded only after a preflight of ({batch.batch_id}, {model}, {arm})")
    preflight.require()
    check_batch_identity(results_root, batch, model=model, arm=arm)
    access = open_confirm_access(results_root, batch, registry)
    try:
        items = list(loader(access))
    except (HoldoutSealedError, HoldoutLedgerError) as error:
        raise ConfirmError(str(error)) from error
    expected = list(registry.keys(batch.pool))
    if [item.item_key for item in items] != expected:
        raise ConfirmError(f"the loader returned {len(items)} items, which are not the {len(expected)} registered keys of {batch.pool!r} in pool order")
    # the keys must be in the ledger of THIS root under THIS batch: a loader that read the pool without the access (or on another root) did not consume them where the batch is accounted
    root = canonical_results_root(results_root)
    try:
        consumed = _ledger(root).consumed()
    except HoldoutLedgerError as error:
        raise ConfirmError(f"the consumption ledger cannot be read: {error}") from error
    missing = sum(consumed.get(key, {}).get("batch_id") != batch.batch_id for key in expected)
    if missing:
        raise ConfirmError(f"{missing} of the {len(expected)} keys of {batch.pool!r} are not in the ledger of {root} under batch {batch.batch_id!r}: the loader did not read through the access of this root")
    return items


def rehearsal_pool(pool: str) -> str:
    """The development pool a rehearsal reads instead of the sealed pool (``REHEARSAL_POOLS``)."""
    if pool not in REHEARSAL_POOLS:
        raise ConfirmError(f"{pool!r} is not a confirmatory pool; known: {sorted(REHEARSAL_POOLS)}")
    return REHEARSAL_POOLS[pool]


def rehearsal_manifest_block(pool: str) -> dict[str, Any]:
    """What a rehearsal adds to its manifest (``extra_manifest`` of ``_c2_common.open_run``): the exploratory marker that keeps the run out of every gate and every analysis
    (the evidence collection and the analyses read it), the reason, and which sealed pool it stands in for. A rehearsal binds no batch, writes no ledger and no ``confirm_runs.jsonl``."""
    development = rehearsal_pool(pool)
    return {
        "exploratory": True,
        "development_reason": (
            f"rehearsal of a confirmatory entry of pre-registration v2 on {development}, the development pool of the bank of {pool}: the same code, no batch, no ledger, no binding; "
            "it enters no gate and no analysis"
        ),
        "rehearsal": {"confirmatory_pool": pool, "pool": development},
    }


__all__ = [
    "ARMS_BY_POOL", "AttemptLimitError", "BATCH_MMLU", "BATCH_POOLS", "BATCH_SCENARIO", "BatchManifest", "ConfirmError", "FORBIDDEN_NAME_PARTS", "FROZEN_PREREG_PATH", "MAX_ATTEMPTS", "OUTCOMES",
    "PREFLIGHT_CHECKS", "PreflightReport", "REHEARSAL_POOLS", "RunBinding", "attempts", "bind_run", "canonical_results_root", "check_batch_identity", "check_bound_files", "check_label", "check_model_dir",
    "check_numeric_settings", "check_output_name", "check_resume_identity", "check_start_allowed", "confirm_manifest_block", "find_binding", "load_batch_manifest",
    "load_confirm_items", "open_confirm_access", "open_run_kwargs", "read_confirm_runs", "record_attempt", "rehearsal_manifest_block", "rehearsal_pool", "run_preflight", "state_of_records",
]
