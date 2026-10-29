"""The fixed state directory of the API experiment.

Everything that must be the same for every run of the experiment is here and nowhere else, in ONE absolute directory, ``FIXED_STATE_DIR`` (``results/api_exp/_state`` of the checkout at
``FIXED_CODE_ROOT``), whichever working tree the code runs from (the runs are in its parent, ``results/api_exp``):

* ``api_exp.lock``: the process lock (``flock``, held for a whole run: two runs never overlap);
* ``ledger.jsonl`` and ``successes.jsonl``: the total ledger and the index of successes (``nmi.api_exp.journal``);
* ``runs.json``: for each (mode, model) the one run directory it lives in. A new tag for a (mode, model) that has a run directory is refused: the run is continued instead;
* ``budget_state.json``: which models have been approved to start in each stage and where a stage stopped for budget, and the md5 of every price version the program has run with;
* ``freeze.json``: the identity of the experiment (the md5 of the protocol, of the frozen configuration, of the prompt file, of the local configuration, of the split, of all rendered
  messages and of the sources that build the request and read the reply), written just before the FIRST real send and checked by every run after it, for every model;
* ``freeze_<name>.json``: the same record of a SUPPLEMENTARY batch, written just before the first real send of that batch and checked by every run of that batch. A batch writes and compares
  its own file only: a supplementary batch never writes ``freeze.json`` and the first batch never writes ``freeze_<name>.json``. A run of a supplementary batch only READS ``freeze.json``:
  the first batch's configuration that it repeats must be the one in the record (``config_frozen_md5``) and its procedure must be the first batch's (``SHARED_FIELDS``). Everything else in
  this directory is shared by the batches.

Two more files of this directory are written by the operator by hand. No code of this package writes them, and neither is part of the identity above (``nmi.api_exp.release``,
``nmi.api_exp.lockout``):

* ``release.json``: the models that have been released, each with its stated basis and the time. A model that is not on it does not start; an absent file is an empty list;
* ``bound_disposition.json``: the operator's disposition of a bound violation, without which the experiment stays locked out after a reply with a count above the bound of its send.

There is no option and no environment variable that moves this directory, and the configuration may only repeat its path (the loader refuses another). A real run is refused unless the
code is the repository at ``FIXED_CODE_ROOT``. A test reaches another place only through the test-only parameters of ``scripts/run_api_decisions.py`` (``ApiHooks.results_root`` and
``ApiHooks.code_root``)."""

from __future__ import annotations

import contextlib
import fcntl
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nmi import paths
from nmi.api_exp.config import FIXED_CODE_ROOT, FIXED_STATE_DIR
from nmi.api_exp.durable import write_json_atomic

STAGE_DIR = "api_exp"
STATE_DIR_NAME = "_state"
LOCK_NAME = "api_exp.lock"
RUNS_NAME = "runs.json"
BUDGET_NAME = "budget_state.json"
FREEZE_NAME = "freeze.json"  # the first batch's record
# the values the freeze record keeps and every run compares (all md5 hex digests)
FREEZE_FIELDS = ("protocol_md5", "config_frozen_md5", "prompts_md5", "local_config_md5", "split_md5", "messages_md5", "code_md5")
# the value that a batch with request-and-reading code of its own adds to its record (the md5 of ``nmi/api_exp/openrouter.py`` for a batch of the kind openrouter); a record that does not have it
# does not compare it, and a run whose batch has none does not write it, so records written without it stay valid
OPTIONAL_FREEZE_FIELDS = ("batch_code_md5",)
# the values a supplementary batch shares with the first batch: the procedure (what is asked, how, and how the reply is read). The protocol text and the configuration (the models) can
# differ between batches, so each batch keeps those two in its own record only.
SHARED_FIELDS = ("prompts_md5", "local_config_md5", "split_md5", "messages_md5", "code_md5")


def freeze_file_name(batch: str | None = None) -> str:
    """``freeze.json`` for the first batch, ``freeze_<name>.json`` for the supplementary batch of that name."""
    return FREEZE_NAME if batch is None else f"freeze_{batch}.json"


class StateError(RuntimeError):
    """The state files contradict what the caller is about to do."""


def api_root(results_root: str | Path | None = None) -> Path:
    """The directory of the runs: ``results/api_exp`` of the checkout at ``FIXED_CODE_ROOT`` (the parent of the fixed state directory), or ``api_exp`` under the results root a test passes."""
    if results_root is None:
        return paths.assert_off_root_partition(Path(FIXED_STATE_DIR).parent)
    return paths.assert_off_root_partition(paths.results_root(results_root) / STAGE_DIR)


def state_dir(root: Path) -> Path:
    return root / STATE_DIR_NAME


def running_code_root() -> Path:
    """The repository this code is in."""
    return paths.REPO_ROOT


def code_root_problem(actual: str | Path | None = None) -> str | None:
    """Why the code (at ``actual``, by default the repository this code is in) may not send real requests, or None if it may. Only the repository at ``FIXED_CODE_ROOT`` may: the state is
    one absolute directory, and a copy of the repository elsewhere (a working tree) has other code, other prices and no right to spend the one budget. Such a copy can run the tests and
    ``--dry-run``."""
    here = Path(running_code_root() if actual is None else actual).resolve()
    if here != Path(FIXED_CODE_ROOT).resolve():
        return f"real requests are sent only from the repository {FIXED_CODE_ROOT}; this code is in {here} (it can run the tests and --dry-run)"
    return None


@contextlib.contextmanager
def process_lock(directory: Path) -> Iterator[None]:
    """Exclusive, non-blocking lock for the whole run: a second run of the experiment is refused, not queued."""
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / LOCK_NAME, "a") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise StateError(f"another run of the API experiment holds {directory / LOCK_NAME}; two runs must not overlap (the ledger is one file and the cap is hard)") from error
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        return default
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as error:
        raise StateError(f"{path.name} is not valid JSON ({type(error).__name__}); it is not to be guessed at") from None
    if not isinstance(data, dict):
        raise StateError(f"{path.name} is not a JSON object")
    return data


class Freeze:
    """The identity of the experiment, written once, just before the first real send, and compared by every later run. Another configuration file, another
    protocol or prompt file, other messages or other request and reading code cannot start a model, a run directory or a resume: the run ends with exit 2."""

    def __init__(self, directory: Path, batch: str | None = None) -> None:
        self.batch = batch  # None: the first batch's record, freeze.json
        self.path = directory / freeze_file_name(batch)
        self.data = _read_json(self.path, {})
        if self.data and not isinstance(self.data.get("identity"), dict):
            raise StateError(f"{self.path.name} has no identity; it is not to be guessed at")
        if self.data and self.data.get("batch") != batch:
            raise StateError(f"{self.path.name} is the record of the batch {self.data.get('batch')!r}, not of {batch!r}; it is not to be guessed at")

    @property
    def exists(self) -> bool:
        return bool(self.data)

    def mismatches(self, identity: Mapping[str, str], fields: Sequence[str] | None = None) -> list[str]:
        """The fields of the identity that differ from the recorded one (empty when the experiment is the frozen one, or is not frozen yet). ``fields``: the ones to compare; by default the
        seven of ``FREEZE_FIELDS`` and, if the record has it or the run brings it, the optional ``batch_code_md5``."""
        if not self.exists:
            return []
        recorded = self.data["identity"]
        if fields is None:
            fields = (*FREEZE_FIELDS, *(f for f in OPTIONAL_FREEZE_FIELDS if f in recorded or identity.get(f)))
        return [f for f in fields if recorded.get(f) != identity.get(f)]

    def write(self, identity: Mapping[str, str], *, run: str, now: str, files: Mapping[str, str]) -> None:
        if self.exists:
            raise StateError("the experiment is frozen already")
        absent = [f for f in FREEZE_FIELDS if not identity.get(f)]
        if absent:
            raise StateError(f"the identity of the experiment lacks {absent}")
        recorded = {f: str(identity[f]) for f in FREEZE_FIELDS}
        recorded.update({f: str(identity[f]) for f in OPTIONAL_FREEZE_FIELDS if identity.get(f)})
        self.data = {"format": 1, "frozen_at": now, "first_run": run, "identity": recorded, "files": dict(files)}
        if self.batch is not None:
            self.data["batch"] = self.batch  # the first batch's record has no such key, as before
        write_json_atomic(self.path, self.data)


class RunRegistry:
    """For each (mode, model) the run directory (its name under ``results/api_exp/``) it lives in."""

    def __init__(self, directory: Path) -> None:
        self.path = directory / RUNS_NAME
        self.data = _read_json(self.path, {"format": 1, "runs": {}})
        self.data.setdefault("runs", {})

    def run_of(self, mode: str, model: str) -> str | None:
        return self.data["runs"].get(mode, {}).get(model)

    def register(self, mode: str, models: Sequence[str], run_name: str) -> None:
        for model in models:
            held = self.run_of(mode, model)
            if held is not None and held != run_name:
                raise StateError(f"model {model} already has a run directory in mode {mode}: {held}")
        runs = self.data["runs"].setdefault(mode, {})
        for model in models:
            runs[model] = run_name
        write_json_atomic(self.path, self.data)

    def models_of(self, mode: str, run_name: str) -> list[str]:
        return sorted(m for m, r in self.data["runs"].get(mode, {}).items() if r == run_name)


@dataclass(frozen=True)
class AdmitDecision:
    admitted: bool
    reason: str  # approved, already_approved, budget or stage_stopped
    stage: str
    model: str
    order: int
    price_version: int
    spent_cny: float
    reserve_cny: float
    cap_cny: float
    stop_model: str | None = None  # the model whose refusal stopped the stage (for stage_stopped and budget)

    def as_record(self) -> dict[str, Any]:
        return {
            "admitted": self.admitted, "reason": self.reason, "stage": self.stage, "model": self.model, "order": self.order, "price_version": self.price_version,
            "spent_cny": self.spent_cny, "reserve_cny": self.reserve_cny, "cap_cny": self.cap_cny, "stop_model": self.stop_model,
        }


class BudgetState:
    """The persisted budget approvals and the md5 of every price version seen.

    A model that has been approved to start in a stage stays approved for that stage and that price version: a resume does not ask the budget again (the hard check before every send
    still applies). A model that is refused for budget stops the stage: no model that comes after it in the run order starts in that stage, whether the run is resumed or another model is
    named, until the price version changes, which opens the approvals again in order."""

    def __init__(self, directory: Path) -> None:
        self.path = directory / BUDGET_NAME
        self.data = _read_json(self.path, {"format": 1, "price_versions": {}, "stages": {}})
        self.data.setdefault("price_versions", {})
        self.data.setdefault("stages", {})

    def save(self) -> None:
        write_json_atomic(self.path, self.data)

    # ---- price versions: a price version is one set of numbers ------------------------------
    def check_price_identity(self, version: int, md5: str) -> None:
        known = self.data["price_versions"].get(str(version))
        if known is not None and known != md5:
            raise StateError(
                f"the price file says version {version}, but this program has run with another file under that version (md5 {known}, now {md5}); raise `version` when the prices change"
            )
        if known is None:
            self.data["price_versions"][str(version)] = md5
            self.save()

    # ---- budget approvals per stage ---------------------------------------------------------
    def stage_state(self, stage: str, price_version: int) -> dict[str, Any]:
        st = self.data["stages"].get(stage)
        if st is None or st.get("price_version") != price_version:
            st = {"price_version": price_version, "approved": {}, "stop": None}
            self.data["stages"][stage] = st
        return st

    def peek(self, stage: str, model: str, order: int, price_version: int) -> str:
        """What ``admit`` would answer without the budget test: already_approved, stage_stopped or open (the budget has still to be tested). Changes nothing."""
        st = self.data["stages"].get(stage)
        if st is None or st.get("price_version") != price_version:
            return "open"
        if model in st["approved"]:
            return "already_approved"
        stop = st.get("stop")
        if stop is not None and order >= stop["order"]:
            return "stage_stopped"
        return "open"

    def admit(self, *, stage: str, model: str, order: int, price_version: int, spent_cny: float, reserve_cny: float, cap_cny: float, now: str) -> AdmitDecision:
        st = self.stage_state(stage, price_version)

        def decision(admitted: bool, reason: str, stop_model: str | None = None) -> AdmitDecision:
            return AdmitDecision(admitted, reason, stage, model, order, price_version, spent_cny, reserve_cny, cap_cny, stop_model)

        if model in st["approved"]:
            return decision(True, "already_approved")
        stop = st.get("stop")
        if stop is not None and order >= stop["order"]:
            return decision(False, "stage_stopped", stop["model"])
        if spent_cny + reserve_cny > cap_cny:
            if stop is None or order < stop["order"]:
                st["stop"] = {"model": model, "order": order, "t": now, "spent_cny": spent_cny, "reserve_cny": reserve_cny}
            self.save()
            return decision(False, "budget", model)
        st["approved"][model] = {"order": order, "t": now, "spent_cny": spent_cny, "reserve_cny": reserve_cny}
        self.save()
        return decision(True, "approved")
