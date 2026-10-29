"""TriviaQA (rc.nocontext validation) for the real scenario S1: the task adapter of the data, the two pools and their registry.

Source. ``mandarjoshi/trivia_qa``, configuration ``rc.nocontext``, split ``validation``, revision ``0f7faf33a3908546c6fd5b73a660e0f8ff173c2f`` (HF cache). The file has two rows for most
questions (the Wikipedia and the Web evidence sets); only the columns ``question`` and ``answer`` are read.

Rows to items:

* The rows are grouped, in file order, by the normalised question (``normalize`` below). A group takes the question text of its FIRST row; its answers are the union, in order of first
  appearance, of ``aliases``, ``normalized_aliases`` and ``value`` of every row of the group (row by row, in this order), blank strings dropped. A group with no answer left cannot be
  scored and takes part in no pool (the count is in the manifest).
* The number of groups whose rows disagree about the answers is counted and written to the manifest, twice: the answer SETS differ as strings, and the answer sets differ after ``normalize``.
* ITEM KEY = sha256(normalize(question) + "|" + normalize(first answer)), the same shape as ``nmi.data.keys.item_key``. The normalisation is the one that scores the answers (NFKC, lower
  case, ASCII punctuation removed, the articles a/an/the removed, white space collapsed), so that one function serves the group, the key and the score. The ``normalize`` of
  ``nmi.data.keys`` (which keeps the punctuation) is not used: the 200 pilot keys (kept in the split as ``keys/triviaqa_s1_pilot.keys``) were made with the scoring normalisation, and the keys of the
  development pool must contain them. On this file both normalisations give the same 9,961 groups (checked when the split was made, not by the code).

Pools (seed 20261011; hash ranks as in the sealing script of the study, not a random-number stream, so that the draw does not depend on a library version):

1. ``triviaqa_s1_dev`` (1,000 items) starts with the 200 pilot items (their groups are found by the pilot keys; each must be found exactly once), then takes the 800 groups with the smallest
   ``sha256("<seed>|development|<key>")`` among the other groups.
2. ``triviaqa_s1_sealed`` (2,000 items) takes the groups with the smallest ``sha256("<seed>|sealed|<key>")`` among the groups that are in neither the pilot nor the 800.
Both pools are listed in the order of the first row of the group in the file. They share no group (a group is a normalised question, so they share no question). Paraphrases of a question that
the normalisation does not merge are not found by this rule.

Registry (the format of ``nmi.data.registry``): ``pools.json`` (rules, source, counts, pool sizes and md5/sha256 of each key list), ``keys/<pool>.keys`` (one key per line, pool
order) and ``locator_map.json`` (the sha256 of the source file and, for every key, the 0-based rows of its group). ``keys/triviaqa_s1_pilot.keys`` holds the 200 pilot keys in pilot order.
The sealing script of the study (not part of this release) is the only code that computes keys and assigns groups to pools. ``TriviaSplit.load`` verifies the artifacts; ``load_pool`` reads a pool
through the frozen locator map and drops every other row at once.

The sealed pool follows the rules of ``nmi.data.holdout``: ``guard_pool_request`` refuses it without the unseal marker file and a ``HoldoutAccess`` of a registered batch, and the
ledger records every key that is read (a consumed key is development data for ever). No entry of the scenario reads it before the confirmatory phase (``scripts/run_scenario.py`` exits 2).

Nothing here prints or logs a question or an answer.
"""

from __future__ import annotations

import hashlib
import json
import re
import string
import unicodedata
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nmi.data.holdout import HoldoutAccess, guard_pool_request
from nmi.data.keys import keylist_digests
from nmi.data.registry import (
    SCENARIO_DEV_POOL,
    SCENARIO_SEALED_POOL,
    InputHashMismatchError,
    HoldoutLeakError,
    SplitError,
    is_holdout_pool,
)
from nmi.paths import split_dir
from nmi.provenance import bytes_md5, file_sha256

FORMAT = 1
BANK = "triviaqa_s1"
DEV_POOL = SCENARIO_DEV_POOL
SEALED_POOL = SCENARIO_SEALED_POOL
POOLS = (DEV_POOL, SEALED_POOL)
DEFAULT_SPLIT_VERSION = "s1_triviaqa_v1"

SOURCE_FILE = "validation-00000-of-00001.parquet"
INPUT_NAME = f"{BANK}/{SOURCE_FILE}"

NORMALIZE_RULE = "NFKC, lowercase, remove ASCII punctuation, remove the articles a/an/the, collapse whitespace"
ITEM_KEY_RULE = "sha256(normalize(question) + '|' + normalize(first answer))"
_KEY_RE = re.compile(r"^[0-9a-f]{64}$")
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = frozenset(string.punctuation)


class TriviaQAError(SplitError):
    """The source file, the pilot keys or the artifacts are not what the registry was made from."""


# ---------------------------------------------------------------------------------------
# normalisation and keys
# ---------------------------------------------------------------------------------------

def normalize(text: str) -> str:
    """NFKC, lower case, ASCII punctuation removed, the articles a/an/the removed, white space collapsed: the one normalisation of the group, the key and the score."""
    if not isinstance(text, str):
        raise TypeError(f"normalize expects str, got {type(text).__name__}")
    folded = unicodedata.normalize("NFKC", text).lower()
    without_punctuation = "".join(ch for ch in folded if ch not in _PUNCTUATION)
    return " ".join(_ARTICLES.sub(" ", without_punctuation).split())


def item_key(question: str, first_answer: str) -> str:
    """Lower-case hex sha256 of ``normalize(question) | normalize(first answer)`` (UTF-8)."""
    return hashlib.sha256((normalize(question) + "|" + normalize(first_answer)).encode("utf-8")).hexdigest()


def locator(row: int) -> str:
    """Source locator of a row of the validation file (0-based)."""
    return f"{INPUT_NAME}#{row}"


# ---------------------------------------------------------------------------------------
# reading the source
# ---------------------------------------------------------------------------------------

def _strings(value: Any) -> list[str]:
    """A cell that holds one string, a list or array of strings, or nothing, as a list of str."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def row_answers(cell: Mapping[str, Any] | None) -> tuple[str, ...]:
    """The answers of one row: aliases, normalized_aliases and value, in this order, each string once, blank strings dropped."""
    if not isinstance(cell, Mapping):
        return ()
    joined = _strings(cell.get("aliases")) + _strings(cell.get("normalized_aliases")) + _strings(cell.get("value"))
    return tuple(s for s in dict.fromkeys(joined) if s.strip())


@dataclass(frozen=True)
class RawRow:
    row: int  # 0-based position in the file
    question: str
    answers: tuple[str, ...]


def iter_rows(path: str | Path, *, batch_size: int = 1024) -> Iterator[RawRow]:
    """The rows of the validation file in file order, one at a time (columns ``question`` and ``answer`` only). Errors mention counts and positions, never text."""
    import pyarrow.parquet as pq

    source = Path(path)
    try:
        parquet = pq.ParquetFile(source)
    except Exception as error:  # noqa: BLE001
        raise TriviaQAError(f"{source.name}: cannot be opened as a parquet file ({type(error).__name__})") from error
    names = set(parquet.schema_arrow.names)
    if not {"question", "answer"} <= names:
        raise TriviaQAError(f"{source.name}: needs the columns question and answer (found {sorted(names)})")
    row = 0
    for batch in parquet.iter_batches(batch_size=batch_size, columns=["question", "answer"]):
        questions = batch.column("question").to_pylist()
        cells = batch.column("answer").to_pylist()
        for question, cell in zip(questions, cells, strict=True):
            if not isinstance(question, str):
                raise TriviaQAError(f"{source.name}: row {row} has no question string")
            yield RawRow(row=row, question=question, answers=row_answers(cell))
            row += 1


# ---------------------------------------------------------------------------------------
# groups
# ---------------------------------------------------------------------------------------

def _union(answer_lists: Iterable[Sequence[str]]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(a for answers in answer_lists for a in answers))


# ---------------------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------------------

class TriviaSplit:
    """Verified view of one frozen split directory of the scenario (the counterpart of ``nmi.data.registry.SplitRegistry``)."""

    def __init__(
        self,
        root: Path,
        pools_doc: Mapping[str, Any],
        keys: Mapping[str, tuple[str, ...]],
        pilot_keys: tuple[str, ...],
        locator_doc: Mapping[str, Any],
    ) -> None:
        self.root = root
        self.doc = pools_doc
        self.version: str = pools_doc["version"]
        self._keys = dict(keys)
        self._key_sets = {name: frozenset(values) for name, values in self._keys.items()}
        self.pilot_keys = pilot_keys
        self._inputs: Mapping[str, Mapping[str, Any]] = locator_doc["inputs"]
        self._groups: Mapping[str, Sequence[int]] = locator_doc["groups"]
        self._row_key: dict[int, str] = {row: key for key, rows in self._groups.items() for row in rows}
        self.file_md5s: dict[str, str] = {}
        self.meta: Mapping[str, Mapping[str, Any]] = pools_doc["pools"]

    # ---- loading -------------------------------------------------------------------------
    @classmethod
    def load(cls, root: str | Path | None = None) -> "TriviaSplit":
        directory = Path(root) if root is not None else split_dir(DEFAULT_SPLIT_VERSION)
        pools_path, map_path = directory / "pools.json", directory / "locator_map.json"
        for needed in (pools_path, map_path):
            if not needed.is_file():
                raise TriviaQAError(f"split artifact {needed} is missing; it is written by the sealing script of the study, which is not part of this release")
        pools_doc = json.loads(pools_path.read_text(encoding="utf-8"))
        locator_doc = json.loads(map_path.read_text(encoding="utf-8"))
        cls._check_docs(pools_doc, locator_doc)
        keys: dict[str, tuple[str, ...]] = {}
        for name, meta in pools_doc["pools"].items():
            keys[name] = cls._read_keys(directory, name, meta)
        pilot_meta = pools_doc["pilot"]
        pilot = cls._read_keys(directory, "pilot", pilot_meta)
        split = cls(directory, pools_doc, keys, pilot, locator_doc)
        split.assert_consistent()
        split.file_md5s = {"pools.json": bytes_md5(pools_path.read_bytes()), "locator_map.json": bytes_md5(map_path.read_bytes())}
        return split

    @staticmethod
    def _read_keys(directory: Path, name: str, meta: Mapping[str, Any]) -> tuple[str, ...]:
        path = directory / meta["file"]
        if not path.is_file():
            raise TriviaQAError(f"key list {path} is missing")
        values = tuple(path.read_text(encoding="ascii").splitlines())
        if not all(_KEY_RE.fullmatch(v) for v in values):
            raise TriviaQAError(f"key list of {name!r} contains a line that is not a sha256 key")
        md5, sha = keylist_digests(values)
        if md5 != meta["md5"] or sha != meta["sha256"]:
            raise TriviaQAError(f"key list of {name!r} does not match its recorded hash (file md5 {md5}, recorded {meta['md5']})")
        if len(values) != meta["n"]:
            raise TriviaQAError(f"{name!r} has {len(values)} keys, recorded {meta['n']}")
        return values

    @staticmethod
    def _check_docs(pools_doc: Mapping[str, Any], locator_doc: Mapping[str, Any]) -> None:
        for label, doc in (("pools.json", pools_doc), ("locator_map.json", locator_doc)):
            if doc.get("format") != FORMAT:
                raise TriviaQAError(f"{label}: unsupported format {doc.get('format')!r}")
            if doc.get("normalize_rule") != NORMALIZE_RULE:
                raise TriviaQAError(f"{label}: normalize rule {doc.get('normalize_rule')!r} differs from the code ({NORMALIZE_RULE!r}); the keys were frozen under another rule")
        if pools_doc.get("item_key_rule") != ITEM_KEY_RULE:
            raise TriviaQAError("pools.json: item key rule differs from the code")
        listed = pools_doc.get("pools")
        if not isinstance(listed, Mapping) or set(listed) != set(POOLS):
            raise TriviaQAError(f"pools.json lists pools {sorted(listed) if isinstance(listed, Mapping) else listed!r}, expected {sorted(POOLS)}")
        if not isinstance(pools_doc.get("pilot"), Mapping) or not isinstance(pools_doc.get("version"), str):
            raise TriviaQAError("pools.json: needs a version and a pilot record")
        if not isinstance(locator_doc.get("groups"), Mapping) or not isinstance(locator_doc.get("inputs"), Mapping) or INPUT_NAME not in locator_doc["inputs"]:
            raise TriviaQAError(f"locator_map.json: needs groups and the input {INPUT_NAME!r}")

    # ---- lookups -------------------------------------------------------------------------
    def keys(self, pool: str) -> tuple[str, ...]:
        """Frozen key list of a pool, in pool order."""
        if pool not in self._keys:
            raise TriviaQAError(f"unknown pool {pool!r}; the scenario split has {list(POOLS)}")
        return self._keys[pool]

    def key_set(self, pool: str) -> frozenset[str]:
        if pool not in self._key_sets:
            raise TriviaQAError(f"unknown pool {pool!r}; the scenario split has {list(POOLS)}")
        return self._key_sets[pool]

    def pool_md5(self, pool: str) -> str:
        return str(self.meta[pool]["md5"])

    def manifest_md5s(self, pools: Iterable[str] | None = None) -> dict[str, str]:
        """``split.<name>`` -> md5 for a run manifest: the key lists asked for and the two description files."""
        names = list(pools) if pools is not None else list(POOLS)
        out = {f"split.{name}": self.pool_md5(name) for name in names}
        out.update({f"split.{name}": md5 for name, md5 in self.file_md5s.items()})
        return out

    def key_of_row(self, row: int) -> str | None:
        """Frozen key of a source row, or None when the row is in no pool."""
        return self._row_key.get(row)

    def rows_of(self, key: str) -> tuple[int, ...]:
        return tuple(self._groups[key])

    def source_record(self) -> Mapping[str, Any]:
        """The record of the source file the map is bound to: sha256, number of rows, revision."""
        return self._inputs[INPUT_NAME]

    def verify_input(self, name: str, path: str | Path) -> Mapping[str, Any]:
        """Check that the file still has the hash the locator map was bound to; return its record."""
        record = self._inputs.get(name)
        if record is None:
            raise InputHashMismatchError(f"input {name!r} is not registered in the locator map")
        actual = file_sha256(path)
        if actual != record["sha256"]:
            raise InputHashMismatchError(
                f"input {name!r} has sha256 {actual[:16]}..., the locator map was bound to {record['sha256'][:16]}...; the source file changed after sealing"
            )
        return record

    # ---- guards ----------------------------------------------------------------------------
    def assert_consistent(self) -> None:
        """Disjoint pools, pilot inside the development pool, and a locator map that hands out exactly the keys of the pools."""
        dev, sealed = self._key_sets[DEV_POOL], self._key_sets[SEALED_POOL]
        for name, values in self._keys.items():
            if len(set(values)) != len(values):
                raise TriviaQAError(f"pool {name!r} contains duplicate keys")
        shared = dev & sealed
        if shared:
            raise TriviaQAError(f"the development and the sealed pool share {len(shared)} keys")
        if not set(self.pilot_keys) <= dev:
            raise TriviaQAError("the development pool does not contain every pilot key")
        if set(self._groups) != dev | sealed:
            raise TriviaQAError("the locator map and the key lists disagree about the keys")
        if len(self._row_key) != sum(len(rows) for rows in self._groups.values()):
            raise TriviaQAError("a source row belongs to two groups in the locator map")

    def assert_no_sealed(self, keys: Iterable[str], *, context: str = "") -> None:
        """No sealed key may pass a filter that only lets development items through."""
        leaked = self._key_sets[SEALED_POOL].intersection(keys)
        if leaked:
            raise HoldoutLeakError(f"{len(leaked)} sealed keys passed the filter{' in ' + context if context else ''}")


# ---------------------------------------------------------------------------------------
# the items
# ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class TriviaItem:
    """One question as presented to a model, with the answers that score it."""

    item_key: str
    pool: str
    bank: str
    locator: str  # of the first row of the group
    question: str
    answers: tuple[str, ...]
    rows: tuple[int, ...]


def load_pool(
    split: TriviaSplit,
    pool: str,
    source: str | Path,
    *,
    results_root: str | Path | None = None,
    holdout_access: HoldoutAccess | None = None,
) -> list[TriviaItem]:
    """Load one pool through the frozen locator map. A row that is not in the requested pool is dropped as it is read; the keys, the order and the sizes must be those of the frozen key list.

    The sealed pool passes ``guard_pool_request`` first (the unseal marker file and a ``HoldoutAccess``) and is entered in the consumption ledger after it has been read."""
    guard_pool_request(pool, results_root=results_root, access=holdout_access)
    if pool not in POOLS:
        raise TriviaQAError(f"{pool!r} is not a pool of the scenario S1; known: {list(POOLS)}")
    path = Path(source)
    record = split.verify_input(INPUT_NAME, path)
    wanted = split.key_set(pool)
    kept: dict[str, list[RawRow]] = {}
    n_rows = 0
    for raw in iter_rows(path):
        n_rows = raw.row + 1
        key = split.key_of_row(raw.row)
        if key is None or key not in wanted:
            continue  # not part of this pool: dropped here
        kept.setdefault(key, []).append(raw)
    if n_rows != record["n_rows"]:
        raise TriviaQAError(f"{path.name}: read {n_rows} rows, the locator map recorded {record['n_rows']}")
    items: list[TriviaItem] = []
    for key in split.keys(pool):
        rows = kept.get(key)
        if rows is None or tuple(r.row for r in rows) != split.rows_of(key):
            raise TriviaQAError(f"pool {pool!r}: the rows of a frozen group are not those in the file")
        answers = _union(r.answers for r in rows)
        if not answers or item_key(rows[0].question, answers[0]) != key:
            raise TriviaQAError(f"pool {pool!r}: a group no longer gives its frozen key (the source file or the rules changed)")
        items.append(
            TriviaItem(
                item_key=key, pool=pool, bank=BANK, locator=locator(rows[0].row), question=rows[0].question, answers=answers, rows=tuple(r.row for r in rows),
            )
        )
    if not is_holdout_pool(pool):
        split.assert_no_sealed([i.item_key for i in items], context=pool)
    else:
        if holdout_access is None:  # guard_pool_request has refused this already; kept so that the type is narrowed and the invariant is explicit
            raise TriviaQAError("a sealed pool is read through a HoldoutAccess only")
        holdout_access.consume(pool, [i.item_key for i in items])
    return items
