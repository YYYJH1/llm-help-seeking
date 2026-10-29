"""MMLU-Redux 2.0 (edinburgh-dawg; CC-BY-4.0), snapshot pinned in ``data.mmlu_redux.source``.

The snapshot has one directory per subject (57), each with ``data-00000-of-00001.arrow``
(an Arrow IPC stream written by HF datasets), ``dataset_info.json`` and ``state.json``.
Columns used here: ``question: string``, ``choices: list<string>`` (4 options),
``answer: int64`` (index of the correct option) and ``error_type: string``; only rows
marked ``ok`` are used. The correct answer text is ``choices[answer]``.

As in ``simpleqa_mc``: ``SubjectTable`` is the raw reader (the sealing script of the study
materialises every row with it), and ``load_pool`` is the experiment loader that locates
rows through the frozen locator map, never touches the text of a row outside the requested
pool, and never computes a key.

Pool order is (subject name ascending, row ascending); the frozen key lists use the same
order, so a loaded pool must reproduce its key list exactly.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc

from nmi.data.holdout import HoldoutAccess, guard_pool_request
from nmi.data.registry import (
    BANK_MMLU,
    MMLU_EVAL_SHUFFLED,
    McItem,
    SplitError,
    SplitRegistry,
    is_holdout_pool,
    physical_pool,
)
from nmi.data.simpleqa_mc import DEFAULT_ARRANGEMENT_SEED, apply_arrangement, generate_fixed_arrangements

ARROW_NAME = "data-00000-of-00001.arrow"
_ARROW_FILE_MAGIC = b"ARROW1"


def locator(subject: str, row: int) -> str:
    """Source locator of a row (0-based row inside the subject's file)."""
    return f"{BANK_MMLU}/{subject}#{row}"


def input_name(subject: str) -> str:
    return f"{BANK_MMLU}/{subject}/{ARROW_NAME}"


def list_subjects(root: str | Path) -> list[str]:
    """Subject directories that contain the arrow file, alphabetical."""
    root = Path(root)
    if not root.is_dir():
        raise SplitError(f"MMLU-Redux directory {root} does not exist")
    return sorted(p.name for p in root.iterdir() if p.is_dir() and (p / ARROW_NAME).is_file())


@dataclass(frozen=True)
class RawRow:
    locator: str
    subject: str
    row: int
    error_type: str
    question: str
    choices: tuple[str, str, str, str]
    answer: int  # index of the correct option

    @property
    def correct_text(self) -> str:
        return self.choices[self.answer]


def _is_string(t: pa.DataType) -> bool:
    return pa.types.is_string(t) or pa.types.is_large_string(t)


def _check_schema(schema: pa.Schema, name: str) -> None:
    def field_type(column: str) -> pa.DataType:
        index = schema.get_field_index(column)
        if index < 0:
            raise SplitError(f"{name}: column {column!r} is missing")
        return schema.field(index).type

    for column in ("question", "error_type"):
        if not _is_string(field_type(column)):
            raise SplitError(f"{name}: column {column!r} is not a string column")
    choices = field_type("choices")
    if not (pa.types.is_list(choices) or pa.types.is_large_list(choices)) or not _is_string(
        choices.value_type
    ):
        raise SplitError(f"{name}: column 'choices' is not list<string>")
    if not pa.types.is_integer(field_type("answer")):
        raise SplitError(f"{name}: column 'answer' is not an integer column")


def _read_table(path: Path) -> pa.Table:
    data = path.read_bytes()  # a few hundred KB; no memory map, so no lifetime hazards
    source = pa.BufferReader(data)
    reader = ipc.open_file(source) if data[: len(_ARROW_FILE_MAGIC)] == _ARROW_FILE_MAGIC else ipc.open_stream(source)
    return reader.read_all()


class SubjectTable:
    """One subject's arrow file. Row text is materialised only when ``row`` is called."""

    def __init__(self, path: str | Path, subject: str) -> None:
        self.path = Path(path)
        self.subject = subject
        self._table = _read_table(self.path)
        _check_schema(self._table.schema, f"{subject}/{ARROW_NAME}")

    @property
    def n_rows(self) -> int:
        return self._table.num_rows

    @property
    def error_types(self) -> list[str]:
        return self._table.column("error_type").to_pylist()

    def row(self, i: int) -> RawRow:
        t = self._table
        question = t.column("question")[i].as_py()
        choices = t.column("choices")[i].as_py()
        answer = t.column("answer")[i].as_py()
        error_type = t.column("error_type")[i].as_py()
        where = f"{self.subject} row {i}"
        if question is None or error_type is None or answer is None or choices is None:
            raise SplitError(f"{where}: null value in a required column")
        if len(choices) != 4 or any(c is None for c in choices):
            raise SplitError(f"{where}: expected 4 options, found {len(choices)}")
        if not 0 <= answer < 4:
            raise SplitError(f"{where}: answer index {answer} is outside 0..3")
        return RawRow(
            locator=locator(self.subject, i), subject=self.subject, row=i, error_type=error_type,
            question=question, choices=(choices[0], choices[1], choices[2], choices[3]), answer=answer,
        )

    def rows(self) -> Iterator[RawRow]:
        """Every row with its text. For the sealing script of the study only."""
        for i in range(self.n_rows):
            yield self.row(i)


def load_pool(
    registry: SplitRegistry,
    pool: str,
    source_dir: str | Path,
    *,
    arrangement_seed: int = DEFAULT_ARRANGEMENT_SEED,
    keep_error_types: Sequence[str] = ("ok",),
    results_root: str | Path | None = None,
    holdout_access: HoldoutAccess | None = None,
) -> list[McItem]:
    """Load one MMLU-Redux pool through the frozen locator map.

    ``mmlu_redux_eval_shuffled`` returns the evaluation items with options shuffled by
    Kumaran's fixed arrangements (seed ``arrangement_seed``, item ``i`` of the pool gets
    arrangement ``i``); the other pools keep the original option order.
    """
    guard_pool_request(pool, results_root=results_root, access=holdout_access)
    base = physical_pool(pool)
    wanted = registry.key_set(base)
    subjects = list_subjects(source_dir)
    recorded = set(registry.recorded_inputs(BANK_MMLU))
    present = {input_name(s) for s in subjects}
    if present != recorded:
        raise SplitError(
            f"subject files differ from the sealed set ({len(present - recorded)} unexpected, "
            f"{len(recorded - present)} missing)"
        )
    keep = set(keep_error_types)
    ordered_keys: list[str] = []
    seen: set[str] = set()
    pending: list[tuple[str, str, RawRow]] = []  # (key, locator, row) in pool order
    for subject in subjects:
        path = Path(source_dir) / subject / ARROW_NAME
        record = registry.verify_input(input_name(subject), path)
        table = SubjectTable(path, subject)
        if table.n_rows != record["n_rows"]:
            raise SplitError(f"{subject}: read {table.n_rows} rows, the locator map recorded {record['n_rows']}")
        for i, error_type in enumerate(table.error_types):
            if error_type not in keep:
                continue
            loc = locator(subject, i)
            key = registry.key_for_locator(loc)
            if key is None:
                raise SplitError(f"row {loc} is marked usable but is not in the locator map")
            if key not in wanted or key in seen:
                continue  # outside the requested pool (or a repeat of a kept key): never materialised
            seen.add(key)
            ordered_keys.append(key)
            pending.append((key, loc, table.row(i)))
    if tuple(ordered_keys) != registry.keys(base):
        raise SplitError(f"pool {pool!r}: loaded keys differ from the frozen key list")
    shuffled = pool == MMLU_EVAL_SHUFFLED
    arrangements = generate_fixed_arrangements(len(pending), arrangement_seed) if shuffled else None
    items: list[McItem] = []
    for position, (key, loc, raw) in enumerate(pending):
        if arrangements is not None:
            order = arrangements[position]
            shown, gt_position = apply_arrangement(raw.choices, order, correct_index=raw.answer)
        else:
            order, shown, gt_position = None, raw.choices, raw.answer + 1
        items.append(
            McItem(
                item_key=key, pool=pool, bank=BANK_MMLU, locator=loc, subject=raw.subject,
                question=raw.question, choices=shown, gt_position=gt_position,
                correct_text=raw.correct_text, arrangement=order, row=raw.row,
            )
        )
    if not is_holdout_pool(pool):
        registry.assert_no_holdout(ordered_keys, context=pool)
    else:
        assert holdout_access is not None
        holdout_access.consume(pool, ordered_keys)
    return items
