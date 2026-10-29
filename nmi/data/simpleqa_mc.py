"""SimpleQA four-option bank (Kumaran et al. 2026, OSF resw4; data CC-BY-4.0).

Files in ``data.simpleqa_mc.source``:

    phase0_questions.csv, phase1_questions.csv   columns question, choice_1..choice_4
    simpleQA_4MCQ_dataset.csv                    the same plus gt_answer (full file, 4,326 items)

Where the correct answer is differs between the files:

* phase files: ``choice_1`` is the correct answer, by the convention of the OSF release;
* full file: ``gt_answer`` is the 1-BASED POSITION of the correct option, so the answer text is
  ``choice_{gt_answer}`` (OSF README: phase 1 is the full file "filtered to gt_answer == 1").
  ``gt_answer`` is not an answer text.

The calibration pool is all of phase 0, the evaluation pool all of phase 1, and the holdout pool
the part of the full file whose keys are in neither (decided by the sealing script of the study,
never here).

Options are shown in the order of Kumaran's ``generate_fixed_arrangements(n, seed=1042)``:
for each question in turn ``random.shuffle(list(range(4)))``.

Two kinds of readers live here:

* ``iter_records``: the raw parser of a source file.
* ``load_pool``: the experiment loader. It locates each row through the frozen locator map,
  drops every row that is not in the requested pool immediately, and asserts that no
  holdout key got through. It never computes a key.
"""

from __future__ import annotations

import csv
import random
from collections.abc import Iterator, Mapping
from pathlib import Path

from nmi.data.holdout import HoldoutAccess, guard_pool_request
from nmi.data.registry import (
    BANK_SIMPLEQA,
    SIMPLEQA_CAL,
    SIMPLEQA_EVAL,
    SIMPLEQA_HOLDOUT,
    McItem,
    SplitError,
    SplitRegistry,
    is_holdout_pool,
)

PHASE_COLUMNS = ("question", "choice_1", "choice_2", "choice_3", "choice_4")
FULL_COLUMNS = PHASE_COLUMNS + ("gt_answer",)
DEFAULT_ARRANGEMENT_SEED = 1042


def locator(file_name: str, row: int) -> str:
    """Source locator of a data row (0-based, header excluded)."""
    return f"{BANK_SIMPLEQA}/{file_name}#{row}"


def input_name(file_name: str) -> str:
    return f"{BANK_SIMPLEQA}/{file_name}"


_GT_INDEX = {"1": 0, "2": 1, "3": 2, "4": 3}


def parse_gt_index(text: str | None) -> int | None:
    """0-based index of the correct option from the full file's ``gt_answer`` cell, or None.

    ``gt_answer`` is the 1-based position of the correct option. After stripping surrounding white
    space the cell must be exactly "1", "2", "3" or "4"; anything else ("0", "5", "x", "", "1.0",
    "01", a non-ASCII digit) is invalid and gives None.
    """
    return _GT_INDEX.get(text.strip()) if text is not None else None


def iter_records(
    path: str | Path, columns: tuple[str, ...] = PHASE_COLUMNS
) -> Iterator[tuple[int, dict[str, str]]]:
    """Yield ``(row index, {column: text})`` for every data row; blank lines are skipped.

    Everything is read as text (a bare ``None`` or ``NA`` answer is a string, not a missing
    value). Errors mention counts and positions only, never cell content.
    """
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration as error:
            raise SplitError(f"{Path(path).name}: file is empty") from error
        index = {name: pos for pos, name in enumerate(header)}
        missing = [c for c in columns if c not in index]
        if missing:
            raise SplitError(f"{Path(path).name}: missing columns {missing}")
        width = max(index[c] for c in columns) + 1
        row = 0
        for fields in reader:
            if not fields:
                continue
            if len(fields) < width:
                raise SplitError(
                    f"{Path(path).name}: row {row} has {len(fields)} fields, expected at least {width}"
                )
            yield row, {c: fields[index[c]] for c in columns}
            row += 1


def generate_fixed_arrangements(
    n: int, seed: int = DEFAULT_ARRANGEMENT_SEED
) -> list[tuple[int, int, int, int]]:
    """Kumaran's fixed option orders: one ``random.shuffle(list(range(4)))`` per question.

    Entry ``i`` lists, for display positions 1..4, the source option index shown there.
    ``random.Random(seed)`` has exactly the state sequence of ``random.seed(seed)`` followed
    by the module-level ``random.shuffle`` calls the notebook makes.
    """
    rng = random.Random(seed)
    orders = []
    for _ in range(n):
        order = list(range(4))
        rng.shuffle(order)
        orders.append((order[0], order[1], order[2], order[3]))
    return orders


def apply_arrangement(
    choices: tuple[str, str, str, str],
    order: tuple[int, int, int, int],
    correct_index: int = 0,
) -> tuple[tuple[str, str, str, str], int]:
    """Present ``choices`` in ``order``; return them with the 1-based position of the answer."""
    arranged = (choices[order[0]], choices[order[1]], choices[order[2]], choices[order[3]])
    return arranged, order.index(correct_index) + 1


_POOL_FILE_KEY = {SIMPLEQA_CAL: "cal", SIMPLEQA_EVAL: "eval", SIMPLEQA_HOLDOUT: "full"}


def load_pool(
    registry: SplitRegistry,
    pool: str,
    source_dir: str | Path,
    *,
    files: Mapping[str, str],
    arrangement_seed: int = DEFAULT_ARRANGEMENT_SEED,
    results_root: str | Path | None = None,
    holdout_access: HoldoutAccess | None = None,
) -> list[McItem]:
    """Load one SimpleQA pool through the frozen locator map.

    ``files`` maps ``cal``, ``eval`` and ``full`` to file names (config ``data.simpleqa_mc``).
    Phase files have the correct answer in ``choice_1``; the holdout pool comes from the full
    file, where the correct option is ``choice_{gt_answer}``.
    """
    guard_pool_request(pool, results_root=results_root, access=holdout_access)
    if pool not in _POOL_FILE_KEY:
        raise SplitError(f"{pool!r} is not a SimpleQA pool")
    file_name = files[_POOL_FILE_KEY[pool]]
    path = Path(source_dir) / file_name
    record = registry.verify_input(input_name(file_name), path)
    arrangements = generate_fixed_arrangements(record["n_rows"], arrangement_seed)
    wanted = registry.key_set(pool)
    from_full_file = pool == SIMPLEQA_HOLDOUT
    ordered_keys: list[str] = []
    seen: set[str] = set()
    items: list[McItem] = []
    n_rows = 0
    for row, rec in iter_records(path, FULL_COLUMNS if from_full_file else PHASE_COLUMNS):
        n_rows = row + 1
        loc = locator(file_name, row)
        key = registry.key_for_locator(loc)
        if key is None:
            raise SplitError(f"row {loc} is not in the locator map")
        if key not in wanted or key in seen:
            continue  # not part of this pool (or a repeat of a key already kept): dropped here
        seen.add(key)
        ordered_keys.append(key)
        source_choices = (rec["choice_1"], rec["choice_2"], rec["choice_3"], rec["choice_4"])
        correct_index = parse_gt_index(rec["gt_answer"]) if from_full_file else 0
        if correct_index is None:  # sealing refuses such a file; this guards a file changed since
            raise SplitError(f"{file_name}: row {row} has a gt_answer that is not 1, 2, 3 or 4")
        order = arrangements[row]
        shown, position = apply_arrangement(source_choices, order, correct_index=correct_index)
        items.append(
            McItem(
                item_key=key, pool=pool, bank=BANK_SIMPLEQA, locator=loc, subject=None,
                question=rec["question"], choices=shown, gt_position=position,
                correct_text=source_choices[correct_index], arrangement=order, row=row,
            )
        )
    if n_rows != record["n_rows"]:
        raise SplitError(f"{file_name}: read {n_rows} rows, the locator map recorded {record['n_rows']}")
    if tuple(ordered_keys) != registry.keys(pool):
        raise SplitError(f"pool {pool!r}: loaded keys differ from the frozen key list")
    if not is_holdout_pool(pool):
        registry.assert_no_holdout(ordered_keys, context=pool)
    else:
        assert holdout_access is not None
        holdout_access.consume(pool, ordered_keys)
    return items
