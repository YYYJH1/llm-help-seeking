"""Seed registry (config section ``seed_namespace``).

Every random stream is built from ``numpy.random.SeedSequence([stage, cell, rep, 0xA0])``.
The key always has exactly four words; a shorter or longer key, or a wrong last word, is an
error, and the set of full keys of a batch must be unique (``assert_unique``).

Registered values (config ``seed_namespace``):

    stage 1  DES                       rep = 42..46, cell = grid condition number
    stage 2  real-loop arrivals        rep = 42..49, cell = run condition number
    stage 3  steering random directions rep = 9101..9120
    stage 9  development and validation cells: development_pilot 1 (rep 9001..9006),
             hf_cross_check 2 (rep 9001), regime_validation 3, k4_bootstrap 4,
             psychometric_bootstrap 5, ladder 6, gate_decidability 7
    run order of the eight real-loop runs: seed 1729 (see ``RUN_ORDER_CELL``)

``cell`` for the stage-1 and stage-2 families is the condition number chosen by the caller;
this module checks the stage and the registered rep values, and the caller passes every key it
uses to ``assert_unique``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

SALT = 0xA0
KEY_LENGTH = 4
_WORD_LIMIT = 2**32

# The run-order draw is a stage-2 (real loop) quantity. Only the value 1729 is registered for it;
# cell 0 is reserved for it, run conditions start at 1 in the caller's numbering
# (a collision is impossible anyway because 1729 is not an arrival seed).
RUN_ORDER_CELL = 0
# One condition covers the 20 random steering directions (stage 3), reps are the seeds.
STEERING_RANDOM_CELL = 0
# Item draw of the HF/vLLM cross-check: seed 9001 in its own stage-9 cell. The
# development pilot also uses rep 9001, in cell 1, so the full keys differ.
HF_CROSS_CHECK_SEED = 9001
# Submission order of the second pass of the C4 repeated readout: the shuffle is drawn from stage 9, cell 9, rep 9001. The cell number is
# fixed here and is not in the ``cells_stage_9`` registry of the configurations (the C4 entry then runs with every configuration that has the stage-9 stage);
# ``SeedNamespace.c4_repeat_order_key`` refuses a configuration that has registered cell 9 for something else.
C4_REPEAT_ORDER_CELL = 9
C4_REPEAT_ORDER_SEED = 9001

SeedKey = tuple[int, int, int, int]


class SeedKeyError(ValueError):
    """A seed key is malformed, unregistered or duplicated."""


def _word(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise SeedKeyError(f"seed key {name} must be an integer, got {value!r}")
    value = int(value)
    if not 0 <= value < _WORD_LIMIT:
        raise SeedKeyError(f"seed key {name} must be in [0, 2^32), got {value}")
    return value


def make_key(stage: int, cell: int, rep: int) -> SeedKey:
    """The four-word key ``(stage, cell, rep, 0xA0)``."""
    return (_word("stage", stage), _word("cell", cell), _word("rep", rep), SALT)


def check_key(key: Sequence[int]) -> SeedKey:
    """Validate a full key: exactly four words, integer words in range, last word 0xA0."""
    words = tuple(key)
    if len(words) != KEY_LENGTH:
        raise SeedKeyError(f"seed key must have {KEY_LENGTH} words, got {len(words)}: {list(words)}")
    checked = tuple(_word(f"word {i}", w) for i, w in enumerate(words))
    if checked[-1] != SALT:
        raise SeedKeyError(f"last word of a seed key must be {SALT:#x}, got {checked[-1]:#x}")
    return checked  # type: ignore[return-value]


def seed_sequence(key: Sequence[int]) -> np.random.SeedSequence:
    return np.random.SeedSequence(list(check_key(key)))


def rng(key: Sequence[int]) -> np.random.Generator:
    """A reproducible ``Generator`` for the key."""
    return np.random.default_rng(seed_sequence(key))


def assert_unique(keys: Iterable[Sequence[int]]) -> int:
    """Raise if any full key occurs twice; returns the number of keys checked."""
    seen: set[SeedKey] = set()
    count = 0
    for key in keys:
        full = check_key(key)
        if full in seen:
            raise SeedKeyError(f"duplicate seed key {list(full)}")
        seen.add(full)
        count += 1
    return count


def _expand(spec: Any, name: str) -> tuple[int, ...]:
    """A registered seed family: a list of seeds, or ``[first, last]`` for an inclusive range."""
    values = tuple(int(v) for v in spec)
    if name in _RANGE_FAMILIES:
        if len(values) != 2 or values[0] > values[1]:
            raise SeedKeyError(f"seed_namespace.{name} must be [first, last] with first <= last")
        return tuple(range(values[0], values[1] + 1))
    return values


_RANGE_FAMILIES = frozenset({"steering_random_directions", "development_pilot"})


@dataclass(frozen=True)
class SeedNamespace:
    """The registered seed families, read from config ``seed_namespace``."""

    stages: Mapping[str, int]
    stage9_cells: Mapping[str, int]
    families: Mapping[str, tuple[int, ...]]
    run_order: int

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "SeedNamespace":
        ns = cfg["seed_namespace"]
        families = {
            name: _expand(ns[name], name)
            for name in ("des", "loop_arrivals", "steering_random_directions", "development_pilot")
        }
        return cls(
            stages=dict(ns["stages"]),
            stage9_cells=dict(ns["cells_stage_9"]),
            families=families,
            run_order=int(ns["run_order"]),
        )

    # ---- names -> numbers --------------------------------------------------------------
    def stage_id(self, name: str) -> int:
        try:
            return int(self.stages[name])
        except KeyError as error:
            raise SeedKeyError(f"unknown seed stage {name!r}") from error

    def stage9_cell(self, name: str) -> int:
        try:
            return int(self.stage9_cells[name])
        except KeyError as error:
            raise SeedKeyError(f"unknown stage-9 cell {name!r}") from error

    def reps(self, family: str) -> tuple[int, ...]:
        try:
            return self.families[family]
        except KeyError as error:
            raise SeedKeyError(f"unknown seed family {family!r}") from error

    def _registered(self, family: str, rep: int) -> int:
        if rep not in self.reps(family):
            raise SeedKeyError(f"seed {rep} is not registered for family {family!r}")
        return rep

    # ---- keys --------------------------------------------------------------------------
    def des_key(self, cell: int, rep: int) -> SeedKey:
        return make_key(self.stage_id("des"), cell, self._registered("des", rep))

    def loop_arrival_key(self, cell: int, rep: int) -> SeedKey:
        return make_key(self.stage_id("loop_arrivals"), cell, self._registered("loop_arrivals", rep))

    def steering_random_key(self, rep: int, cell: int = STEERING_RANDOM_CELL) -> SeedKey:
        return make_key(
            self.stage_id("steering_random"), cell, self._registered("steering_random_directions", rep)
        )

    def development_pilot_key(self, rep: int) -> SeedKey:
        return make_key(
            self.stage_id("development_and_validation"),
            self.stage9_cell("development_pilot"),
            self._registered("development_pilot", rep),
        )

    def hf_cross_check_key(self) -> SeedKey:
        """Key of the cross-check item draw: stage 9, its own cell, rep 9001."""
        return make_key(
            self.stage_id("development_and_validation"), self.stage9_cell("hf_cross_check"),
            HF_CROSS_CHECK_SEED,
        )

    def stage9_key(self, cell_name: str, rep: int) -> SeedKey:
        """Key for a stage-9 cell without a fixed rep list (validation replicates, bootstraps, ladder)."""
        return make_key(self.stage_id("development_and_validation"), self.stage9_cell(cell_name), rep)

    def c4_repeat_order_key(self) -> SeedKey:
        """Key of the shuffle that sets the submission order of the second pass of the C4 repeated readout: stage 9, cell 9 (``C4_REPEAT_ORDER_CELL``), rep 9001.

        The cell number is fixed in this module, not read from the configuration. A configuration that registers cell 9 for another stage-9 draw would share the key with that
        draw, so it is refused."""
        taken = sorted(name for name, cell in self.stage9_cells.items() if int(cell) == C4_REPEAT_ORDER_CELL)
        if taken:
            raise SeedKeyError(f"stage-9 cell {C4_REPEAT_ORDER_CELL} is registered for {taken} in this configuration; the C4 repeated readout draws its order there")
        return make_key(self.stage_id("development_and_validation"), C4_REPEAT_ORDER_CELL, C4_REPEAT_ORDER_SEED)

    def run_order_key(self) -> SeedKey:
        return make_key(self.stage_id("loop_arrivals"), RUN_ORDER_CELL, self.run_order)

    def registered_keys(
        self,
        *,
        des_cells: Iterable[int] = (),
        loop_cells: Iterable[int] = (),
    ) -> list[SeedKey]:
        """Every key of the fixed families, plus the DES and loop families for the given cells."""
        keys: list[SeedKey] = []
        for cell in des_cells:
            keys.extend(self.des_key(cell, rep) for rep in self.reps("des"))
        for cell in loop_cells:
            keys.extend(self.loop_arrival_key(cell, rep) for rep in self.reps("loop_arrivals"))
        keys.extend(self.steering_random_key(rep) for rep in self.reps("steering_random_directions"))
        keys.extend(self.development_pilot_key(rep) for rep in self.reps("development_pilot"))
        keys.append(self.hf_cross_check_key())
        keys.append(self.run_order_key())
        return keys
