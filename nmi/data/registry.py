"""Frozen split registry: the read side of the sealed key lists.

The sealing script of the study (not part of this release) is the only program that generates keys and assigns items to
pools. It writes three kinds of artifacts into one split directory (default
``data/splits/v1``):

    pools.json        pool names, sizes, md5/sha256 of each key list, rule texts
    keys/<pool>.keys  one canonical item key per line, in pool order
    locator_map.json  input file hashes and the map  source locator -> canonical key

This module loads and verifies them. Loaders (``simpleqa_mc``, ``mmlu_redux``) locate a
row through the map, filter by the frozen key lists straight away, and never compute a key
themselves. The module has no reference to ``nmi.data.keys.item_key`` on purpose.

Source locators:  ``simpleqa_mc/<file>#<row>``  and  ``mmlu_redux/<subject>#<row>``
(row = 0-based data row, header excluded).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from nmi.data.keys import ITEM_KEY_RULE, NORMALIZE_RULE, keylist_digests
from nmi.paths import DEFAULT_SPLIT_VERSION, split_dir
from nmi.provenance import bytes_md5, file_sha256

FORMAT = 1
_KEY_RE = re.compile(r"^[0-9a-f]{64}$")

# pool names
SIMPLEQA_CAL = "simpleqa_mc_cal"
SIMPLEQA_EVAL = "simpleqa_mc_eval"
SIMPLEQA_HOLDOUT = "simpleqa_mc_holdout"
MMLU_CAL = "mmlu_redux_cal"
MMLU_EVAL = "mmlu_redux_eval"
MMLU_HOLDOUT = "mmlu_redux_holdout"
MMLU_EVAL_SHUFFLED = "mmlu_redux_eval_shuffled"  # derived view of MMLU_EVAL, no key list of its own

BANK_SIMPLEQA = "simpleqa_mc"
BANK_MMLU = "mmlu_redux"

PHYSICAL_POOLS = (
    SIMPLEQA_CAL, SIMPLEQA_EVAL, SIMPLEQA_HOLDOUT, MMLU_CAL, MMLU_EVAL, MMLU_HOLDOUT,
)
DERIVED_POOLS = {MMLU_EVAL_SHUFFLED: MMLU_EVAL}
POOL_NAMES = frozenset(PHYSICAL_POOLS) | frozenset(DERIVED_POOLS)
# The two pools of the frozen split of the real scenario S1 (``nmi.data.triviaqa``, ``data/splits/s1_triviaqa_v1``). They are named here so that the holdout guard
# (``guard_pool_request`` asks ``is_holdout_pool``) and the consumption ledger (``HoldoutLedger.consume`` records ``physical_pool(pool)``) treat the sealed pool of the scenario as they treat
# the two holdout pools above. They are NOT part of ``POOL_NAMES`` or ``PHYSICAL_POOLS``: the configuration schema, the kill-test entries and ``SplitRegistry`` do not know them.
SCENARIO_DEV_POOL = "triviaqa_s1_dev"
SCENARIO_SEALED_POOL = "triviaqa_s1_sealed"
SCENARIO_POOLS = frozenset({SCENARIO_DEV_POOL, SCENARIO_SEALED_POOL})
HOLDOUT_POOLS = frozenset({SIMPLEQA_HOLDOUT, MMLU_HOLDOUT, SCENARIO_SEALED_POOL})
POOL_BANK = {
    SIMPLEQA_CAL: BANK_SIMPLEQA, SIMPLEQA_EVAL: BANK_SIMPLEQA, SIMPLEQA_HOLDOUT: BANK_SIMPLEQA,
    MMLU_CAL: BANK_MMLU, MMLU_EVAL: BANK_MMLU, MMLU_HOLDOUT: BANK_MMLU,
    MMLU_EVAL_SHUFFLED: BANK_MMLU,
}


class SplitError(RuntimeError):
    """The frozen split artifacts are missing, inconsistent or do not match the inputs."""


class InputHashMismatchError(SplitError):
    """An input file differs from the one the locator map was bound to."""


class HoldoutLeakError(SplitError):
    """A holdout key passed a filter that must only let calibration and evaluation through."""


def is_holdout_pool(pool: str) -> bool:
    return DERIVED_POOLS.get(pool, pool) in HOLDOUT_POOLS


def physical_pool(pool: str) -> str:
    """Map a derived view (the shuffled copy) to the pool that holds its keys."""
    if pool not in POOL_NAMES and pool not in SCENARIO_POOLS:
        raise SplitError(f"unknown pool {pool!r}")
    return DERIVED_POOLS.get(pool, pool)


@dataclass(frozen=True)
class McItem:
    """One four-option multiple-choice item as presented to a model."""

    item_key: str
    pool: str
    bank: str
    locator: str
    subject: str | None
    question: str
    choices: tuple[str, str, str, str]  # in presented order
    gt_position: int  # 1..4, position of the correct option in ``choices``
    correct_text: str
    arrangement: tuple[int, int, int, int] | None  # order[k] = source index shown at k+1
    row: int  # 0-based row in the source file


@dataclass(frozen=True)
class PoolMeta:
    name: str
    n: int
    md5: str
    sha256: str
    file: str


class SplitRegistry:
    """Verified view of one frozen split directory."""

    def __init__(
        self,
        root: Path,
        pools_doc: Mapping,
        keys: Mapping[str, tuple[str, ...]],
        locator_doc: Mapping,
    ) -> None:
        self.root = root
        self._doc = pools_doc
        self._keys = dict(keys)
        self._key_sets = {name: frozenset(values) for name, values in self._keys.items()}
        self._inputs: Mapping[str, Mapping] = locator_doc["inputs"]
        self._map: Mapping[str, str] = locator_doc["map"]
        self.version: str = pools_doc["version"]
        self.file_md5s: dict[str, str] = {}  # md5 of pools.json and locator_map.json (set by ``load``)
        self.meta: dict[str, PoolMeta] = {
            name: PoolMeta(name, m["n"], m["md5"], m["sha256"], m["file"])
            for name, m in pools_doc["pools"].items()
        }

    # ---- loading -------------------------------------------------------------------------
    @classmethod
    def load(cls, root: str | Path | None = None) -> "SplitRegistry":
        directory = Path(root) if root is not None else split_dir(DEFAULT_SPLIT_VERSION)
        pools_path = directory / "pools.json"
        map_path = directory / "locator_map.json"
        for needed in (pools_path, map_path):
            if not needed.is_file():
                raise SplitError(f"split artifact {needed} is missing; it is written by the sealing script of the study, which is not part of this release")
        pools_doc = json.loads(pools_path.read_text(encoding="utf-8"))
        locator_doc = json.loads(map_path.read_text(encoding="utf-8"))
        cls._check_docs(pools_doc, locator_doc)
        keys: dict[str, tuple[str, ...]] = {}
        for name, meta in pools_doc["pools"].items():
            path = directory / meta["file"]
            if not path.is_file():
                raise SplitError(f"key list {path} is missing")
            values = tuple(path.read_text(encoding="ascii").splitlines())
            if not all(_KEY_RE.fullmatch(v) for v in values):
                raise SplitError(f"key list of pool {name!r} contains a line that is not a sha256 key")
            md5, sha = keylist_digests(values)
            if md5 != meta["md5"] or sha != meta["sha256"]:
                raise SplitError(
                    f"key list of pool {name!r} does not match its recorded hash "
                    f"(file md5 {md5}, recorded {meta['md5']})"
                )
            if len(values) != meta["n"]:
                raise SplitError(f"pool {name!r} has {len(values)} keys, recorded {meta['n']}")
            keys[name] = values
        registry = cls(directory, pools_doc, keys, locator_doc)
        registry.assert_disjoint()
        registry.assert_map_matches_pools()
        registry.file_md5s = {
            "pools.json": bytes_md5(pools_path.read_bytes()),
            "locator_map.json": bytes_md5(map_path.read_bytes()),
        }
        return registry

    @staticmethod
    def _check_docs(pools_doc: Mapping, locator_doc: Mapping) -> None:
        for label, doc in (("pools.json", pools_doc), ("locator_map.json", locator_doc)):
            if doc.get("format") != FORMAT:
                raise SplitError(f"{label}: unsupported format {doc.get('format')!r}")
            if doc.get("normalize_rule") != NORMALIZE_RULE:
                raise SplitError(
                    f"{label}: normalize rule {doc.get('normalize_rule')!r} differs from the code "
                    f"({NORMALIZE_RULE!r}); the key lists were frozen under another rule"
                )
        if pools_doc.get("item_key_rule") != ITEM_KEY_RULE:
            raise SplitError("pools.json: item key rule differs from the code")
        if set(pools_doc["pools"]) != set(PHYSICAL_POOLS):
            raise SplitError(
                f"pools.json lists pools {sorted(pools_doc['pools'])}, expected {sorted(PHYSICAL_POOLS)}"
            )

    # ---- lookups -------------------------------------------------------------------------
    def keys(self, pool: str) -> tuple[str, ...]:
        """Frozen key list of a pool, in pool order (a derived view shares its parent's keys)."""
        return self._keys[physical_pool(pool)]

    def key_set(self, pool: str) -> frozenset[str]:
        return self._key_sets[physical_pool(pool)]

    def pool_md5(self, pool: str) -> str:
        return self.meta[physical_pool(pool)].md5

    def split_md5s(self, pools: Iterable[str] | None = None) -> dict[str, str]:
        """Pool -> md5 of the key list, for the manifest."""
        names = list(pools) if pools is not None else list(PHYSICAL_POOLS)
        return {name: self.pool_md5(name) for name in names}

    def manifest_md5s(self, pools: Iterable[str] | None = None) -> dict[str, str]:
        """``split.<name>`` -> md5 for a run manifest: the key lists asked for plus the two
        description files (the locator map is what binds the inputs)."""
        out = {f"split.{name}": md5 for name, md5 in self.split_md5s(pools).items()}
        out.update({f"split.{name}": md5 for name, md5 in self.file_md5s.items()})
        return out

    def holdout_keys(self) -> frozenset[str]:
        return self._key_sets[SIMPLEQA_HOLDOUT] | self._key_sets[MMLU_HOLDOUT]

    def key_for_locator(self, locator: str) -> str | None:
        """Frozen key of a source row, or None when the row is not part of any pool."""
        return self._map.get(locator)

    def locators_with_prefix(self, prefix: str) -> list[str]:
        return [loc for loc in self._map if loc.startswith(prefix)]

    # ---- input binding ---------------------------------------------------------------------
    def recorded_inputs(self, bank: str) -> dict[str, Mapping]:
        prefix = bank + "/"
        return {name: meta for name, meta in self._inputs.items() if name.startswith(prefix)}

    def verify_input(self, name: str, path: str | Path) -> Mapping:
        """Check that the file still has the hash the map was bound to; return its record."""
        record = self._inputs.get(name)
        if record is None:
            raise InputHashMismatchError(f"input {name!r} is not registered in the locator map")
        actual = file_sha256(path)
        if actual != record["sha256"]:
            raise InputHashMismatchError(
                f"input {name!r} has sha256 {actual[:16]}..., the locator map was bound to "
                f"{record['sha256'][:16]}...; the source file changed after sealing"
            )
        return record

    # ---- guards ----------------------------------------------------------------------------
    def assert_disjoint(self) -> None:
        """Split exclusivity: within one bank the cal, eval and holdout key sets share nothing."""
        for bank, cal, ev, ho in (
            (BANK_SIMPLEQA, SIMPLEQA_CAL, SIMPLEQA_EVAL, SIMPLEQA_HOLDOUT),
            (BANK_MMLU, MMLU_CAL, MMLU_EVAL, MMLU_HOLDOUT),
        ):
            trio = {cal: self._key_sets[cal], ev: self._key_sets[ev], ho: self._key_sets[ho]}
            names = list(trio)
            for i in range(3):
                for j in range(i + 1, 3):
                    shared = trio[names[i]] & trio[names[j]]
                    if shared:
                        raise SplitError(
                            f"{bank}: pools {names[i]} and {names[j]} share {len(shared)} keys"
                        )
        for name, values in self._keys.items():
            if len(set(values)) != len(values):
                raise SplitError(f"pool {name!r} contains duplicate keys")

    def assert_map_matches_pools(self) -> None:
        """Every key the locator map hands out belongs to a pool, and every pool key is mapped."""
        mapped = set(self._map.values())
        pooled = set().union(*self._key_sets.values())
        if mapped != pooled:
            raise SplitError(
                f"locator map and key lists disagree ({len(mapped - pooled)} mapped keys are in no pool, "
                f"{len(pooled - mapped)} pool keys are not mapped)"
            )

    def assert_no_holdout(self, keys: Iterable[str], *, context: str = "") -> None:
        """No holdout key may pass a calibration/evaluation filter."""
        leaked = self.holdout_keys().intersection(keys)
        if leaked:
            raise HoldoutLeakError(
                f"{len(leaked)} holdout keys passed the filter{' in ' + context if context else ''}"
            )

    def assert_fit_eval_disjoint(
        self, fit_keys: Iterable[str], eval_pools: Sequence[str] = (SIMPLEQA_EVAL, MMLU_EVAL)
    ) -> None:
        """Keys used to fit components never appear in an evaluation list."""
        fit = set(fit_keys)
        for pool in eval_pools:
            shared = fit & self.key_set(pool)
            if shared:
                raise SplitError(f"{len(shared)} fitting keys also appear in evaluation pool {pool!r}")


def load_items(
    pool: str,
    *,
    registry: SplitRegistry,
    sources: Mapping[str, str | Path],
    results_root: str | Path | None = None,
    holdout_access: object | None = None,
    arrangement_seed: int = 1042,
    simpleqa_files: Mapping[str, str] | None = None,
    keep_error_types: Sequence[str] = ("ok",),
) -> list[McItem]:
    """Single entry point for pool contents; routes to the bank loader.

    ``sources`` maps bank names to source directories (config ``data.*.source``).
    ``simpleqa_files`` maps ``cal``, ``eval`` and ``full`` to file names inside the SimpleQA
    directory (config ``data.simpleqa_mc.*_file``). Holdout pools additionally need a
    ``HoldoutAccess`` (see ``nmi.data.holdout``).
    """
    from nmi.data import mmlu_redux, simpleqa_mc
    from nmi.data.holdout import guard_pool_request

    guard_pool_request(pool, results_root=results_root, access=holdout_access)
    bank = POOL_BANK.get(pool)
    if bank == BANK_SIMPLEQA:
        if simpleqa_files is None:
            raise SplitError("simpleqa_files is required to load SimpleQA pools")
        return simpleqa_mc.load_pool(
            registry, pool, sources[BANK_SIMPLEQA],
            files=simpleqa_files, arrangement_seed=arrangement_seed,
            results_root=results_root, holdout_access=holdout_access,
        )
    if bank == BANK_MMLU:
        return mmlu_redux.load_pool(
            registry, pool, sources[BANK_MMLU],
            arrangement_seed=arrangement_seed, keep_error_types=keep_error_types,
            results_root=results_root, holdout_access=holdout_access,
        )
    raise SplitError(f"unknown pool {pool!r}")


def load_items_from_config(
    cfg: Mapping,
    pool: str,
    *,
    registry: SplitRegistry | None = None,
    results_root: str | Path | None = None,
    holdout_access: object | None = None,
) -> list[McItem]:
    """``load_items`` with every setting taken from the configuration (``cfg["data"]``).

    The SimpleQA arrangements use ``data.simpleqa_mc.arrangement_seed``; the shuffled copy of the
    MMLU-Redux evaluation pool uses ``data.mmlu_redux.shuffled_copy_seed``. ``registry`` defaults to
    the frozen split in ``data/splits/v1``.
    """
    data = cfg["data"]
    simple, mmlu = data["simpleqa_mc"], data["mmlu_redux"]
    bank = POOL_BANK.get(pool)
    seed = simple["arrangement_seed"] if bank == BANK_SIMPLEQA else mmlu["shuffled_copy_seed"]
    return load_items(
        pool,
        registry=registry if registry is not None else SplitRegistry.load(),
        sources={BANK_SIMPLEQA: simple["source"], BANK_MMLU: mmlu["source"]},
        results_root=results_root,
        holdout_access=holdout_access,
        arrangement_seed=seed,
        simpleqa_files={
            "cal": simple["calibration_file"], "eval": simple["evaluation_file"], "full": simple["full_file"],
        },
        keep_error_types=tuple(mmlu["keep_error_type"]),
    )
