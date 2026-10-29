"""Holdout guard and consumption ledger.

Rules implemented here:

* Content of a holdout pool is available only after the marker file
  ``results/HOLDOUT_UNSEALED_BY_AUTHOR`` has been created by hand. Without it every
  request for a holdout pool fails. Every read entry point calls ``guard_pool_request``;
  there is no other check.
* A holdout request also needs a ``HoldoutAccess``: a confirmatory batch whose id was
  registered in the ledger BEFORE the marker was created.
* The ledger ``results/holdout_ledger.jsonl`` is keyed by item key and append-only. A key
  that a batch consumed is development data for ever, whatever the protocol version. A
  second batch that asks for the same key fails. Re-reading inside the batch that consumed
  it is allowed, because one confirmatory batch spans several models and arms.

Nothing here logs or prints item content.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from nmi.data.registry import SplitRegistry, is_holdout_pool, physical_pool
from nmi.paths import holdout_ledger_path, holdout_marker_path

_BATCH_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")


class HoldoutSealedError(PermissionError):
    """Holdout content was requested while the pool is sealed (or without a registered batch)."""


class HoldoutLedgerError(RuntimeError):
    """The consumption ledger refuses the operation."""


def guard_pool_request(
    pool: str,
    *,
    results_root: str | Path | None = None,
    access: object | None = None,
) -> None:
    """The one guard every read entry point calls first.

    Non-holdout pools pass. A holdout pool needs the marker file and a ``HoldoutAccess``.
    """
    if not is_holdout_pool(pool):
        return
    marker = holdout_marker_path(results_root)
    if not marker.is_file():
        raise HoldoutSealedError(
            f"pool {pool!r} is sealed: marker {marker} does not exist "
            "(the marker is created by hand at the start of the confirmatory phase)"
        )
    if not isinstance(access, HoldoutAccess):
        raise HoldoutSealedError(
            f"pool {pool!r} needs a HoldoutAccess for a registered confirmatory batch"
        )


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class HoldoutLedger:
    """Append-only JSONL ledger. All mutations hold an exclusive lock on ``<path>.lock``."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @contextmanager
    def _locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_name(self.path.name + ".lock")
        with open(lock_path, "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _read(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        records = []
        with open(self.path, encoding="utf-8") as fh:
            for number, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise HoldoutLedgerError(f"{self.path}:{number} is not valid JSON") from error
        return records

    def _append(self, records: Sequence[Mapping[str, Any]]) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    # ---- batches -----------------------------------------------------------------------
    def batches(self) -> dict[str, dict[str, Any]]:
        return {r["batch_id"]: r for r in self._read() if r.get("kind") == "batch"}

    def register_batch(
        self,
        batch_id: str,
        *,
        protocol_md5: str,
        description: str = "",
        now_epoch: float | None = None,
        marker_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Register a confirmatory batch id. Must happen before the unseal marker exists
        (checked here when ``marker_path`` is given, and again by ``consume``)."""
        if not _BATCH_ID_RE.fullmatch(batch_id):
            raise HoldoutLedgerError(f"batch id {batch_id!r} must match [A-Za-z0-9][A-Za-z0-9_.-]*")
        if marker_path is not None and Path(marker_path).exists():
            raise HoldoutLedgerError(
                f"cannot register batch {batch_id!r}: the unseal marker already exists, "
                "batch ids are registered before unsealing"
            )
        record = {
            "kind": "batch",
            "batch_id": batch_id,
            "protocol_md5": protocol_md5,
            "description": description,
            "registered_epoch": time.time() if now_epoch is None else now_epoch,
            "registered_utc": _now_iso(),
        }
        with self._locked():
            if batch_id in self.batches():
                raise HoldoutLedgerError(f"batch {batch_id!r} is already registered")
            self._append([record])
        return record

    # ---- consumption ----------------------------------------------------------------------
    def consumed(self) -> dict[str, dict[str, Any]]:
        """Item key -> the ledger record of the batch that consumed it."""
        return {r["item_key"]: r for r in self._read() if r.get("kind") == "consumed"}

    def consume(
        self,
        keys: Iterable[str],
        *,
        batch_id: str,
        pool: str,
        models: Sequence[str] = (),
        arms: Sequence[str] = (),
        marker_path: str | Path | None = None,
        allowed_keys: Iterable[str] | None = None,
    ) -> int:
        """Record that ``batch_id`` uses these holdout keys; returns how many are new.

        Fails, writing nothing, if the batch is not registered, was registered after the
        marker was created, a key is outside ``allowed_keys`` (the pool's frozen list), or
        a key was consumed by another batch. Keys this batch already consumed are ignored.
        """
        key_list = list(keys)
        if not key_list:
            raise HoldoutLedgerError("no keys to consume")
        if len(set(key_list)) != len(key_list):
            raise HoldoutLedgerError("duplicate keys in one consumption request")
        if allowed_keys is not None:
            outside = set(key_list) - set(allowed_keys)
            if outside:
                raise HoldoutLedgerError(f"{len(outside)} keys are not in pool {pool!r}")
        with self._locked():
            batch = self.batches().get(batch_id)
            if batch is None:
                raise HoldoutLedgerError(f"batch {batch_id!r} was not registered before unsealing")
            if marker_path is not None:
                marker = Path(marker_path)
                if not marker.is_file():
                    raise HoldoutSealedError(f"unseal marker {marker} does not exist")
                if batch["registered_epoch"] >= marker.stat().st_mtime:
                    raise HoldoutLedgerError(
                        f"batch {batch_id!r} was registered after the unseal marker was created"
                    )
            already = self.consumed()
            by_other = [k for k in key_list if k in already and already[k]["batch_id"] != batch_id]
            if by_other:
                raise HoldoutLedgerError(
                    f"{len(by_other)} keys were already consumed by another batch; "
                    "a consumed key is development data for ever"
                )
            fresh = [k for k in key_list if k not in already]
            stamp = _now_iso()
            self._append(
                [
                    {
                        "kind": "consumed",
                        "item_key": k,
                        "pool": physical_pool(pool),
                        "batch_id": batch_id,
                        "models": list(models),
                        "arms": list(arms),
                        "utc": stamp,
                    }
                    for k in fresh
                ]
            )
            return len(fresh)


class HoldoutAccess:
    """Capability for reading holdout pools inside one registered confirmatory batch."""

    def __init__(
        self,
        *,
        registry: SplitRegistry,
        ledger: HoldoutLedger,
        batch_id: str,
        models: Sequence[str],
        arms: Sequence[str],
        results_root: str | Path | None = None,
    ) -> None:
        self.registry = registry
        self.ledger = ledger
        self.batch_id = batch_id
        self.models = tuple(models)
        self.arms = tuple(arms)
        self.results_root = results_root

    def consume(self, pool: str, keys: Iterable[str]) -> int:
        """Called by a loader after it has read and verified the pool."""
        return self.ledger.consume(
            keys,
            batch_id=self.batch_id,
            pool=pool,
            models=self.models,
            arms=self.arms,
            marker_path=holdout_marker_path(self.results_root),
            allowed_keys=self.registry.key_set(pool),
        )


def open_holdout(
    *,
    registry: SplitRegistry,
    batch_id: str,
    models: Sequence[str],
    arms: Sequence[str],
    results_root: str | Path | None = None,
) -> HoldoutAccess:
    """Create a ``HoldoutAccess``; fails now if the marker is missing or the batch unregistered."""
    marker = holdout_marker_path(results_root)
    if not marker.is_file():
        raise HoldoutSealedError(f"holdout is sealed: marker {marker} does not exist")
    ledger = HoldoutLedger(holdout_ledger_path(results_root))
    if batch_id not in ledger.batches():
        raise HoldoutLedgerError(f"batch {batch_id!r} was not registered before unsealing")
    return HoldoutAccess(
        registry=registry, ledger=ledger, batch_id=batch_id, models=models, arms=arms,
        results_root=results_root,
    )
