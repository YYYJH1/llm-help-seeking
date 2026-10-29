"""Durable writes for the files that decide whether a request may be sent: the ledger journal, the index of successes, the state files, the manifest.

Every append is written, flushed and ``fsync``-ed before the call returns (the attempt record is on disk before the request leaves; the result record before the next step). Every
whole-file write goes to a temporary name, is ``fsync``-ed and replaces the real file with ``os.replace``, which is atomic, and the directory is ``fsync``-ed after it. All the ``fsync``
calls go through ``fsync_fd`` and ``fsync_dir`` so that a test can observe their order.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from nmi import paths


def fsync_fd(fd: int) -> None:
    os.fsync(fd)


def fsync_dir(directory: Path) -> None:
    """Make a rename or a new file in ``directory`` durable (best effort: some file systems refuse to open a directory)."""
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        fsync_fd(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def dump_line(record: dict[str, Any]) -> str:
    """One line of ASCII (non-ASCII characters, and a lone surrogate, are escaped) with sorted keys."""
    return json.dumps(record, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)


def write_json_atomic(path: Path, obj: Any) -> Path:
    """Write ``obj`` as JSON to ``path`` through a temporary file that atomically replaces the target."""
    target = paths.assert_off_root_partition(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=str, allow_nan=False)
        fh.write("\n")
        fh.flush()
        fsync_fd(fh.fileno())
    os.replace(tmp, target)
    fsync_dir(target.parent)
    return target


def write_text_atomic(path: Path, text: str) -> Path:
    """The same for a text file (the versioned key files of a plan)."""
    target = paths.assert_off_root_partition(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    with open(tmp, "w", encoding="ascii") as fh:
        fh.write(text)
        fh.flush()
        fsync_fd(fh.fileno())
    os.replace(tmp, target)
    fsync_dir(target.parent)
    return target


class DurableAppender:
    """An append-only JSONL file whose every record is on disk when ``append`` returns. A last line without its newline is a write cut off by a crash (it was not on disk when the
    program went on, and ``nmi.api_exp.journal`` does not read it): it is cut off the file when the file is opened, so the next record starts on its own line."""

    def __init__(self, path: Path) -> None:
        self.path = paths.assert_off_root_partition(path)
        self._fh: Any = None

    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        existed = self.path.is_file()
        if existed and self.path.stat().st_size > 0:
            data = self.path.read_bytes()
            if not data.endswith(b"\n"):
                with open(self.path, "r+b") as fh:
                    fh.truncate(data.rfind(b"\n") + 1)
                    fh.flush()
                    fsync_fd(fh.fileno())
        self._fh = open(self.path, "a", encoding="ascii")  # noqa: SIM115
        if not existed:
            fsync_dir(self.path.parent)

    def append(self, record: dict[str, Any]) -> None:
        if self._fh is None:
            raise RuntimeError(f"{self.path.name} is not open")
        self._fh.write(dump_line(record) + "\n")
        self._fh.flush()
        fsync_fd(self._fh.fileno())

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
