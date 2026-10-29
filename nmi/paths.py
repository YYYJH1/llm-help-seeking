"""Canonical locations for every input, output and temporary file.

Layout (config key ``paths`` in configs/v6_confirm.yaml):

    results/<stage>/<YYYYMMDD_HHMMSS>_<tag>/   one run directory, see nmi.provenance
    results/seal_log.jsonl                     sealing log (written by the sealing step of the study)
    results/holdout_ledger.jsonl               consumption ledger (nmi.data.holdout)
    results/HOLDOUT_UNSEALED_BY_AUTHOR         marker file created by hand to unseal the hold-out
    data/splits/<version>/                     key lists and locator map
    tmp/<YYYYMMDD>-<purpose>/                  scratch files (the root can be set with NMI_TMP_ROOT)

``assert_off_root_partition`` refuses every directory under a prefix listed in ``ROOT_PARTITION_PREFIXES``
(none by default).
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

CONFIG_DIR = REPO_ROOT / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "v6_confirm.yaml"
DEFAULT_PROMPTS = CONFIG_DIR / "prompts" / "ask_v1_1.yaml"
# The pre-registration and the protocol documents of the study are not part of this release. Where a reader supplies them under protocol/, their md5 values are
# recorded in the run manifests and checked as before; where a file is absent, the check prints one line and is skipped (nmi.provenance.note_skipped).
PROTOCOL_DIR = REPO_ROOT / "protocol"
PREREG = PROTOCOL_DIR / "preregistration_v1.md"

RESULTS_ROOT = REPO_ROOT / "results"
SPLITS_ROOT = REPO_ROOT / "data" / "splits"
DEFAULT_SPLIT_VERSION = "v1"

HOLDOUT_LEDGER_NAME = "holdout_ledger.jsonl"
HOLDOUT_MARKER_NAME = "HOLDOUT_UNSEALED_BY_AUTHOR"

TMP_ROOT = Path(os.environ.get("NMI_TMP_ROOT", REPO_ROOT / "tmp"))
# Local Hugging Face hub cache that holds the pinned dataset snapshots.
HF_HUB = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface") / "hub")

# Directory prefixes in which writing is refused (none by default).
ROOT_PARTITION_PREFIXES: tuple[str, ...] = ()

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")


class PathPolicyError(ValueError):
    """A path or name violates the storage or naming rules."""


def validate_name(kind: str, value: str) -> str:
    """Stage names, run tags and temp purposes become directory names, so keep them plain."""
    if not isinstance(value, str) or not _NAME_RE.fullmatch(value) or ".." in value:
        raise PathPolicyError(
            f"{kind} {value!r} must match [A-Za-z0-9][A-Za-z0-9_.-]* and contain no '..'"
        )
    return value


def assert_off_root_partition(path: str | Path) -> Path:
    """Return the resolved path, or raise if it lies under one of ``ROOT_PARTITION_PREFIXES``."""
    resolved = Path(path).resolve()
    text = str(resolved)
    for prefix in ROOT_PARTITION_PREFIXES:
        if text == prefix or text.startswith(prefix + "/"):
            raise PathPolicyError(
                f"{text} is under {prefix}, where writing is refused"
            )
    return resolved


def _anchored(path: str | Path) -> Path:
    """A relative path (the config writes ``results``) is taken relative to the repository,
    never to the current directory."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def results_root(root: str | Path | None = None) -> Path:
    return _anchored(root) if root is not None else RESULTS_ROOT


def run_dir_name(tag: str, when: datetime) -> str:
    """``<YYYYMMDD_HHMMSS>_<tag>``; the stamp is local time, the manifest carries the offset."""
    return f"{when.strftime('%Y%m%d_%H%M%S')}_{validate_name('tag', tag)}"


def split_dir(version: str = DEFAULT_SPLIT_VERSION, *, root: str | Path | None = None) -> Path:
    return (_anchored(root) if root is not None else SPLITS_ROOT) / validate_name("split version", version)


def holdout_ledger_path(root: str | Path | None = None) -> Path:
    return results_root(root) / HOLDOUT_LEDGER_NAME


def holdout_marker_path(root: str | Path | None = None) -> Path:
    return results_root(root) / HOLDOUT_MARKER_NAME
