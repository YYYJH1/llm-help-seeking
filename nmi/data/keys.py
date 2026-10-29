"""Canonical item keys.

    key = sha256( normalize(question) + '|' + normalize(correct_answer_text) )

``normalize`` is, in this order: Unicode NFKC, lower case, collapse every run of
whitespace to one space (this also strips both ends), then remove ONE trailing full stop
and strip whitespace again. Splits, subtraction of the holdout pools and the fit/eval
intersection check all judge by this key, never by file row.

Only the sealing script of the study (not part of this release) calls ``item_key`` to produce keys for pools. Loaders
look keys up in the locator map written by that script and never compute them.

Boundary case: "strip trailing period" is read as removing a single final "." (so "a.."
becomes "a."). ``data.normalize`` of the run configuration is checked against
``NORMALIZE_RULE`` when the config is loaded."""

from __future__ import annotations

import hashlib
import unicodedata

NORMALIZE_RULE = "NFKC, lowercase, collapse whitespace, strip trailing period"
ITEM_KEY_RULE = "sha256(normalize(question) + '|' + normalize(correct_answer_text))"


def normalize(text: str) -> str:
    """Normalise a question or answer string for key generation."""
    if not isinstance(text, str):
        raise TypeError(f"normalize expects str, got {type(text).__name__}")
    folded = unicodedata.normalize("NFKC", text).lower()
    collapsed = " ".join(folded.split())
    if collapsed.endswith("."):
        collapsed = collapsed[:-1].rstrip()
    return collapsed


def item_key(question: str, correct_answer_text: str) -> str:
    """Lower-case hex sha256 of ``normalize(question) | normalize(answer)`` (UTF-8)."""
    payload = normalize(question) + "|" + normalize(correct_answer_text)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def keylist_text(keys: "list[str] | tuple[str, ...]") -> str:
    """The exact text of a key-list file: one key per line, newline-terminated."""
    return "".join(f"{key}\n" for key in keys)


def keylist_digests(keys: "list[str] | tuple[str, ...]") -> tuple[str, str]:
    """(md5, sha256) of the key-list text."""
    data = keylist_text(keys).encode("ascii")
    return (
        hashlib.md5(data, usedforsecurity=False).hexdigest(),
        hashlib.sha256(data).hexdigest(),
    )
