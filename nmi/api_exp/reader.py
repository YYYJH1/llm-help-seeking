"""The reading of a reply.

Take the content of the reply and strip white space from both ends. If the first character is 1, 2, 3 or 4 the model answered itself; if it is 5 the model asked for help; anything else
(including an empty reply) is a format failure. Only the first character is read: "5." is a request for help, "3) Paris" is an answer, "**5**" and "Answer: 3" are format failures.
Only the ASCII digits count (a full-width or Arabic-Indic digit is a format failure). A reply is never sent again because of its content.

The denominators (every trial, including format failures and requests that never succeeded) are applied in the analysis of the study (not part of this release); this module reads one reply.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

ANSWER = "answer"
ASK = "ask"
FORMAT_FAILURE = "format_failure"
DECISION_VALUES = (ANSWER, ASK, FORMAT_FAILURE)
VALID_DECISION_VALUES = (ANSWER, ASK)
_ANSWER_DIGITS = "1234"
_ASK_DIGIT = "5"
ANSWER_DIGITS = _ANSWER_DIGITS  # the configuration states the reading rule (block ``reading``) and the loader checks it against these
ASK_DIGIT = _ASK_DIGIT


@dataclass(frozen=True)
class Reading:
    decision: str  # answer, ask or format_failure
    first_char: str | None  # the character that was read (None for an empty reply)
    choice: int | None  # 1-4 for an answer, 5 for a request for help, None for a format failure

    @property
    def valid(self) -> bool:
        return self.decision in VALID_DECISION_VALUES


def read_reply(content: Any) -> Reading:
    """The decision in the content of a reply. A content that is not a string (None, a list, a number) reads as empty."""
    text = content.strip() if isinstance(content, str) else ""
    if not text:
        return Reading(FORMAT_FAILURE, None, None)
    first = text[0]
    if first in _ANSWER_DIGITS:
        return Reading(ANSWER, first, int(first))
    if first == _ASK_DIGIT:
        return Reading(ASK, first, 5)
    return Reading(FORMAT_FAILURE, first, None)
