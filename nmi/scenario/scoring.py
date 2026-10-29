"""Scoring of a short answer against the possible answers of a question, for the expert, P-answer and the self-answers of P-decide: one implementation, one short-answer constraint.

Primary score. The reply is cleaned (``adapter.clean``: think blocks and end-of-turn tokens dropped, stripped), its FIRST LINE is taken and normalised (``nmi.data.triviaqa.normalize``: NFKC, lower case,
ASCII punctuation and the articles a/an/the removed, white space collapsed); it is correct when some normalised alias, non-empty, occurs in it as a SUBSTRING. An empty first line is wrong.

Three sensitivity variants (each is recomputed in full downstream, with the same denominator and pairing):

* ``full_alias``: the normalised first line EQUALS a normalised alias (no substring match);
* ``negation``: an answer that contains a negation word is wrong (it is not removed: the denominator does not change). The words are ``NEGATIONS``, matched as whole words of the normalised first line, after
  the matched alias itself has been taken out of it, so that an answer such as the alias "No Doubt" is not made wrong by the "No" that belongs to the alias (the longest matching alias is the one taken out);
* ``short_boundary``: an alias of at most 2 characters (after normalisation) must match at word boundaries; longer aliases still match as substrings (a pilot sample of 200 items had 9 questions with such an
  alias, which a substring match over-counts).

Agreement with the normalisation of the official TriviaQA evaluation script has NOT been verified (the script was not available offline). The primary score is the substring rule on the normalisation
above; any difference from the official one has to be reported, not assumed away.

Nothing here keeps, prints or logs a reply or an answer.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from nmi.data.triviaqa import normalize
from nmi.scenario.adapter import clean

SHORT_ALIAS_MAX = 2
# whole words of the normalised first line (the normalisation removes the apostrophe: "isn't" is "isnt")
NEGATIONS = frozenset({
    "no", "not", "never", "none", "nobody", "nothing", "neither", "nor", "cannot", "cant", "dont", "doesnt", "didnt", "isnt", "arent", "wasnt", "werent", "wont", "wouldnt", "shouldnt", "couldnt",
    "hasnt", "havent", "hadnt", "mustnt", "neednt", "aint",
})


@dataclass(frozen=True)
class Scores:
    """The four scores of one reply."""

    primary: bool
    full_alias: bool
    negation: bool
    short_boundary: bool

    def as_dict(self) -> dict[str, bool]:
        return {"primary": self.primary, "full_alias": self.full_alias, "negation": self.negation, "short_boundary": self.short_boundary}


WRONG = Scores(False, False, False, False)


def first_line(reply: str) -> str:
    """The first line of the cleaned reply ('' for an empty reply)."""
    cleaned = clean(reply)
    return cleaned.split("\n")[0] if cleaned else ""


def prepare(answers: Iterable[str]) -> tuple[str, ...]:
    """The normalised, non-empty aliases, each once, in order of first appearance."""
    return tuple(dict.fromkeys(a for a in (normalize(x) for x in answers) if a))


def _boundary(alias: str, line: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", line) is not None


def has_negation(line: str, matched: Sequence[str]) -> bool:
    """A negation word in the normalised first line outside the longest matched alias."""
    remainder = line
    if matched:
        remainder = line.replace(max(matched, key=len), " ")
    return any(word in NEGATIONS for word in remainder.split())


def score_line(line: str, aliases: Sequence[str]) -> Scores:
    """The four scores of an already normalised first line against normalised aliases (``prepare``)."""
    if not line or not aliases:
        return WRONG
    contained = [a for a in aliases if a in line]
    primary = bool(contained)
    full = any(a == line for a in aliases)
    negation = primary and not has_negation(line, contained)
    short = any(_boundary(a, line) if len(a) <= SHORT_ALIAS_MAX else a in line for a in aliases)
    return Scores(primary=primary, full_alias=full, negation=negation, short_boundary=short)


def score(reply: str, answers: Iterable[str]) -> Scores:
    """The scores of a reply against the possible answers of the question (raw strings; they are normalised here)."""
    return score_line(normalize(first_line(reply)), prepare(answers))
