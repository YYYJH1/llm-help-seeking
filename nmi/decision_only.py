"""Decision-only arm of the model extension: the constants of its protocol and the reading of a generated reply.

The arm asks a local model the same P-ask prompt as the digit readout, lets it generate at most four tokens greedily and reads the first character of what it wrote, as the closed-model
protocol does (``nmi.api_exp.reader``). Every number and rule that the protocol fixes is a constant of this file, and ``protocol_constants`` is what a run writes into its manifest.

The reading (``read_ids``) is a function of the generated token ids and of a tokenizer object (``decode``, ``all_special_ids`` and ``added_tokens_decoder``) and of nothing else: no engine, no stop set.

1. Decode. The whole generated sequence is decoded in one call with ``DECODE_KWARGS``: the tokenizer skips every token it marks as special (``skip_special_tokens=True``). Nothing is cut.
2. Strip. ``str.strip()`` (the whitespace of Python, the same call as the closed-model reader).
3. First character. "1".."4" (ASCII) is an answer of the model itself, "5" a request for help, anything else (also the empty text) a format failure. No digit is looked for further
   along. A full-width or Arabic-Indic digit is a format failure.

Special tokens are of two kinds, and the reading treats both alike. A terminating mark (an end-of-text id that the engine stops at) can only be the last token of a reply. A control mark that
does not stop the generation can stand anywhere, and the text after it takes part in the reading. Two facts about every reply are recorded and take no part in the reading:
``first_special_position`` (the index of the first special token, None if there is none) and ``non_special_after_special`` (whether a token that is not special comes after it; False when
there is no special token). The second fact says only that the model went on generating after a special token. It does not say whether the decision comes from before or after the mark
("5", mark, "2" decodes to "52": the decision is the one before the mark).

"The special tokens" are the tokens that the tokenizer marks as special, which are the ones its ``decode(skip_special_tokens=True)`` skips: the ids of ``tokenizer.all_special_ids`` and the added
tokens that are flagged ``special`` (``special_id_set``). The attribute alone is not enough. An audit of the tokenizer files of the 34 models that have them on disk (every id of the
vocabulary decoded alone, with and without skipping, to find what decode skips) found the set that decode skips equal to this union for all 34. The attribute alone is a strict subset for 15 of them
(attribute -> union): Qwen3.5-4B 9 -> 21, Phi-3.5-mini 3 -> 13 (it lacks ``<|end|>`` and ``<|assistant|>``, two eos ids of its generation config), Phi-4 3 -> 96, Mistral-7B-v0.3 3 -> 771,
GLM-4-9B-0414 2 -> 14, Granite-3.3-2B and 8B 1 -> 26, Granite-4.2-3B and 8B 3 -> 84, SmolLM2 3 -> 17, SmolLM3 1 -> 248, Gemma-3-4B 8 -> 10 (it lacks ``<eos>``), OLMo-2-1B and OLMo-3-7B 2 -> 8,
Phi-4-mini 1 -> 7. The other 19 have the two equal (the Qwen3 and Qwen2.5 models, Gemma-4, GLM-Edge, GLM-4-9B-chat-hf, the two DeepSeek models, the Mistral-Nemo and Ministral-3
tokenizers, ``MistralCommonBackend``, which list 1000 ids in the attribute and give no added tokens).
``attribute_special_ids`` is the attribute alone; a run records both.

A run is the development run of the protocol only if it read the whole development pool under the 54 conditions of the protocol, with the paraphrase a00, the standard
layout and no pilot subset (``PROTOCOL_*``). The entry decides that from the table itself and writes it as ``protocol_run`` into the summary; the other runs (another grid, layout or paraphrase, a subset
of the questions) are diagnostics.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

PROTOCOL_FILE = "protocol/decision_arm.md"  # path of the protocol text, relative to the repository root; its hash enters the manifest of every run

# ---------------------------------------------------------------------------------------
# the request
# ---------------------------------------------------------------------------------------

MAX_NEW_TOKENS = 4  # at most four new tokens; a whitespace token and a special token (also the one that ends the generation) are counted in the four
TEMPERATURE = 0.0  # greedy decoding
GENERATIONS_PER_TRIAL = 1  # one generation per (item, condition); the greedy decoding is deterministic and is not repeated, nor is a result chosen among repetitions
# What is NOT set, on purpose: no stop string, no stop token id of the entry, ignore_eos stays False. The tokens that end a generation are those that vLLM takes for the model by
# its own default (the end-of-text id of the tokenizer and the eos_token_id of generation_config.json); the entry only reports them (scripts/run_local_decisions.py).


def sampling_kwargs() -> dict[str, Any]:
    """The arguments of ``vllm.SamplingParams`` that the entry sets. Everything else is the default of vLLM; no probability is requested."""
    return {"max_tokens": MAX_NEW_TOKENS, "temperature": TEMPERATURE}


# ---------------------------------------------------------------------------------------
# the reading
# ---------------------------------------------------------------------------------------

ANSWER_CHARS = ("1", "2", "3", "4")  # the model answered itself, with this option
ASK_CHAR = "5"  # the model asked for help
# The one decode of the reading: the whole generated sequence, every token that the tokenizer marks as special skipped, the spaces not "cleaned up".
DECODE_KWARGS: Mapping[str, Any] = {"skip_special_tokens": True, "clean_up_tokenization_spaces": False}
MIN_VALID_FRACTION = 0.99  # a condition with a smaller share of valid trials is not testable; a flag of the summary, not a gate of the entry
SPECIAL_IDS_RECIPE = "md5 of the sorted special token ids as decimal numbers joined by ','"
SPECIAL_IDS_DEFINITION = "all_special_ids of the tokenizer and the added tokens it flags special (what decode(skip_special_tokens=True) skips)"
MAX_LISTED_SPECIAL_IDS = 256  # the manifest lists the ids themselves when the tokenizer has at most this many

ANSWER = "answer"
ASK = "ask"
FORMAT_FAILURE = "format_failure"
CATEGORIES = (ANSWER, ASK, FORMAT_FAILURE)

# ---------------------------------------------------------------------------------------
# the protocol run: what a run must be to count as the development run of the protocol, as opposed to a diagnostic or a smoke run
# ---------------------------------------------------------------------------------------

PROTOCOL_POOL = "mmlu_redux_cal"  # the development pool
PROTOCOL_POOL_ITEMS = 1000  # its size; the run must hold every one of them
PROTOCOL_GRIDS = ("psychometric",)
PROTOCOL_PARAPHRASES = ("a00",)  # the rule paragraph a00
PROTOCOL_LAYOUT = "standard"
PROTOCOL_FRAMING = "f2"  # the conditions block B1_f2
PROTOCOL_A_H = (0.60, 0.80, 0.95)  # with C0 and W: the 54 conditions of the probability readout
PROTOCOL_C0 = (0.0, 0.1, 0.2)
PROTOCOL_W = (0.0, 0.1, 0.2, 0.45, 0.8, 1.5)


def protocol_conditions() -> list[tuple[float, float, float]]:
    """The 54 conditions ``(a_H, c0, w)`` of the protocol, in the order of ``itertools.product`` (the order of the psychometric grid of ``run_decisions``)."""
    return [(a_H, c0, w) for a_H in PROTOCOL_A_H for c0 in PROTOCOL_C0 for w in PROTOCOL_W]


def protocol_constants() -> dict[str, Any]:
    """Every constant of the protocol as plain JSON values, as recorded in the manifest of a run."""
    return {
        "protocol_file": PROTOCOL_FILE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "generations_per_trial": GENERATIONS_PER_TRIAL,
        "sampling_kwargs": sampling_kwargs(),
        "stop_strings": [],
        "entry_stop_token_ids": [],
        "answer_chars": list(ANSWER_CHARS),
        "ask_char": ASK_CHAR,
        "decode": "the whole generated sequence in one call of the tokenizer's decode; the tokens it marks as special are skipped; nothing is cut",
        "decode_kwargs": dict(DECODE_KWARGS),
        "strip": "str.strip()",
        "min_valid_fraction": MIN_VALID_FRACTION,
        "special_ids_recipe": SPECIAL_IDS_RECIPE,
        "special_ids_definition": SPECIAL_IDS_DEFINITION,
        "categories": list(CATEGORIES),
        "protocol_run": {
            "pool": PROTOCOL_POOL, "pool_items": PROTOCOL_POOL_ITEMS, "grids": list(PROTOCOL_GRIDS), "paraphrases": list(PROTOCOL_PARAPHRASES), "layout": PROTOCOL_LAYOUT,
            "framing": PROTOCOL_FRAMING, "a_H": list(PROTOCOL_A_H), "c0": list(PROTOCOL_C0), "w": list(PROTOCOL_W), "n_conditions": len(protocol_conditions()),
            "pilot_subset": False,
        },
    }


class DecisionReadingError(ValueError):
    """A tokenizer cannot be read as the reading needs (no special token ids, a decode that does not return text)."""


@dataclass(frozen=True)
class Reading:
    """What one reply says."""

    category: str  # ANSWER, ASK or FORMAT_FAILURE
    first_char: str | None  # the first character of the stripped text, only when it is one of "1".."5"
    self_choice: int | None  # 1..4 for an answer, None otherwise
    first_special_position: int | None  # index (in the generated ids) of the first special token, None if there is none; not used by the reading
    non_special_after_special: bool  # a token that is not special comes after the first special token (False without a special token); not used by the reading
    decoded: str  # the whole generated sequence, decoded, special tokens skipped, not stripped
    text: str  # ``decoded`` stripped: the text whose first character is read

    @property
    def ask(self) -> bool:
        return self.category == ASK

    @property
    def valid(self) -> bool:
        return self.category != FORMAT_FAILURE


def attribute_special_ids(tokenizer: Any) -> frozenset[int]:
    """``tokenizer.all_special_ids`` (the attribute of a Hugging Face tokenizer) as a set of ints."""
    try:
        raw = tokenizer.all_special_ids
    except AttributeError as error:
        raise DecisionReadingError("the tokenizer has no all_special_ids") from error
    return frozenset(int(i) for i in raw if i is not None)


def _added_tokens(tokenizer: Any) -> Mapping[Any, Any]:
    """The added tokens of a tokenizer by id (``added_tokens_decoder``, a dict; a method on some backends, which is called); empty when there is none or it cannot be read."""
    added = getattr(tokenizer, "added_tokens_decoder", None)
    if callable(added):
        try:
            added = added()
        except Exception:  # noqa: BLE001 - a backend that does not list its added tokens
            added = None
    return added if isinstance(added, Mapping) else {}


def special_id_set(tokenizer: Any) -> frozenset[int]:
    """The tokens that the tokenizer marks as special: ``all_special_ids`` and the added tokens flagged ``special`` (see the module docstring for why the attribute alone is not enough)."""
    flagged = {int(i) for i, token in _added_tokens(tokenizer).items() if getattr(token, "special", False)}
    return attribute_special_ids(tokenizer) | flagged


def special_ids_record(special_ids: frozenset[int] | set[int]) -> dict[str, Any]:
    """The size and a digest of the set of special token ids (and the ids themselves when there are few), for the manifest of a run."""
    ordered = sorted(int(i) for i in special_ids)
    digest = hashlib.md5(",".join(str(i) for i in ordered).encode("ascii"), usedforsecurity=False).hexdigest()
    record: dict[str, Any] = {"n": len(ordered), "md5": digest, "md5_of": SPECIAL_IDS_RECIPE}
    if len(ordered) <= MAX_LISTED_SPECIAL_IDS:
        record["ids"] = ordered
    return record


def special_positions(token_ids: Sequence[int], special_ids: frozenset[int] | set[int]) -> tuple[int | None, bool]:
    """``(first_special_position, non_special_after_special)`` of a generated sequence."""
    first = next((i for i, t in enumerate(token_ids) if int(t) in special_ids), None)
    if first is None:
        return None, False
    return first, any(int(t) not in special_ids for t in token_ids[first + 1:])


def read_text(decoded: str, *, first_special_position: int | None = None, non_special_after_special: bool = False) -> Reading:
    """Steps 2 (strip) and 3 (first character) on a text that is already decoded. The two facts about special tokens are carried into the reading unchanged."""
    stripped = decoded.strip()
    first = stripped[0] if stripped else None
    if first is not None and first in ANSWER_CHARS:
        category, recorded, choice = ANSWER, first, int(first)
    elif first == ASK_CHAR:
        category, recorded, choice = ASK, first, None
    else:
        category, recorded, choice = FORMAT_FAILURE, None, None
    return Reading(category, recorded, choice, first_special_position, non_special_after_special, decoded, stripped)


def decode_ids(token_ids: Sequence[int], tokenizer: Any) -> str:
    """The whole generated sequence as text (step 1): one call of ``tokenizer.decode`` with ``DECODE_KWARGS``; an empty sequence is the empty text."""
    ids = [int(t) for t in token_ids]
    if not ids:
        return ""
    text = tokenizer.decode(ids, **dict(DECODE_KWARGS))
    if not isinstance(text, str):
        raise DecisionReadingError(f"tokenizer.decode returned {type(text).__name__}, expected str")
    return text


def read_ids(token_ids: Sequence[int], tokenizer: Any, *, special_ids: frozenset[int] | None = None) -> Reading:
    """The reading of one generated reply: decode the whole sequence with the special tokens skipped, strip, read the first character.

    ``special_ids`` is the set of special token ids that the two recorded facts refer to; it is taken from the tokenizer when it is not given (an entry computes it once and passes it)."""
    ids = [int(t) for t in token_ids]
    special = special_id_set(tokenizer) if special_ids is None else special_ids
    first, after = special_positions(ids, special)
    return read_text(decode_ids(ids, tokenizer), first_special_position=first, non_special_after_special=after)
