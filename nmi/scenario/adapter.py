"""The parsing and execution adapter of the real scenario S1.

One fixed piece of code, used both to analyse the generated replies offline and, in a real loop, to decide what a server does with a reply. It is NOT a claim about what a production tool-call parser
accepts: it is the adapter of this study, applied alike to every model, condition and prompt version.

A reply is cleaned (``<think>...</think>`` blocks and the end-of-turn tokens are dropped) and scanned from left to right for call fragments:

* the formats with tags (the native writings of the families): ``<tool_call>{json}</tool_call>`` (Qwen, SmolLM3), ``<|tool_call|>{json}<|/tool_call|>`` (Phi), ``<|tool_call>call:name{question:<|"|>...<|"|>}<tool_call|>``
  (Gemma-4) and a ``<function_calls>name(...)</function_calls>`` block (OLMo-3);
* a bare JSON object or list with a ``name`` and ``arguments`` (or ``parameters``);
* the call expression ``ask_edge_expert(...)`` (not after a word character, a dot or a hyphen);
* the tool name directly followed (whitespace allowed) by one complete JSON object, ``ask_edge_expert {"question": "..."}``. An object with a ``name`` key after the name is a call object of its own and is judged
  as bare JSON (the name before it is text).

NOT accepted, hence a format failure when nothing else in the reply is a valid call: a name preceded by a dot, a single-quoted Python dict, the name followed by a string, a Gemma call without its
``<|tool_call>call:`` shell, the bare name. A call inside a quotation or a code block is a call: the rule has no exemption.

STRICT ARGUMENTS. A call is valid if and only if its arguments, normalised, are exactly ``{"question": s}`` with ``s`` a string that is not blank. Another key (also next to a valid ``question``), a missing
``question``, a repeated key (the object is ambiguous), a non-string or blank question, several arguments, a positional argument next to ``question=``, an empty argument list, a trailing comma: invalid. A call
expression takes three argument forms only (``question="..."``, one JSON object, one double-quoted string). For a call object (a JSON object with a ``name`` key): a repeated key anywhere in it (two ``name``,
two ``arguments``, two ``parameters``) and both ``arguments`` and ``parameters`` together are ambiguous and make the call invalid; the arguments may be an object or a string that decodes to one.
A fragment that cannot be split into calls (cut off, undecodable, junk in a block, an unclosed tag) is one invalid segment, as is a call to another tool.

EXECUTION (``execution``). At most one call is executed per request: the first valid segment from the left. The server forwards the question it holds (the original question), never the generated one; the
generated question only decides validity, and ``arg_equals_question`` records whether it equals the original one after ``triviaqa.normalize``. Text before or after the call is never delivered. Further valid
calls are not executed (``multi_call``). ``execution_native`` reads the reply as an adapter that knows only the native writing of one model family (the native-writing sensitivity analysis): of the call fragments only
those that begin with the family's opening tag are calls; the others, tags of other families included, are text.

THREE OUTCOMES (``outcome_of``; every reply has exactly one): ``offload`` (at least one valid call), ``self_answer`` (a non-empty reply without call syntax and without the tool name; the answer is the
first line) and ``format_failure`` (call syntax without a valid call, the tool name only in prose, or an empty reply). Missing replies (a generation that failed twice) are none of the three.

DIAGNOSTICS without content: the structure signature (C valid call, I invalid fragment, P text, N text that names the tool, E empty), the syntax of the executed call, the shape of a non-compliant reply (every letter
and digit replaced by x). Rules version: see ``RULES_VERSION``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from nmi.data.triviaqa import normalize

RULES_VERSION = "v3.1"  # version label of the parsing rules; recorded in the manifests of the runs and analyses
CALL_NAME = "ask_edge_expert"
MARKERS = ("<tool_call>", "<|tool_call|>", "<|tool_call>", "<function_calls>", "functools[", "[TOOL_CALLS]", "<function_call>", "```tool_call")
END_TOKENS = ("<|im_end|>", "<|end|>", "<|endoftext|>", "<turn|>", "<end_of_turn>", "</s>", "<|eot_id|>")
# opening tag -> (closing tag, format of the body): json = JSON between the tags (Qwen, SmolLM3, Phi); gemma = call:name{question:<|"|>...<|"|>}; py = name(...) calls (OLMo-3)
TAGS: Mapping[str, tuple[str, str]] = {
    "<tool_call>": ("</tool_call>", "json"), "<|tool_call|>": ("<|/tool_call|>", "json"), "<|tool_call>": ("<tool_call|>", "gemma"), "<function_calls>": ("</function_calls>", "py"),
}
TAG_SYNTAX = {"json": "tagged_json", "gemma": "gemma", "py": "olmo_tagged"}
TAGGED_SYNTAXES = frozenset(("tagged_json", "gemma", "olmo_tagged"))
FAMILY_TAG = {"qwen3": "<tool_call>", "qwen2.5": "<tool_call>", "smollm3": "<tool_call>", "phi": "<|tool_call|>", "gemma": "<|tool_call>", "olmo": "<function_calls>"}
FORWARDED = "original_question"
_OUTCOME_OF_CLASS = {"valid_call": "offload", "direct_answer": "self_answer", "incomplete_call": "format_failure", "name_in_prose": "format_failure", "empty": "format_failure"}

_STR = r'"(?:[^"\\]|\\.)*"'  # a double-quoted string literal with backslash escapes
_STR_FULL = re.compile(_STR)
_PY_ARGS = re.compile(r'\(((?:[^()"]|' + _STR + r')*)\)', re.S)  # "(...)" with parentheses inside string literals skipped; group 1 is the argument text
_PY_CALL = re.compile(r"[\s,]*([\w.-]+)\(")  # the next name( in a series of calls
_PY_CLOSE = re.compile(r"\s*</function_calls>")
_JSON_CALL_HEAD = re.compile(r'(?:\[\s*)?\{\s*"name"\s*(?::|$)')  # the beginning of a JSON call that does not decode (cut off by max_tokens, or malformed)
_NAME_RX = {"json": re.compile(r'"name"\s*:\s*"([^"]*)"'), "gemma": re.compile(r"call:([\w.-]+)"), "py": re.compile(r"([\w.-]+)\(")}
_START = re.compile("|".join(re.escape(t) for t in TAGS) + r"|(?<![\w.-])" + CALL_NAME + r"(?:\(|(?=\s*\{))|[\[{]")  # where a call fragment can begin
_WS = re.compile(r"\s*")
_KEYWORD_FULL = re.compile(r"question\s*=\s*(" + _STR + r")", re.S)  # the whole argument text, stripped, is this one keyword argument
_GEMMA_CALL = re.compile(r"\s*call:([\w.-]+)\{(.*)\}\s*", re.S)
_GEMMA_ARGS = re.compile(r'\s*question:<\|"\|>((?:(?!<\|"\|>).)*)<\|"\|>\s*', re.S)  # between the braces: the one item question, its text without the quote token
_THINK = re.compile(r"<think>.*?</think>", re.S)
_MARKER_TAIL = max(len(mk) for mk in MARKERS)


# ---------------------------------------------------------------- JSON with a mark for repeated keys
class _Obj(dict):
    """A decoded JSON object. dup: a key occurred twice (the dict keeps the last value, so the object is ambiguous and never a valid argument object or call object)."""

    dup = False


def _hook(pairs: list[tuple[str, Any]]) -> _Obj:
    d = _Obj(pairs)
    if len(d) != len(pairs):
        d.dup = True
    return d


_JSON = json.JSONDecoder(object_pairs_hook=_hook)


def _question_of(obj: Any) -> str | None:
    """The question when obj is exactly {"question": non-blank string} (one key, not repeated), otherwise None."""
    if isinstance(obj, dict) and not getattr(obj, "dup", False) and len(obj) == 1 and isinstance(obj.get("question"), str) and obj["question"].strip():
        return obj["question"]
    return None


def _literal(text: str) -> str | None:
    """The non-blank string that the double-quoted literal ``text`` stands for (backslash escapes as in JSON), otherwise None."""
    try:
        value = json.loads(text)
    except (ValueError, RecursionError):
        return None
    return value if isinstance(value, str) and value.strip() else None


def _object_arg(text: str) -> dict | None:
    """The dict when ``text`` (argument text, stripped) is exactly one JSON object and nothing else, otherwise None."""
    try:
        obj, end = _JSON.raw_decode(text)
    except (ValueError, RecursionError):
        return None
    return obj if isinstance(obj, dict) and not text[end:].strip() else None


# ---------------------------------------------------------------- the argument rule of the call expression ask_edge_expert(...)
def _py_arg(argtext: str) -> tuple[str | None, str | None]:
    """(form, question) of the argument text of a call to the tool: form is py_keyword, py_json_arg or py_string_arg, question the generated question; (None, None) when it is none of the three."""
    t = argtext.strip()
    m = _KEYWORD_FULL.fullmatch(t)
    if m is not None:
        q = _literal(m.group(1))
        return ("py_keyword", q) if q is not None else (None, None)
    if t.startswith("{"):
        q = _question_of(_object_arg(t))
        return ("py_json_arg", q) if q is not None else (None, None)
    if t.startswith('"') and _STR_FULL.fullmatch(t):
        q = _literal(t)
        return ("py_string_arg", q) if q is not None else (None, None)
    return (None, None)


# ---------------------------------------------------------------- the calls inside one fragment: lists of (name, ok, question)
def _json_calls(obj: Any) -> list[tuple[str | None, bool, str | None]]:
    """(name, ok, question) of every call in a decoded JSON value: a dict with a "name" key, or each such dict in a list. ok: the call object is not ambiguous (no repeated key, not both "arguments" and
    "parameters"), the name is the tool and the arguments (key "arguments", else key "parameters"; an object, or a string that decodes to one) are exactly {"question": non-blank string}.
    An ambiguous call object is one invalid call; the name that is given for it is the one a decoder keeps (the last value of a repeated "name"), only ok is False."""
    calls = []
    for it in obj if isinstance(obj, list) else [obj]:
        if not isinstance(it, dict) or "name" not in it:
            continue
        ambiguous = getattr(it, "dup", False) or ("arguments" in it and "parameters" in it)
        args = it["arguments"] if "arguments" in it else it.get("parameters", {})
        if isinstance(args, str):
            try:
                args = _JSON.decode(args)
            except (ValueError, RecursionError):
                args = None
        q = _question_of(args)
        name = it["name"] if isinstance(it["name"], str) else None
        ok = not ambiguous and name == CALL_NAME and q is not None
        calls.append((name, ok, q if ok else None))
    return calls


def _py_calls(text: str) -> list[tuple[str | None, bool, str | None]]:
    """(name, ok, question) of the name(args) calls that make up ``text`` (separated by whitespace or commas); [] unless text consists of closed calls only."""
    calls, pos = [], 0
    while True:
        m = _PY_CALL.match(text, pos)
        a = _PY_ARGS.match(text, m.end() - 1) if m else None
        if a is None:
            break
        name = m.group(1)
        form, q = _py_arg(a.group(1)) if name == CALL_NAME else (None, None)
        calls.append((name, form is not None, q))
        pos = a.end()
    return calls if calls and not text[pos:].strip() else []


def _tag_calls(family: str, body: str) -> list[tuple[str | None, bool, str | None]]:
    """(name, ok, question) of the calls between an opening tag and its closing tag; [] when the body is not a call in the format of the family."""
    if family == "json":
        try:
            return _json_calls(_JSON.decode(body))
        except (ValueError, RecursionError):
            return []
    if family == "gemma":
        m = _GEMMA_CALL.fullmatch(body)
        if m is None:
            return []
        a = _GEMMA_ARGS.fullmatch(m.group(2))
        q = a.group(1) if a is not None and m.group(1) == CALL_NAME and a.group(1).strip() else None
        return [(m.group(1), q is not None, q)]
    return _py_calls(body)


def _guess_name(family: str, text: str) -> str | None:
    m = _NAME_RX[family].search(text)
    return m.group(1) if m else None


# ---------------------------------------------------------------- the scan
def _frag(start: int, end: int, calls: list, guess: str | None = None, syntax: str | None = None) -> dict[str, Any]:
    n_valid = sum(ok for _, ok, _ in calls)
    return {"start": start, "end": end, "valid": n_valid > 0, "name": calls[0][0] if calls else guess, "n_valid": n_valid, "calls": calls, "syntax": syntax if n_valid > 0 else None}


def scan(s: str) -> list[dict[str, Any]]:
    """The call fragments of the cleaned text ``s``, left to right and without overlap. Each is {"start", "end", "valid", "name", "n_valid", "calls", "syntax"}: valid = it holds a closed valid call to the tool;
    n_valid counts such calls (a JSON list or a <function_calls> block can hold several); name is the tool name of the first call; calls is the list of (name, ok, question) of the calls inside the fragment
    ([] when it cannot be split into calls); syntax is the notation of the fragment when it is valid (one of tagged_json, gemma, olmo_tagged, bare_json, py_keyword, py_json_arg, py_string_arg, name_json), otherwise None. An opening tag without its closing tag, a call expression that does
    not close and a JSON call cut off after {"name": run to the end of the text."""
    found: list[dict[str, Any]] = []
    i, n = 0, len(s)
    while i < n:
        m = _START.search(s, i)
        if m is None:
            break
        p, tok = m.start(), m.group()
        if tok in TAGS:
            close, family = TAGS[tok]
            e = s.find(close, m.end())
            body = s[m.end(): n if e < 0 else e]
            end = n if e < 0 else e + len(close)
            found.append(_frag(p, end, [] if e < 0 else _tag_calls(family, body), _guess_name(family, body), TAG_SYNTAX[family]))
            i = end
        elif tok in ("[", "{"):
            try:
                obj, end = _JSON.raw_decode(s, p)
            except (ValueError, RecursionError):
                if _JSON_CALL_HEAD.match(s, p):
                    found.append(_frag(p, n, [], _guess_name("json", s[p:]), "bare_json"))
                    break
                i = p + 1
                continue
            head = obj[0] if isinstance(obj, list) and obj else obj
            if isinstance(head, dict) and "name" in head:
                found.append(_frag(p, end, _json_calls(obj), None, "bare_json"))
                i = end
            else:
                i = p + 1
        elif tok.endswith("("):  # ask_edge_expert( without an opening tag, with or without the closing </function_calls>
            a = _PY_ARGS.match(s, m.end() - 1)
            if a is None:
                found.append(_frag(p, n, [], CALL_NAME))
                break
            t = _PY_CLOSE.match(s, a.end())
            end = t.end() if t else a.end()
            form, q = _py_arg(a.group(1))
            found.append(_frag(p, end, [(CALL_NAME, form is not None, q)], None, form))
            i = end
        else:  # the tool name before whitespace and "{": the name directly followed by a JSON object
            b = _WS.match(s, m.end()).end()
            try:
                obj, end = _JSON.raw_decode(s, b)
            except (ValueError, RecursionError):
                if _JSON_CALL_HEAD.match(s, b):  # a cut-off call object of its own ({"name": ...): the name is text and the object is judged as bare JSON
                    i = b
                    continue
                found.append(_frag(p, n, [], CALL_NAME))
                break
            if "name" in obj:  # a call object of its own ({"name": ..., "arguments": ...}): the name is text and the object is judged as bare JSON
                i = b
                continue
            q = _question_of(obj)
            t = _PY_CLOSE.match(s, end)
            end = t.end() if t else end
            found.append(_frag(p, end, [(CALL_NAME, q is not None, q)], None, "name_json"))
            i = end
    return found


# ---------------------------------------------------------------- cleaning, classes, diagnostics
def clean(out: str) -> str:
    """Drop empty or full think blocks and the end-of-turn tokens; then strip."""
    s = _THINK.sub("", out)
    for tok in END_TOKENS:
        s = s.replace(tok, "")
    return s.strip()


def classify(out: str) -> str:
    """empty; valid_call: a closed valid call anywhere in the text; incomplete_call: an invalid call fragment (cut off, malformed, unknown tool or bad arguments), a call marker, or call syntax at the
    start of the text that does not parse; name_in_prose: the tool name without any of these; direct_answer: anything else."""
    s = clean(out)
    if not s:
        return "empty"
    calls = scan(s)
    if any(c["valid"] for c in calls):
        return "valid_call"
    if calls or any(mk in s for mk in MARKERS):
        return "incomplete_call"
    if s.startswith((CALL_NAME + "(", CALL_NAME + "{")):
        return "incomplete_call"
    if CALL_NAME in s:
        return "name_in_prose"
    return "direct_answer"


def outcome_of(label: str) -> str:
    """The three outcomes: ``offload`` (valid_call), ``self_answer`` (direct_answer), ``format_failure`` (incomplete_call, name_in_prose, empty)."""
    return _OUTCOME_OF_CLASS[label]


def _text_kind(t: str) -> str:
    return "N" if CALL_NAME in t else ("P" if t.strip() else "")


def structure(out: str) -> str:
    """Signature of a reply without its content: the call fragments (C valid, I invalid) and the text between them (N when it mentions the tool, otherwise P when it is not blank), adjacent repeats
    merged and joined by +, for example P+C, C or P; E for an empty reply."""
    s = clean(out)
    if not s:
        return "E"
    parts, pos = [], 0
    for c in scan(s):
        parts += [_text_kind(s[pos: c["start"]]), "C" if c["valid"] else "I"]
        pos = c["end"]
    parts.append(_text_kind(s[pos:]))
    parts = [p for p in parts if p]
    return "+".join(p for k, p in enumerate(parts) if k == 0 or p != parts[k - 1])


def shape(out: str) -> str:
    """The structure of a reply without its content: every letter and digit becomes x, the tool name and the known markers are kept; first 48 characters (at most 80 in all)."""
    s = clean(out)
    keep = [*MARKERS, CALL_NAME, "<tool_call|>", "</tool_call>", "<|/tool_call|>", "</function_calls>"]
    out_chars: list[str] = []
    i = 0
    while i < len(s) and len(out_chars) < 48:
        hit = next((k for k in keep if s.startswith(k, i)), None)
        if hit:
            out_chars.append(hit)
            i += len(hit)
        else:
            ch = s[i]
            out_chars.append("x" if ch.isalnum() else ch)
            i += 1
    return "".join(out_chars)[:80]


# ---------------------------------------------------------------- execution
def _gap_has_text(s: str, a: int, b: int) -> bool:
    """True when s[a:b], the part of the reply between two call fragments (b is where the next fragment begins, or the end), holds text. The call markers (functools[, [TOOL_CALLS], <function_call>,
    ```tool_call ...) are syntax and not text, also when a marker runs from the end of the gap into the fragment (functools[ before a JSON list); a closing tag that is not a marker, or a markdown fence,
    counts as text."""
    piece = s[a:b]
    for mk in MARKERS:
        piece = piece.replace(mk, "")
    tail = s[b: b + _MARKER_TAIL]
    for mk in MARKERS:
        for k in range(1, len(mk)):
            if piece.endswith(mk[:k]) and tail.startswith(mk[k:]):
                piece = piece[: len(piece) - k]
                break
    return bool(piece.strip())


def _execute(out: str, original_question: str | None, tag: str | None = None) -> dict[str, Any]:
    """execution() for tag None (every call fragment is a call) and execution_native() for the opening tag of one family (only the fragments that begin with it are calls; the other fragments are text)."""
    s = clean(out)
    frags = scan(s)
    if tag is not None:
        frags = [f for f in frags if s.startswith(tag, f["start"])]
    segs: list[tuple[int, bool, str | None]] = []  # (fragment index, ok, generated question) of every call, in order; an unsplittable fragment is one invalid segment
    for k, f in enumerate(frags):
        segs += [(k, ok, q) for _, ok, q in f["calls"]] if f["calls"] else [(k, False, None)]
    n_valid = sum(ok for _, ok, _ in segs)
    idx = next((j for j, (_, ok, _) in enumerate(segs) if ok), None)
    res: dict[str, Any] = {
        "offload": idx is not None, "executed_index": idx, "n_valid": n_valid, "n_invalid": len(segs) - n_valid, "forwarded": FORWARDED, "arg_equals_question": None,
        "text_before": None, "text_after": None, "syntax": None, "first_valid_tagged": None,
    }
    if idx is None:
        return res
    k, _, q = segs[idx]
    bounds = [0] + [x for f in frags for x in (f["start"], f["end"])] + [len(s)]
    gaps = [_gap_has_text(s, bounds[2 * j], bounds[2 * j + 1]) for j in range(len(frags) + 1)]  # the text outside the fragments: gap j lies before fragment j
    res["text_before"] = any(gaps[: k + 1])
    res["text_after"] = any(gaps[k + 1:])
    res["syntax"] = frags[k]["syntax"]
    res["first_valid_tagged"] = res["syntax"] in TAGGED_SYNTAXES
    if original_question is not None and q is not None:
        res["arg_equals_question"] = normalize(q) == normalize(original_question)
    return res


def execution(out: str, original_question: str | None = None) -> dict[str, Any]:
    """What a server does with the reply ``out``:
      offload              at least one valid call (the same as classify(out) == "valid_call")
      executed_index       0-based position of the first valid segment among all segments (valid or invalid) from the left; None without a valid call
      n_valid, n_invalid   numbers of valid and invalid segments
      forwarded            "original_question" for every reply: the server forwards the question it holds, the generated one only decides validity
      arg_equals_question  normalize(generated question of the executed call) == normalize(original_question); None when original_question is None or there is no valid call
      text_before          non-blank text outside the call fragments before the executed call (call markers are not text); None without a valid call
      text_after           the same after the executed call (text between two calls and after the last one); None without a valid call
      syntax               notation of the executed call, one of the notations listed under ``scan``; None without a valid call
      first_valid_tagged   syntax in TAGGED_SYNTAXES (the executed call is written with tags, of whatever family); None without a valid call"""
    return _execute(out, original_question)


def execution_native(out: str, family: str, original_question: str | None = None) -> dict[str, Any]:
    """execution() for an adapter that knows only the native writing of one model family (one of FAMILY_TAG: qwen3, qwen2.5, smollm3, phi, gemma, olmo): of the call fragments of the reply only those that
    begin with the opening tag of the family are calls, and the first valid call among them is executed. The other call fragments (another notation, the tags of another family) are text. Same keys and
    meaning as execution() on that reading: executed_index, n_valid and n_invalid count the native fragments only; offload is True only if execution(out)["offload"] is. An unknown family is a ValueError."""
    try:
        tag = FAMILY_TAG[family]
    except (KeyError, TypeError):
        raise ValueError(f"unknown model family {family!r}; choose one of {', '.join(FAMILY_TAG)}") from None
    return _execute(out, original_question, tag)


def executed_native(out: str, family: str) -> bool | None:
    """Whether the call that ``execution`` executes is written in the native tags of ``family`` (the fragment of the first valid call begins with the opening tag of the family); None without a valid call.
    The share of True among the hand-overs is the share of native writing; ``execution()["first_valid_tagged"]`` is the weaker statement that the call has the tags of SOME family."""
    try:
        tag = FAMILY_TAG[family]
    except (KeyError, TypeError):
        raise ValueError(f"unknown model family {family!r}; choose one of {', '.join(FAMILY_TAG)}") from None
    s = clean(out)
    for fragment in scan(s):
        if fragment["valid"]:
            return s.startswith(tag, fragment["start"])
    return None


# ---------------------------------------------------------------- one reply, all derived fields
def analyze(out: str, original_question: str | None, family: str) -> dict[str, Any]:
    """Every derived field of a P-decide reply, content-free: the class (five labels) and the outcome (three), the structure signature, the syntax of the executed call, whether it is written with tags,
    the number of valid and invalid segments, multi_call, whether the generated question equals the original one, text before and after, whether the executed call is in the family's own tags
    (``executed_native``), the native-writing reading (``native_offload``) and, for a format failure, the shape. The reply itself is not part of the result."""
    label = classify(out)
    ex = execution(out, original_question)
    nat = execution_native(out, family, original_question)
    return {
        "class": label, "outcome": outcome_of(label), "offload": ex["offload"], "structure": structure(out), "syntax": ex["syntax"], "first_valid_tagged": ex["first_valid_tagged"],
        "n_valid": ex["n_valid"], "n_invalid": ex["n_invalid"], "multi_call": ex["n_valid"] > 1, "executed_index": ex["executed_index"], "arg_equals_question": ex["arg_equals_question"],
        "text_before": ex["text_before"], "text_after": ex["text_after"], "executed_native": executed_native(out, family), "native_offload": nat["offload"], "native_syntax": nat["syntax"],
        "native_n_valid": nat["n_valid"],
        "shape": shape(out) if outcome_of(label) == "format_failure" else None,
    }
