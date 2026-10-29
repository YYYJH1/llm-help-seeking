"""Rendering of the scenario's messages through a model's chat template, with the tool injected by family.

Two ways in which a chat template takes a tool, which depend on the template and not on the weights:

* ``tools_arg`` (the default): the list of tool schemas goes to ``apply_chat_template(..., tools=...)`` (Qwen3, Qwen2.5, SmolLM3, Gemma-4, OLMo-3);
* ``system_field``: the template reads ``message['tools']`` of the system message (Phi-4-mini), so the schemas go into the system message as a JSON string and the ``tools`` argument would be ignored.

``detect_injection`` tells the two apart by the text of the template; a model entry of the run configuration may pin the mode and a different detection is an error. After rendering with a tool, the
tool name must be in the prompt, otherwise the template dropped the tool and ``render`` raises: a model that never saw its tool would answer every question itself.

The generation prompt ends the text (``add_generation_prompt``), the ``chat_template_kwargs`` of the model entry (no thinking) are passed on, and no special token is added when the text is tokenised
(``nmi.readout.hf_backend.encode_prompts``). Nothing here prints a prompt.

System turn. All nine local chat templates (Qwen3-1.7B, 4B, 14B and the expert Qwen3-32B-AWQ, Qwen2.5-7B-Instruct, Gemma-4-E4B-it, OLMo-3-7B-Instruct, Phi-4-mini-instruct, SmolLM3-3B) carry a system
message as a turn of its own: ``system_role`` gives ``"turn"`` for every one of them. So no local model, Gemma-4-E4B-it included, has its system text moved into the first user turn.
``system_role`` stays in the manifest (``scenario_run.system_role``) as the check that would catch a template that merged or dropped the system text.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from nmi.prompts import PromptError

INJECTION_TOOLS_ARG = "tools_arg"
INJECTION_SYSTEM_FIELD = "system_field"
INJECTIONS = (INJECTION_TOOLS_ARG, INJECTION_SYSTEM_FIELD)
_SYSTEM_FIELD_MARK = "message['tools']"


def template_text(tokenizer: Any) -> str:
    """The text of the chat template of a tokenizer (the named templates of a dict, joined)."""
    template = getattr(tokenizer, "chat_template", None)
    if isinstance(template, str):
        return template
    if isinstance(template, Mapping):
        return "\n".join(str(v) for v in template.values())
    return "" if template is None else str(template)


def detect_injection(tokenizer: Any) -> str:
    """``system_field`` when the template of the tokenizer reads ``message['tools']``, else ``tools_arg``."""
    return INJECTION_SYSTEM_FIELD if _SYSTEM_FIELD_MARK in template_text(tokenizer) else INJECTION_TOOLS_ARG


def resolve_injection(tokenizer: Any, pinned: str | None = None) -> str:
    """The injection mode of a model: the detected one, which must equal ``pinned`` when the configuration pins one."""
    detected = detect_injection(tokenizer)
    if pinned is not None:
        if pinned not in INJECTIONS:
            raise PromptError(f"tool_injection must be one of {list(INJECTIONS)}, got {pinned!r}")
        if pinned != detected:
            raise PromptError(f"the configuration pins tool injection {pinned!r}, the chat template calls for {detected!r}")
    return detected


def render(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    chat_template_kwargs: Mapping[str, Any] | None = None,
    *,
    tools: Sequence[Mapping[str, Any]] | None = None,
    injection: str | None = None,
    assistant_prefix: str = "",
) -> str:
    """The prompt as the model reads it: the messages through the chat template, ending at the generation prompt. With ``tools``, the schemas are injected the way ``injection`` says (default: detected)
    and the name of the first tool must appear in the text, else ``PromptError``."""
    kwargs = dict(chat_template_kwargs or {})
    msgs = [dict(m) for m in messages]
    if not msgs:
        raise PromptError("no messages to render")
    if tools:
        mode = injection or detect_injection(tokenizer)
        if mode not in INJECTIONS:
            raise PromptError(f"injection must be one of {list(INJECTIONS)}, got {mode!r}")
        if mode == INJECTION_SYSTEM_FIELD:
            if msgs[0].get("role") != "system":
                raise PromptError("tool injection through the system message needs a system message first")
            msgs[0]["tools"] = json.dumps([t["function"] for t in tools])
            text = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **kwargs)
        else:
            text = tokenizer.apply_chat_template(msgs, tools=list(tools), add_generation_prompt=True, tokenize=False, **kwargs)
        name = tools[0]["function"]["name"]
        if not isinstance(text, str) or name not in text:
            raise PromptError(f"the tool {name!r} did not reach the prompt (injection {mode!r}): the chat template dropped it")
    else:
        text = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **kwargs)
        if not isinstance(text, str):
            raise PromptError(f"apply_chat_template returned {type(text).__name__}, expected str")
    return text + assistant_prefix


def verify_injection(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    chat_template_kwargs: Mapping[str, Any] | None,
    tools: Sequence[Mapping[str, Any]],
    injection: str,
) -> None:
    """The tool schema really reaches the prompt in this mode: the tool name occurs MORE often than in the same conversation rendered without tools. (The system prompts of the scenario name the tool
    themselves, so the presence of the name alone, which ``render`` checks for every prompt, would pass for a template that drops the schema.) ``PromptError`` otherwise. Run once per model and condition
    on a probe question, not for every item."""
    name = tools[0]["function"]["name"]
    with_tools = render(tokenizer, messages, chat_template_kwargs, tools=tools, injection=injection)
    without = render(tokenizer, messages, chat_template_kwargs)
    if with_tools.count(name) <= without.count(name):
        raise PromptError(f"the schema of the tool {name!r} did not reach the prompt (injection {injection!r}): the name is no more frequent than without tools")


_PROBE_SYSTEM, _PROBE_USER = "SYSTEM-PROBE statement.", "USER-PROBE question?"


def system_role(tokenizer: Any, chat_template_kwargs: Mapping[str, Any] | None = None) -> str:
    """How the chat template carries a system message: ``turn`` (a turn of its own), ``merged`` (the template has no system role and puts the system text into the first user turn, joined to the question
    by a blank line, a line break, a space or nothing) or ``dropped`` (the system text is not in the prompt). Judged on invented probe strings."""
    kwargs = dict(chat_template_kwargs or {})
    full = tokenizer.apply_chat_template(
        [{"role": "system", "content": _PROBE_SYSTEM}, {"role": "user", "content": _PROBE_USER}], add_generation_prompt=True, tokenize=False, **kwargs
    )
    if not isinstance(full, str) or _PROBE_SYSTEM not in full:
        return "dropped"
    for joint in ("\n\n", "\n", " ", ""):
        merged = tokenizer.apply_chat_template([{"role": "user", "content": _PROBE_SYSTEM + joint + _PROBE_USER}], add_generation_prompt=True, tokenize=False, **kwargs)
        if merged == full:
            return "merged"
    return "turn"


def fingerprint(text: str) -> str:
    """md5 of a rendered prompt (the manifest records these instead of the prompts)."""
    return hashlib.md5(text.encode("utf-8"), usedforsecurity=False).hexdigest()
