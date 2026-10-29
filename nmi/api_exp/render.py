"""The conditions, the user message and the pilot pool of the API experiment.

The user message is produced by ``nmi.prompts.PromptBook.ask_prompt``, the same call that ``scripts/_c2_common.ask_texts`` makes before the chat template of a local model is applied
(``render_chat``). Nothing here formats a prompt by itself, so the message is the local user message character for character. The closed model's own chat template is applied by the service.

``load_pilot_items`` is the only place of this package that reads question content, and it reads the development pool ``mmlu_redux_cal`` only: a pool that is not in ``PILOT_POOLS`` (and
every holdout pool) is refused before the loader is called. Messages are produced on demand; nothing that holds a question text is returned from here except the message itself."""

from __future__ import annotations

import hashlib
import itertools
import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nmi.api_exp.config import PILOT_POOLS, ApiExpConfig, GridSpec
from nmi.data.registry import McItem, SplitRegistry, is_holdout_pool
from nmi.prompts import DisplayRule, PromptBook

_CONDITION_ID = re.compile(r"^a(\d\.\d\d)_c(\d\.\d\d)_w(\d+\.\d\d)$")


class RenderError(RuntimeError):
    """The prompt, the pool or the conditions are not what the configuration says."""


@dataclass(frozen=True)
class Condition:
    """One (a_H, c0, w) cell of the grid, F2 framing."""

    a_H: float
    c0: float
    w: float

    @property
    def id(self) -> str:
        return f"a{self.a_H:.2f}_c{self.c0:.2f}_w{self.w:.2f}"

    @classmethod
    def parse(cls, text: str) -> Condition:
        match = _CONDITION_ID.fullmatch(text)
        if match is None:
            raise RenderError(f"{text!r} is not a condition id (a0.80_c0.00_w0.10)")
        return cls(float(match.group(1)), float(match.group(2)), float(match.group(3)))

    @property
    def cell(self) -> tuple[float, float, float]:
        return (round(self.a_H, 9), round(self.c0, 9), round(self.w, 9))


def conditions_of(grid: GridSpec) -> list[Condition]:
    """The conditions in the order of ``itertools.product(a_H, c0, w)`` (that of ``scripts/run_decisions.grid_conditions``)."""
    return [Condition(a, c, w) for a, c, w in itertools.product(grid.a_H, grid.c0, grid.w)]


def message_md5(message: str) -> str:
    return hashlib.md5(message.encode("utf-8"), usedforsecurity=False).hexdigest()


def load_prompt_book(cfg: ApiExpConfig, rule: DisplayRule) -> PromptBook:
    """Load the prompt file of the configuration and check that the configured paraphrase exists and uses the configured condition block."""
    book = PromptBook.load(cfg.prompt.file, rule=rule)
    if cfg.prompt.paraphrase not in book.paraphrases:
        raise RenderError(f"the prompt file has no paraphrase {cfg.prompt.paraphrase!r}")
    block = book.paraphrases[cfg.prompt.paraphrase].block
    if block != cfg.prompt.block:
        raise RenderError(f"paraphrase {cfg.prompt.paraphrase} uses the condition block {block}, the configuration says {cfg.prompt.block}")
    return book


def render_user_message(book: PromptBook, item: McItem, condition: Condition, *, paraphrase: str, framing: str = "f2", layout: str = "standard") -> str:
    """The user message of one (item, condition): the ask prompt of the existing prompt code, before any chat template."""
    return book.ask_prompt(
        paraphrase, item.question, item.choices, framing=framing, a_H=condition.a_H, c0=condition.c0, w=condition.w, layout=layout,
    )


@dataclass(frozen=True)
class RenderedRequest:
    item_key: str
    condition: Condition
    message: str
    message_md5: str


class RequestPlan:
    """The (item, condition) pairs of a run in item-major order (every condition of an item together, so that a run that stops early keeps whole items, which the paired contrasts need)."""

    def __init__(self, book: PromptBook, items: Sequence[McItem], conditions: Sequence[Condition], *, paraphrase: str, framing: str = "f2", layout: str = "standard") -> None:
        if not items:
            raise RenderError("the plan has no item")
        keys = [i.item_key for i in items]
        if len(set(keys)) != len(keys):
            raise RenderError("the plan holds a repeated item key")
        ids = [c.id for c in conditions]
        if len(set(ids)) != len(ids):
            raise RenderError("the plan holds a repeated condition")
        self.book, self.items, self.conditions = book, list(items), list(conditions)
        self.paraphrase, self.framing, self.layout = paraphrase, framing, layout

    @property
    def item_keys(self) -> list[str]:
        return [i.item_key for i in self.items]

    def limited(self, n_items: int) -> RequestPlan:
        """The plan for the first ``n_items`` questions of the pool, in pool order (the pilot stages use prefixes of the pool of increasing size)."""
        if not 1 <= n_items <= len(self.items):
            raise RenderError(f"a plan of {n_items} questions was asked for, the pool has {len(self.items)}")
        return RequestPlan(self.book, self.items[:n_items], self.conditions, paraphrase=self.paraphrase, framing=self.framing, layout=self.layout)

    @property
    def n_requests(self) -> int:
        return len(self.items) * len(self.conditions)

    def render(self, item: McItem, condition: Condition) -> str:
        return render_user_message(self.book, item, condition, paraphrase=self.paraphrase, framing=self.framing, layout=self.layout)

    def requests(self, skip: Callable[[str, str], bool] | None = None) -> Iterator[RenderedRequest]:
        """Every request, rendered when it is reached. ``skip(item_key, condition_id)`` leaves a pair out (it has a stored reply)."""
        for item in self.items:
            for cond in self.conditions:
                if skip is not None and skip(item.item_key, cond.id):
                    continue
                message = self.render(item, cond)
                yield RenderedRequest(item.item_key, cond, message, message_md5(message))

    def length_stats(self) -> dict[str, Any]:
        """Counts and lengths of the rendered messages, and one md5 over all message md5 values (no text is kept or returned)."""
        digest = hashlib.md5(usedforsecurity=False)
        total = longest = n = 0
        for req in self.requests():
            n += 1
            total += len(req.message)
            longest = max(longest, len(req.message))
            digest.update(req.message_md5.encode("ascii"))
        return {"n_requests": n, "chars_total": total, "chars_mean": total / n, "chars_max": longest, "messages_md5": digest.hexdigest()}


def keys_text(keys: Sequence[str]) -> str:
    """One item key per line (a versioned key file ``plans/<model>.v<k>.keys`` of a run directory: hashes, not question text)."""
    return "".join(f"{k}\n" for k in keys)


def keys_md5(keys: Sequence[str]) -> str:
    return hashlib.md5(keys_text(keys).encode("ascii"), usedforsecurity=False).hexdigest()


def load_pilot_items(local_cfg: Any, registry: SplitRegistry, pool: str, *, results_root: str | Path, loader: Callable[..., list[McItem]] | None = None) -> list[McItem]:
    """The items of the development pool, in pool order. Any pool but ``mmlu_redux_cal`` is refused, and a holdout key among the loaded items raises an error."""
    if pool not in PILOT_POOLS or is_holdout_pool(pool):
        raise RenderError(f"pool {pool!r} is not read by this package; the pilot reads {list(PILOT_POOLS)} and nothing else")
    if loader is None:
        from nmi.data.registry import load_items_from_config

        loader = load_items_from_config
    items = list(loader(local_cfg, pool, registry=registry, results_root=results_root))
    registry.assert_no_holdout([i.item_key for i in items], context=pool)
    return items
