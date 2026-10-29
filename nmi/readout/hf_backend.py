"""Hugging Face readout backend and the HF / vLLM cross-check.

The backend reads the digit distribution of the first generated token from one forward pass:
the logits of the LAST prompt position, converted to float32, ``log_softmax`` over the full
vocabulary, no temperature. This is the same quantity vLLM 0.30 returns under its default
``logprobs_mode = "raw_logprobs"`` (``vllm/config/model.py:259``): the values before any logit
processor or temperature. The two sides are therefore comparable item by item.

Design

* Model family neutral: ``from_pretrained`` uses ``AutoModelForCausalLM`` and ``AutoTokenizer``; nothing
  refers to a decoder class or to ``model.model.layers``. Optional forward arguments
  (``position_ids``, ``logits_to_keep``, ``use_cache``) are passed only when the model's ``forward``
  declares them (the same test ``generate`` applies); otherwise the full logits are computed and the
  last position is taken. ``use_position_ids=False`` drops ``position_ids`` for a family that needs
  another layout (RoPE depends only on relative positions, so left padding stays harmless).
* Prompts are already rendered chat text (``nmi.prompts.render_chat``). They are tokenised with
  ``add_special_tokens=False`` because the template carries every special token (a second BOS would
  change the model input); ``encode_prompts`` is the one tokenisation rule for all backends.
* Batches are padded on the LEFT by hand (pad id from the tokenizer, attention mask zero on the
  pad), so the last position is the last prompt token for every row; ``position_ids`` count from
  the first real token. Hand padding does not depend on the tokenizer's ``padding_side`` (Qwen3 pads
  on the right by default). Batches are formed in order of length to save padding; the result
  order is the input order.
* ``TrialReadout`` is the record all three backends (HF, offline vLLM, HTTP) return.

The cross-check helpers at the bottom (``CrossCheckSpec``, read from ``readout.hf_cross_check`` of the configuration, and ``draw_cross_check_order``) fix which items are compared
between two backends: the items are drawn in a seeded order. The record of a readout holds the id and the log-probability of the most likely first token over the whole vocabulary beside the digit
log-probabilities, all from the one forward pass of the prompt (``_forward_batch`` here, the single ``RequestOutput`` of vLLM): ``TrialReadout.free_token_prob``.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from nmi import seeds
from nmi.readout.digits import DigitReadout, ReadoutSpec, free_token_is_digit

if TYPE_CHECKING:  # pragma: no cover
    import torch


class ReadoutError(RuntimeError):
    """A backend could not produce a readout (never silently turned into "did not ask")."""


# ---------------------------------------------------------------------------------------
# the record every backend returns
# ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class TrialReadout:
    """One trial: the digit readout plus what the model freely generated first."""

    readout: DigitReadout
    free_token_id: int | None  # greedy first token over the full vocabulary (None: HTTP without ids)
    free_is_digit: bool
    free_token_logprob: float | None = None
    finish_reason: str | None = None
    backend: str = ""  # "hf", "vllm_offline" or "vllm_http"

    FIELDS = (
        "digit_logprobs", "digit_mass", "gate_pass", "margin", "ask", "self_choice",
        "free_token_id", "free_is_digit", "free_token_logprob", "finish_reason", "backend",
    )

    @property
    def free_token_prob(self) -> float | None:
        """Probability of the most likely first token over the whole vocabulary (``exp`` of its log-probability, from the same forward pass as the digits)."""
        return None if self.free_token_logprob is None else math.exp(self.free_token_logprob)

    def to_dict(self) -> dict[str, Any]:
        out = self.readout.to_dict()
        out.update(
            free_token_id=self.free_token_id, free_is_digit=self.free_is_digit,
            free_token_logprob=self.free_token_logprob, finish_reason=self.finish_reason, backend=self.backend,
        )
        return out


def encode_prompts(tokenizer: Any, texts: Sequence[str]) -> list[list[int]]:
    """Token ids of rendered chat texts; no special tokens are added (the template has them)."""
    ids: list[list[int]] = []
    for text in texts:
        if not isinstance(text, str):
            raise ReadoutError(f"prompt must be a string, got {type(text).__name__}")
        encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(encoded) == 0:
            raise ReadoutError("a prompt produced no tokens")
        ids.append([int(t) for t in encoded])
    return ids


def left_pad(
    id_lists: Sequence[Sequence[int]], pad_id: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Left-pad token id lists: ``(input_ids, attention_mask, position_ids)`` as int64 arrays.

    Positions count from the first real token (0, 1, 2, ...) and are 0 on the pad.
    """
    if not id_lists:
        raise ReadoutError("nothing to pad")
    longest = max(len(ids) for ids in id_lists)
    n = len(id_lists)
    input_ids = np.full((n, longest), int(pad_id), dtype=np.int64)
    attention = np.zeros((n, longest), dtype=np.int64)
    for row, ids in enumerate(id_lists):
        if len(ids) == 0:
            raise ReadoutError("empty prompt")
        input_ids[row, longest - len(ids):] = ids
        attention[row, longest - len(ids):] = 1
    position = np.clip(np.cumsum(attention, axis=1) - 1, 0, None)
    return input_ids, attention, position


def _declared(fn: Any, names: Sequence[str]) -> dict[str, bool]:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):  # pragma: no cover
        params = {}
    return {name: name in params for name in names}


# ---------------------------------------------------------------------------------------
# backend
# ---------------------------------------------------------------------------------------

class HFBackend:
    """Digit readout from a Hugging Face causal LM (any family ``AutoModelForCausalLM`` loads)."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        spec: ReadoutSpec,
        *,
        device: Any = None,
        batch_size: int = 16,
        sort_by_length: bool = True,
        use_position_ids: bool = True,
    ) -> None:
        import torch

        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.spec = spec
        self.token_ids = spec.token_ids(tokenizer)  # validates that every digit is one clean token
        self.batch_size = batch_size
        self.sort_by_length = sort_by_length
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        pad = getattr(tokenizer, "pad_token_id", None)
        if pad is None:
            pad = getattr(tokenizer, "eos_token_id", None)
        self.pad_id = int(pad) if pad is not None else 0
        self._declared = _declared(model.forward, ("position_ids", "logits_to_keep", "use_cache"))
        if not use_position_ids:  # for a family whose forward expects another layout of position_ids
            self._declared["position_ids"] = False
        self._ids = torch.tensor(self.token_ids, dtype=torch.long, device=self.device)

    # ---- construction from disk ----------------------------------------------------------
    @classmethod
    def from_pretrained(
        cls,
        path: str,
        spec: ReadoutSpec,
        *,
        dtype: Any = "bfloat16",
        device: str = "cuda",
        batch_size: int = 16,
        attn_implementation: str | None = None,
        use_position_ids: bool = True,
        **kwargs: Any,
    ) -> HFBackend:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) and dtype != "auto" else dtype
        tokenizer = AutoTokenizer.from_pretrained(path)
        model_kwargs = dict(kwargs)
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch_dtype, **model_kwargs)
        model.to(device)
        return cls(model, tokenizer, spec, device=device, batch_size=batch_size, use_position_ids=use_position_ids)

    # ---- readout ------------------------------------------------------------------------
    def readouts(self, texts: Sequence[str]) -> list[TrialReadout]:
        """Readouts of rendered chat prompts, in input order."""
        return self.readouts_from_ids(encode_prompts(self.tokenizer, texts))

    def readouts_from_ids(self, id_lists: Sequence[Sequence[int]]) -> list[TrialReadout]:
        n = len(id_lists)
        order = list(range(n))
        if self.sort_by_length:
            order.sort(key=lambda i: (len(id_lists[i]), i))
        out: list[TrialReadout | None] = [None] * n
        for start in range(0, n, self.batch_size):
            chunk = order[start:start + self.batch_size]
            digit_lp, free_id, free_lp = self._forward_batch([id_lists[i] for i in chunk])
            for j, i in enumerate(chunk):
                out[i] = self._record(digit_lp[j], int(free_id[j]), float(free_lp[j]))
        if any(r is None for r in out):  # pragma: no cover
            raise ReadoutError("internal error: a prompt got no readout")
        return out  # type: ignore[return-value]

    def _record(self, digit_logprobs: np.ndarray, free_id: int, free_lp: float) -> TrialReadout:
        return TrialReadout(
            readout=self.spec.readout([float(x) for x in digit_logprobs]),
            free_token_id=free_id,
            free_is_digit=free_token_is_digit(free_id, self.token_ids),
            free_token_logprob=free_lp,
            finish_reason=None,
            backend="hf",
        )

    def last_position_logprobs(self, id_lists: Sequence[Sequence[int]]) -> np.ndarray:
        """Full-vocabulary float32 log-softmax of the last position, one row per prompt (testing aid)."""
        import torch

        input_ids, attention, position = left_pad(id_lists, self.pad_id)
        with torch.inference_mode():
            last = self._last_logits(input_ids, attention, position)
            return last.float().log_softmax(dim=-1).cpu().numpy()

    def _last_logits(self, input_ids: np.ndarray, attention: np.ndarray, position: np.ndarray) -> torch.Tensor:
        import torch

        kwargs: dict[str, Any] = {
            "input_ids": torch.from_numpy(input_ids).to(self.device),
            "attention_mask": torch.from_numpy(attention).to(self.device),
        }
        if self._declared["position_ids"]:
            kwargs["position_ids"] = torch.from_numpy(position).to(self.device)
        if self._declared["logits_to_keep"]:
            kwargs["logits_to_keep"] = 1
        if self._declared["use_cache"]:
            kwargs["use_cache"] = False
        output = self.model(**kwargs)
        logits = output.logits if hasattr(output, "logits") else output[0]
        return logits[:, -1, :]

    def _forward_batch(self, id_lists: Sequence[Sequence[int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import torch

        input_ids, attention, position = left_pad(id_lists, self.pad_id)
        with torch.inference_mode():
            last = self._last_logits(input_ids, attention, position)
            logprobs = last.float().log_softmax(dim=-1)  # raw logits, float32, no temperature
            if not torch.isfinite(logprobs).all():
                raise ReadoutError("non-finite logits at the last position")
            digit = logprobs.index_select(1, self._ids).double().cpu().numpy()
            top_lp, top_id = logprobs.max(dim=-1)
            return digit, top_id.cpu().numpy(), top_lp.cpu().numpy()


# ---------------------------------------------------------------------------------------
# cross-check
# ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class CrossCheckSpec:
    pool: str
    seed_cell: str
    target_non_tie_items: int
    max_items_drawn: int
    near_tie_abs_margin: float
    max_disagreements_non_tie: int

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> CrossCheckSpec:
        c = cfg["readout"]["hf_cross_check"]
        return cls(
            pool=str(c["pool"]), seed_cell=str(c["seed_cell"]),
            target_non_tie_items=int(c["target_non_tie_items"]), max_items_drawn=int(c["max_items_drawn"]),
            near_tie_abs_margin=float(c["near_tie_abs_margin"]),
            max_disagreements_non_tie=int(c["max_disagreements_non_tie"]),
        )


def draw_cross_check_order(n_pool: int, key: Sequence[int], max_items: int) -> np.ndarray:
    """The first ``max_items`` entries of a seeded permutation of the pool (seed key from
    ``SeedNamespace.hf_cross_check_key()``): the items are taken one after the other in this order."""
    if n_pool < 1 or max_items < 1:
        raise ValueError("pool and draw size must be positive")
    return seeds.rng(key).permutation(n_pool)[: min(max_items, n_pool)]
