"""Train LoRA adapters for cost-conditional help-seeking: tau estimation, LoRA training, checkpoint selection and merging.

CPU tests use a tiny randomly initialised model. This file never prints or stores question text: a prompt exists in memory, the manifest holds a sha256 of the token ids.

Commands
    tau              tau = max(1, median |m|) of the frozen base model over the 12,800 original ``train`` decision prompts; writes the tau file (JSON), once per model. The file holds
                     the sha256 over the bytes of every base file: a Hugging Face cache blob is also checked against its name; ``train`` and ``select`` compare these hashes.
    train            LoRA training of one run (one model, one objective, one seed, one (lr, beta)). ``--device`` defaults to cpu (all GPUs hidden from the process); a GPU needs an explicit ``cuda:N``.
                     routing, fixed_cost and shuffled need ``--tau-file``; sft refuses it. ``--time-only N`` times N steps of the formal configuration and writes ``timing.json``.
    select           the fixed-rule choice of a checkpoint among the seed-0 runs of one (model, objective) search, from the readouts of the sets ``val`` and ``val_panswer`` (a candidate whose
                     readouts fail the quality gate is "readout invalid" and not ranked; the base must pass).
    merge            writes a merged bf16 checkpoint for vLLM and checks it. Acceptance is check 1 alone: the weights, bitwise, on the fixed CPU path. Check 2 is a report without
                     thresholds: margins, digit argmax and decision of three computations on the 4 fixed prompts (A merged checkpoint upcast, B adapter on in the training numerics, C adapter on in float32).
    estimate-memory  the GPU memory estimate from the model configuration.

Objectives. ``l`` are the exact log-probabilities of the digit tokens "1".."5" at the readout position (the position after the assistant prefix; log-softmax over the
full vocabulary), ``help`` is the item's own help label and ``a`` ranges over the item's own four answer labels:

    m = l_help - max_a l_a,   q = sigmoid(m / tau_model),   A = a_H - c0 - w,   u = q A + (1 - q) z0
    routing     L = -mean(u) + beta (KL1 + KL2) + lambda_M (L_M,decision + L_M,P-answer)
    sft         L = mean(CE of the target token) + beta KL2 + lambda_M (L_M,decision + L_M,P-answer);  target = help label if A > z0 else the frozen model's top answer label on the same prompt
    fixed_cost  as routing, but every training condition has w = 0.20 (a_H, c0, mapping as in ``train``)
    shuffled    as routing, but the displayed (a_H, c0, w) of prompt i is those of prompt i + 1 mod 16 of the batch; item, z0, own A, mapping and labels stay
    KL1 = KL(P(four answer labels | decision prompt) || frozen reference),   KL2 = KL(P(options 1..4 | P-answer prompt) || frozen reference),   both renormalised over their four labels;
    L_M = KL(Bern(M_ref) || Bern(M_theta)) with M the total probability of the five digits (decision prompt) or of the four answer digits (P-answer prompt), lambda_M = 1;
    tau_model = max(1, median |m|) of the frozen model on the original train prompts, from the tau file; the readout and every evaluation still use m itself (decision m > 0);
    the reference is the same network with the adapter disabled (``peft`` ``disable_adapter()``) on the same batch, without gradient.

Interface to the prompt module (``scripts/_part3_b_prompts.py``, imported, never edited): ``decision_prompt(item, cond)``, ``panswer_prompt(item, domain)`` (user text only), ``render_for_model(tokenizer,
user_text, model_id)`` (token ids with chat template and prefix, exactly once). The items, z0 and the ``train`` / ``val`` condition plans come from that module through ``load_split_from_b1``.

Device policy: ``--device`` defaults to ``cpu``, a GPU needs an explicit ``cuda:N`` (a bare ``cuda`` is refused), and the device is settled before ``peft``, ``bitsandbytes`` or
``transformers`` are imported, because ``bitsandbytes`` (imported by ``peft``) initialises EVERY visible card. A CPU run sets ``CUDA_VISIBLE_DEVICES=""`` for its own process. A GPU run with
``CUDA_VISIBLE_DEVICES`` unset is shown only card N (used as cuda:0 inside the process); with the variable already set, N indexes the visible list (``CUDA_VISIBLE_DEVICES=2`` goes with ``cuda:0``).
``select`` and ``estimate-memory`` never see a GPU.

One (model, objective) search, then the final runs
    1. ``train`` the 4 cells (lr 1e-4, 2e-4) x (beta 0.1, 1.0) at seed 0 with ``--steps 1000 --save-at 250,500,1000``; give every run directory a unique name, e.g. ``<model>_<objective>_s0_lr<lr>_b<beta>``.
    2. ``merge`` each of the 12 adapters (``adapter_step0250`` ...), then ``scripts/run_part3_b_eval.py read --sets val,val_panswer`` on every merged checkpoint and on ``base``. ``select`` finds the readouts at
       ``<val-out>/base/{val,val_panswer}/rows.jsonl`` and ``<val-out>/<run directory name>_step<N>/{val,val_panswer}/rows.jsonl``; nothing else is opened.
    3. ``select --runs <the 4 run directories> --val-out <val-out>``: writes ``<val-out>/selection.json`` (exit 0 selected, exit 3 "no selectable checkpoint", the constraint is never relaxed).
    4. The final run is the selected seed-0 checkpoint plus seeds 1 and 2 trained to the selected step (``selection.json`` lists them under ``retrain``); the Qwen3-4B ``routing`` selection also lists
       the ``fixed_cost`` and ``shuffled`` controls (seed 0, same lr, beta and step) under ``controls``.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import functools
import gc
import hashlib
import importlib
import json
import math
import os
import platform
import re
import shutil
import statistics
import struct
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for _p in (str(REPO), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

PROGRAM = "train_part3_b/1"
CONFIG_PATH = REPO / "configs" / "v6_confirm.yaml"
PROMPT_BOOK_PATH = REPO / "configs" / "prompts" / "ask_v1_1.yaml"
DESIGN_PATH = REPO / "protocol" / "fine_tuning_design.md"
GATE_MIN_SHARE = 0.99  # share of planned trials with five-digit mass >= 0.5 that a gate unit needs

MODEL_IDS = ("qwen3-4b", "gemma-4-e4b-it", "smollm3-3b")
OBJECTIVES = ("routing", "sft", "fixed_cost", "shuffled")
CONTROL_OBJECTIVES = ("fixed_cost", "shuffled")  # Qwen3-4B only, seed 0, inherit the routing selection
GRID_LR = (1e-4, 2e-4)
GRID_BETA = (0.1, 1.0)
GRID_STEPS = (250, 500, 1000)
MAX_ACCURACY_DROP_PP = 1.0
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
BATCH = 16
WARMUP_STEPS = 50
TAU = 1.0  # tau of the logged q of sft, whose loss does not use it; routing, fixed_cost and shuffled take tau from the tau file
TAU_OBJECTIVES = ("routing", "fixed_cost", "shuffled")  # the objectives that use tau
TAU_SCHEMA = "part3_b.tau/1"
LAMBDA_M = 1.0  # the weight of each of the two digit-mass terms; the same for every objective, candidate and seed; not searched and not tied to beta
DECISION_SET, PANSWER_SET = 5, 4  # size of the digit set S of the mass: the five digits of a decision prompt; the four answer digits "1".."4" of a P-answer prompt
TIMING_FIRST_STABLE_STEP = 6  # the stable seconds per step is the median over the steps 6..N
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_MICRO_BATCH = 8
FIXED_COST_W = 0.20
DIGITS = ("1", "2", "3", "4", "5")
CHECK2_COMPUTATIONS = {  # the three computations of check 2, on the 4 fixed prompts and with the item's own labels
    "A": "the merged bf16 checkpoint, read back from disk and upcast to float32",
    "B": "the adapter-on model in the training numerics: bf16 base weights, float32 LoRA, bf16 autocast, eval mode, on the device of --device",
    "C": "the adapter-on model in float32: the base weights upcast, the LoRA in float32",
}
ADAPTER_PREFIX = "adapter_step"
EXIT_NO_SELECTABLE = 3
EXIT_MERGE_REJECTED = 4

# modules that receive LoRA: attention and MLP projections of the language-model decoder layers only. For Gemma-4 the decoder sits under the multimodal wrapper
# (``model.language_model.layers``); the vision and audio towers have modules with the same short names, so the match is on the full module name.
LORA_NAME_RE = re.compile(r"^(?:model\.language_model|model)\.layers\.\d+\.(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)$")

# the fixed support of the training conditions
TRAIN_A_H = tuple(Decimal(i) / 100 for i in list(range(55, 75)) + list(range(86, 100)))
TRAIN_C0 = (Decimal("0.00"), Decimal("0.10"), Decimal("0.20"))
TRAIN_W_RANGE = (Decimal("-0.20"), Decimal("0.40"))


class TrainPlanError(RuntimeError):
    """The training data or settings violate a rule of the training protocol."""


# ---------------------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------------------

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def sig(x: float, digits: int = 9) -> float:
    """``x`` rounded to ``digits`` significant digits. The step log holds masses and mass terms that can be 1e-9 or 1e-7; rounding to decimals would turn them into 0."""
    return float(f"{x:.{digits}g}")


def write_json(path: str | Path, obj: Any) -> None:
    """Atomic JSON write that refuses NaN and infinity."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, allow_nan=False)
        fh.write("\n")
    os.replace(tmp, path)


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def assert_off_root(path: str | Path) -> Path:
    from nmi.paths import assert_off_root_partition

    return assert_off_root_partition(path)


def resolve_device(spec: str) -> torch.device:
    """``cpu`` (the default of every command) or an explicit ``cuda:N``. A bare ``cuda`` is refused: a GPU is never picked silently."""
    text = str(spec).strip().lower()
    if text == "cpu":
        return torch.device("cpu")
    if re.fullmatch(r"cuda:\d+", text):
        return torch.device(text)
    raise TrainPlanError(f"--device must be 'cpu' or an explicit 'cuda:N' (a bare 'cuda' would pick a GPU silently), got {spec!r}")


def plan_visibility(device: torch.device, env_value: str | None) -> tuple[str, torch.device]:
    """The ``CUDA_VISIBLE_DEVICES`` to set and the device index to use inside the process.

    cpu: nothing is visible. cuda:N with the variable unset: the process is shown ONLY card N and uses it as cuda:0, because ``bitsandbytes`` (imported by ``peft``) queries and so initialises every visible
    card, which would touch card 0 on a shared machine even when card 2 was asked for. cuda:N with the variable already set: the caller chose the visible cards, N indexes that list."""
    if device.type == "cpu":
        return "", device
    if env_value is None:
        return str(device.index), torch.device("cuda:0")
    return env_value, device


def prepare_device(device: torch.device) -> torch.device:
    """Settle which GPU this process may see before ``peft``, ``bitsandbytes`` or ``transformers`` are imported. CPU run: no GPU is visible, so no library can create a CUDA context behind the scenes.
    GPU run: only the requested card is visible and it becomes the current device. Returns the device to use inside the process."""
    if torch.cuda.is_initialized():
        raise TrainPlanError("CUDA is already initialised in this process; the device has to be settled before anything touches it")
    env, local = plan_visibility(device, os.environ.get("CUDA_VISIBLE_DEVICES"))
    os.environ["CUDA_VISIBLE_DEVICES"] = env
    if local.type == "cpu":
        return local
    if not torch.cuda.is_available() or local.index >= torch.cuda.device_count():
        raise TrainPlanError(f"{device} is not available: {torch.cuda.device_count()} CUDA devices are visible to this process (CUDA_VISIBLE_DEVICES={env!r})")
    torch.cuda.set_device(local)
    return local


def net_value(cond: Any) -> float:
    """A = a_H - c0 - w of the displayed two-decimal terms, computed exactly (decimal arithmetic) and then converted to float."""
    q = lambda x: Decimal(f"{float(x):.2f}")  # noqa: E731
    return float(q(cond.a_h) - q(cond.c0) - q(cond.w))


def two_decimals(x: float, name: str = "value") -> Decimal:
    d = Decimal(str(x))
    if d != d.quantize(Decimal("0.01")):
        raise TrainPlanError(f"{name}={x!r} is not on the two-decimal grid")
    return d.quantize(Decimal("0.01"))


def model_entry(model_id: str, config_path: str | Path = CONFIG_PATH) -> dict:
    import yaml

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    for entry in cfg["models"]["devices"]:
        if entry["id"] == model_id:
            return dict(entry)
    raise TrainPlanError(f"model {model_id!r} is not in {config_path}")


# ---------------------------------------------------------------------------------------
# data: rows, transforms, batches
# ---------------------------------------------------------------------------------------

@dataclass
class SplitData:
    """Items, frozen z0 and the condition plan of one split (``train`` or ``val``)."""

    items: dict[str, Any]  # item_key -> item (an ``nmi.data.registry.McItem``)
    z0: dict[str, float]  # item_key -> frozen calibrated confidence
    conditions: list[tuple[str, Any]]  # (item_key, condition) in plan order; a condition has a_h, c0, w, frame, domain, mapping_id, wait_seconds (``Cond`` of the prompt module or ``Condition``)
    source: dict = field(default_factory=dict)  # provenance (paths -> sha256, settings), no text
    raw_rows: list[dict] | None = None  # the plan rows of the prompt module, kept for cross-checks


@dataclass(frozen=True)
class Row:
    """One (item, condition) training example. ``cond`` holds the row's own terms; the loss uses ``A`` of these."""

    item: Any
    cond: Any
    z0: float

    @property
    def item_key(self) -> str:
        return self.item.item_key

    @property
    def A(self) -> float:
        return net_value(self.cond)


def build_rows(split: SplitData) -> list[Row]:
    rows = []
    for key, cond in split.conditions:
        if key not in split.items or key not in split.z0:
            raise TrainPlanError("a planned condition refers to an item without text or without z0")
        rows.append(Row(split.items[key], cond, float(split.z0[key])))
    return rows


def to_fixed_cost(rows: Sequence[Row], w: float = FIXED_COST_W) -> list[Row]:
    """``fixed_cost`` control: every condition has the same w; a_H, c0 and mapping stay as drawn for ``train``."""
    return [dataclasses.replace(r, cond=dataclasses.replace(r.cond, w=w)) for r in rows]


def rotate_terms(conds: Sequence[Any]) -> list[Any]:
    """``shuffled`` control: prompt i displays the (a_H, c0, w) of prompt i + 1 mod n; everything else of the condition stays with the row."""
    n = len(conds)
    if n < 2:
        raise TrainPlanError("rotating the terms needs at least two prompts in the batch")
    return [dataclasses.replace(c, a_h=conds[(i + 1) % n].a_h, c0=conds[(i + 1) % n].c0, w=conds[(i + 1) % n].w) for i, c in enumerate(conds)]


def check_training_support(rows: Sequence[Row], objective: str) -> None:
    """Assert that every condition lies in the training support, after the objective's transform."""
    grid_a = set(TRAIN_A_H)
    for r in rows:
        c = r.cond
        a, c0, w = two_decimals(c.a_h, "a_H"), two_decimals(c.c0, "c0"), two_decimals(c.w, "w")
        if c.frame != "neutral" or c.domain != "id" or c.wait_seconds is not None or c.mapping_id not in (0, 1) or getattr(c, "variant", "base") != "base" or getattr(c, "x", None) is not None:
            raise TrainPlanError("a training condition is not the neutral base prompt of the id domain with mapping in {0, 1}")
        if a not in grid_a:
            raise TrainPlanError(f"a_H={a} is outside the training support (0.55..0.99 without 0.75..0.85)")
        if c0 not in TRAIN_C0:
            raise TrainPlanError(f"c0={c0} is not in {{0, 0.10, 0.20}}")
        if not TRAIN_W_RANGE[0] <= w <= TRAIN_W_RANGE[1]:
            raise TrainPlanError(f"w={w} is outside [-0.20, 0.40]")
        if c0 == Decimal("0.10") and w > Decimal("0.20"):
            raise TrainPlanError("c0 = 0.10 occurs with w > 0.20")
        if objective == "fixed_cost" and w != two_decimals(FIXED_COST_W):
            raise TrainPlanError("fixed_cost needs w = 0.20 in every condition")
        if not 0.25 - 1e-6 <= r.z0 <= 1.0 + 1e-9:
            raise TrainPlanError(f"z0={r.z0} is outside [0.25, 1] (a maximum over four options)")


class BatchStream:
    """Batches of row indices drawn with the seed: epoch e is ``permutation`` of ``SeedSequence([seed, e])``; a step takes the next ``batch`` positions of the endless stream."""

    def __init__(self, n: int, batch: int, seed: int):
        if n < 1 or batch < 1:
            raise ValueError("n and batch must be positive")
        self.n, self.batch, self.seed = n, batch, seed
        self._perm_epoch = -1
        self._perm: np.ndarray | None = None

    def _epoch_perm(self, epoch: int) -> np.ndarray:
        if epoch != self._perm_epoch:
            self._perm = np.random.default_rng(np.random.SeedSequence([self.seed, epoch])).permutation(self.n)
            self._perm_epoch = epoch
        assert self._perm is not None
        return self._perm

    def indices(self, step0: int) -> np.ndarray:
        """Row indices of the 0-based step ``step0``."""
        positions = np.arange(step0 * self.batch, (step0 + 1) * self.batch)
        return np.array([self._epoch_perm(int(p) // self.n)[int(p) % self.n] for p in positions], dtype=np.int64)


def lr_factor(step: int, warmup: int = WARMUP_STEPS) -> float:
    """Linear warmup over ``warmup`` steps (the 1-based step t gives t / warmup), constant afterwards."""
    return min(1.0, step / warmup) if warmup > 0 else 1.0


# ---------------------------------------------------------------------------------------
# rendering: one tokenised prompt per (item, displayed condition)
# ---------------------------------------------------------------------------------------

class Renderer:
    """Token ids of the decision and P-answer prompts, cached. All text comes from the prompt module; nothing is printed or stored."""

    def __init__(self, prompts: Any, tokenizer: Any, model_id: str):
        self.prompts, self.tokenizer, self.model_id = prompts, tokenizer, model_id
        self._dec: dict[tuple, tuple[np.ndarray, tuple[str, ...], str]] = {}
        self._pa: dict[str, np.ndarray] = {}

    @staticmethod
    def _cond_key(item_key: str, c: Any) -> tuple:
        return (item_key, c.domain, c.a_h, c.c0, c.w, c.frame, c.mapping_id, c.wait_seconds, getattr(c, "variant", "base"), getattr(c, "x", None))

    def decision(self, item: Any, cond: Any) -> tuple[np.ndarray, tuple[str, ...], str]:
        key = self._cond_key(item.item_key, cond)
        hit = self._dec.get(key)
        if hit is None:
            d = self.prompts.decision_prompt(item, cond)
            labels, help_label = tuple(str(x) for x in d["option_labels"]), str(d["help_label"])
            if len(labels) != 4 or sorted(labels + (help_label,)) != list(DIGITS):
                raise TrainPlanError("a decision prompt must use the five digits 1..5, four answers and one help label")
            ids = np.asarray(self.prompts.render_for_model(self.tokenizer, d["text"], self.model_id), dtype=np.int32)
            if ids.ndim != 1 or len(ids) == 0:
                raise TrainPlanError("render_for_model returned no tokens")
            hit = (ids, labels, help_label)
            self._dec[key] = hit
        return hit

    def panswer(self, item: Any, domain: str = "id") -> np.ndarray:
        hit = self._pa.get(item.item_key)
        if hit is None:
            ids = np.asarray(self.prompts.render_for_model(self.tokenizer, self.prompts.panswer_prompt(item, domain), self.model_id), dtype=np.int32)
            if ids.ndim != 1 or len(ids) == 0:
                raise TrainPlanError("render_for_model returned no tokens")
            hit = self._pa[item.item_key] = ids
        return hit


@dataclass
class Batch:
    dec_ids: list[np.ndarray]
    pa_ids: list[np.ndarray]
    help_idx: torch.Tensor  # (B,) index 0..4 of the help digit
    ans_idx: torch.Tensor  # (B, 4) indices 0..4 of the item's four answer digits, in option order
    A: torch.Tensor  # (B,) float64
    z0: torch.Tensor  # (B,) float64
    gt_idx: torch.Tensor  # (B,) index 0..3 of the correct P-answer option

    def __len__(self) -> int:
        return len(self.dec_ids)


def make_batch(rows: Sequence[Row], shown: Sequence[Any], renderer: Renderer) -> Batch:
    """``rows`` are the batch rows, ``shown`` the condition each decision prompt displays (the row's own, or the rotated terms)."""
    dec, pa, h, a = [], [], [], []
    for r, c in zip(rows, shown):
        ids, labels, help_label = renderer.decision(r.item, c)
        dec.append(ids)
        pa.append(renderer.panswer(r.item))
        h.append(int(help_label) - 1)
        a.append([int(x) - 1 for x in labels])
    return Batch(
        dec_ids=dec,
        pa_ids=pa,
        help_idx=torch.tensor(h, dtype=torch.long),
        ans_idx=torch.tensor(a, dtype=torch.long),
        A=torch.tensor([r.A for r in rows], dtype=torch.float64),
        z0=torch.tensor([r.z0 for r in rows], dtype=torch.float64),
        gt_idx=torch.tensor([int(r.item.gt_position) - 1 for r in rows], dtype=torch.long),
    )


# ---------------------------------------------------------------------------------------
# the objectives
# ---------------------------------------------------------------------------------------

def margin_terms(lp5: torch.Tensor, help_idx: torch.Tensor, ans_idx: torch.Tensor, A: torch.Tensor, z0: torch.Tensor, tau: float = TAU):
    """m = l_help - max over the row's own four answer labels; q = sigmoid(m / tau); u = q A + (1 - q) z0. ``lp5`` is (B, 5), the log-probabilities of the digits 1..5."""
    l_help = lp5.gather(1, help_idx.unsqueeze(1)).squeeze(1)
    l_ans = lp5.gather(1, ans_idx).max(dim=1).values
    m = l_help - l_ans
    q = torch.sigmoid(m / tau)
    return m, q, q * A.to(q.dtype) + (1.0 - q) * z0.to(q.dtype)


def kl_to_reference(sel_policy: torch.Tensor, sel_reference: torch.Tensor) -> torch.Tensor:
    """KL(P || P_ref) per row for two (B, 4) blocks of digit log-probabilities, each renormalised over its four labels."""
    lp = torch.log_softmax(sel_policy, dim=-1)
    lr_ = torch.log_softmax(sel_reference, dim=-1)
    return (lp.exp() * (lp - lr_)).sum(dim=-1)


def sft_target_index(ref_lp5: torch.Tensor, help_idx: torch.Tensor, ans_idx: torch.Tensor, A: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
    """Target digit index of the SFT control: the help label if A > z0 (compared in float64), else the reference's top answer label; a tie goes to the lowest label."""
    vals = ref_lp5.gather(1, ans_idx)
    tied = vals == vals.max(dim=1, keepdim=True).values
    top = torch.where(tied, ans_idx, torch.full_like(ans_idx, len(DIGITS))).min(dim=1).values
    return torch.where(A.to(torch.float64) > z0.to(torch.float64), help_idx, top)


@dataclass
class Readout:
    """One forward pass at the readout position, all float32: the log-probabilities of the digits "1".."5" (log-softmax over the full vocabulary) and, for the digit set S of the prompt type,
    log M and log(1 - M), where M is the total probability of S."""

    lp5: torch.Tensor  # (B, 5)
    log_m: torch.Tensor  # (B,)
    log_1m: torch.Tensor  # (B,)

    @property
    def mass(self) -> torch.Tensor:
        return self.log_m.exp()


def readout_from_logits(logits: torch.Tensor, digit_ids: torch.Tensor, n_set: int) -> Readout:
    """log M = logsumexp(l_S) - logsumexp(l_V) and log(1 - M) = logsumexp(l_{V minus S}) - logsumexp(l_V), both from the logits in float32. 1 - M is never formed from M, because M
    can round to 1. S is the first ``n_set`` digits: the five digits of a decision prompt, the four answer digits of a P-answer prompt."""
    lp = torch.log_softmax(logits.float(), dim=-1)  # l - logsumexp(l_V), so that a logsumexp over a subset of lp is the formula above
    in_set = torch.zeros(lp.shape[-1], dtype=torch.bool, device=lp.device)
    in_set[digit_ids[:n_set]] = True
    return Readout(lp.index_select(1, digit_ids), torch.logsumexp(lp[:, in_set], dim=-1), torch.logsumexp(lp.masked_fill(in_set, float("-inf")), dim=-1))


def mass_kl(ref: Readout, pol: Readout) -> torch.Tensor:
    """L_M = KL(Bern(M_ref) || Bern(M_theta)) per row = M_ref (log M_ref - log M_theta) + (1 - M_ref) (log(1 - M_ref) - log(1 - M_theta)). The reference comes first, so that the gradient
    with respect to the aggregate log-odds is M_theta - M_ref and the term still pulls when the mass collapses. 0 log 0 = 0."""
    w1, w0 = ref.log_m.exp(), ref.log_1m.exp()  # M_ref and 1 - M_ref, both from logsumexp
    d1 = torch.where(w1 > 0, ref.log_m - pol.log_m, torch.zeros_like(w1))
    d0 = torch.where(w0 > 0, ref.log_1m - pol.log_1m, torch.zeros_like(w0))
    return w1 * d1 + w0 * d0


def row_terms(objective: str, pol_dec: Readout, ref_dec: Readout, pol_pa: Readout, ref_pa: Readout, batch_like: Any, tau: float = TAU) -> dict[str, torch.Tensor]:
    """Per-row ingredients of the loss. ``pol_*`` and ``ref_*`` are the readouts of the decision and the P-answer prompts, policy (adapter on) and frozen reference (adapter off)."""
    m, q, util = margin_terms(pol_dec.lp5, batch_like.help_idx, batch_like.ans_idx, batch_like.A, batch_like.z0, tau)
    out = {
        "m": m,
        "q": q,
        "util": util,
        "kl1": kl_to_reference(pol_dec.lp5.gather(1, batch_like.ans_idx), ref_dec.lp5.gather(1, batch_like.ans_idx)),
        "kl2": kl_to_reference(pol_pa.lp5[:, :4], ref_pa.lp5[:, :4]),
        "lm_dec": mass_kl(ref_dec, pol_dec),
        "lm_pa": mass_kl(ref_pa, pol_pa),
    }
    if objective == "sft":
        target = sft_target_index(ref_dec.lp5, batch_like.help_idx, batch_like.ans_idx, batch_like.A, batch_like.z0)
        out["target"] = target
        out["ce"] = -pol_dec.lp5.gather(1, target.unsqueeze(1)).squeeze(1)  # cross-entropy over the full vocabulary (the digit log-probs are full-vocabulary log-softmax entries)
    return out


def row_loss(objective: str, terms: Mapping[str, torch.Tensor], beta: float, lambda_m: float = LAMBDA_M) -> torch.Tensor:
    """Per row. routing, fixed_cost, shuffled: -(q A + (1 - q) z0) + beta (KL1 + KL2) + lambda_M (L_M,decision + L_M,P-answer). sft: CE + beta KL2 + lambda_M (L_M,decision + L_M,P-answer)."""
    mass = lambda_m * (terms["lm_dec"] + terms["lm_pa"])
    if objective == "sft":
        return terms["ce"] + beta * terms["kl2"] + mass
    return -terms["util"] + beta * (terms["kl1"] + terms["kl2"]) + mass


# ---------------------------------------------------------------------------------------
# model: loading, LoRA, forward
# ---------------------------------------------------------------------------------------

def lora_target_names(model: torch.nn.Module) -> list[str]:
    names = [n for n, mod in model.named_modules() if LORA_NAME_RE.match(n) and isinstance(mod, torch.nn.Linear)]
    if not names:
        raise TrainPlanError("no LoRA target module found; the model does not follow the expected decoder layout")
    return names


def pad_token_id(tokenizer: Any) -> int:
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is None:
        pad = getattr(tokenizer, "eos_token_id", None)
    return int(pad) if pad is not None else 0


def load_base_model(path: str | Path, dtype: torch.dtype, device: torch.device, attn_implementation: str = "sdpa", gradient_checkpointing: bool = False):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(str(path), dtype=dtype, attn_implementation=attn_implementation, local_files_only=True)
    model.to(device)
    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def add_lora(model: torch.nn.Module, seed: int):
    """LoRA rank 16, alpha 32, dropout 0.05 on the language model's attention and MLP projections; the adapter weights are float32."""
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(seed)
    cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none", target_modules=lora_target_names(model), task_type="CAUSAL_LM")
    peft_model = get_peft_model(model, cfg)
    params = [p for p in peft_model.parameters() if p.requires_grad]
    if not params or any(p.dtype != torch.float32 for p in params):
        raise TrainPlanError("the LoRA parameters must be float32")
    return peft_model


def last_logits(model: torch.nn.Module, id_lists: Sequence[Sequence[int]], pad_id: int, device: torch.device, amp: bool) -> torch.Tensor:
    """Float32 logits (B, V) of the last input position. Prompts are padded on the left by hand, with positions counted from the first real token."""
    from nmi.readout.hf_backend import left_pad

    input_ids, attention, position = left_pad([list(map(int, x)) for x in id_lists], pad_id)
    kwargs = dict(
        input_ids=torch.from_numpy(input_ids).to(device),
        attention_mask=torch.from_numpy(attention).to(device),
        position_ids=torch.from_numpy(position).to(device),
        use_cache=False,
        logits_to_keep=1,
    )
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
        out = model(**kwargs)
    return out.logits[:, -1, :].float()


def digit_logprobs(logits: torch.Tensor, digit_ids: torch.Tensor) -> torch.Tensor:
    """(B, 5) exact log-probabilities of the digit tokens: log-softmax over the full vocabulary, then the five columns."""
    return torch.log_softmax(logits, dim=-1).index_select(1, digit_ids)


@dataclass
class StepContext:
    pad_id: int
    device: torch.device
    amp: bool
    digit_ids: torch.Tensor
    micro: int
    tau: float = TAU  # the tau of the model for routing, fixed_cost and shuffled
    lambda_m: float = LAMBDA_M  # fixed, not searched


def forward_backward(model, batch: Batch, objective: str, beta: float, ctx: StepContext, backward: bool = True) -> dict[str, float]:
    """One logical step: the reference (adapter disabled, no gradient) and the policy forward on every micro-batch, and the backward of ``sum(row losses) / B``."""
    n = len(batch)
    acc: dict[str, float] = collections.defaultdict(float)
    mins = {"M_dec": 1.0, "M_pa": 1.0, "mass5_pa": 1.0}
    m_all: list[torch.Tensor] = []
    # Rows of similar length share a micro-batch (less left padding). The loss is a sum over rows, so the order changes nothing but the padding.
    order = sorted(range(n), key=lambda i: len(batch.dec_ids[i]) + len(batch.pa_ids[i]))
    for lo in range(0, n, ctx.micro):
        take = order[lo:lo + ctx.micro]
        sel = torch.tensor(take, dtype=torch.long)
        part = Batch([batch.dec_ids[i] for i in take], [batch.pa_ids[i] for i in take], batch.help_idx[sel], batch.ans_idx[sel], batch.A[sel], batch.z0[sel], batch.gt_idx[sel])
        dev = ctx.device
        ids = ctx.digit_ids.to(dev)
        with torch.no_grad(), model.disable_adapter():
            ref_dec = readout_from_logits(last_logits(model, part.dec_ids, ctx.pad_id, dev, ctx.amp), ids, DECISION_SET)
            ref_pa = readout_from_logits(last_logits(model, part.pa_ids, ctx.pad_id, dev, ctx.amp), ids, PANSWER_SET)
        pol_dec = readout_from_logits(last_logits(model, part.dec_ids, ctx.pad_id, dev, ctx.amp), ids, DECISION_SET)
        pol_pa = readout_from_logits(last_logits(model, part.pa_ids, ctx.pad_id, dev, ctx.amp), ids, PANSWER_SET)
        on_dev = dataclasses.replace(part, help_idx=part.help_idx.to(dev), ans_idx=part.ans_idx.to(dev), A=part.A.to(dev), z0=part.z0.to(dev), gt_idx=part.gt_idx.to(dev))
        terms = row_terms(objective, pol_dec, ref_dec, pol_pa, ref_pa, on_dev, ctx.tau)
        losses = row_loss(objective, terms, beta, ctx.lambda_m)
        if backward:
            (losses.sum() / n).backward()
        with torch.no_grad():
            acc["loss"] += float(losses.sum())
            acc["main"] += float((terms["ce"] if objective == "sft" else -terms["util"]).sum())
            for k in ("kl1", "kl2", "q", "m", "util", "lm_dec", "lm_pa"):
                acc[k] += float(terms[k].sum())
            acc["ask"] += float((terms["m"] > 0).sum())
            m_all.append(terms["m"].detach().cpu())
            acc["slope"] += float((terms["q"] * (1 - terms["q"])).sum()) / ctx.tau  # d q / d m: the weight of the routing gradient of each row
            mass_dec, mass_pa, mass5_pa = pol_dec.mass, pol_pa.mass, pol_pa.lp5.exp().sum(1)  # M of the decision prompt (five digits), of the P-answer prompt (four answer digits), five-digit mass of the P-answer prompt (the gate)
            acc["M_dec"] += float(mass_dec.sum())
            acc["M_pa"] += float(mass_pa.sum())
            acc["mass5_pa"] += float(mass5_pa.sum())
            for key, value in (("M_dec", mass_dec), ("M_pa", mass_pa), ("mass5_pa", mass5_pa)):
                mins[key] = min(mins[key], float(value.min()))
            gt = on_dev.gt_idx
            acc["pa_acc"] += float((pol_pa.lp5[:, :4].argmax(1) == gt).sum())
            acc["pa_acc_ref"] += float((ref_pa.lp5[:, :4].argmax(1) == gt).sum())
            if objective == "sft":
                acc["target_help"] += float((terms["target"] == on_dev.help_idx).sum())
    out = {k: v / n for k, v in acc.items()}
    m_abs = torch.cat(m_all).abs()
    out["m_abs_median"], out["unsat_frac"] = float(m_abs.median()), float((m_abs / ctx.tau <= 4.0).float().mean())  # rows with |m| / tau > 4 have q (1 - q) < 0.018 and hardly any routing gradient
    out["M_dec_min"], out["M_pa_min"], out["mass5_pa_min"] = mins["M_dec"], mins["M_pa"], mins["mass5_pa"]
    return out


# ---------------------------------------------------------------------------------------
# data from the prompt module and synthetic data for the CPU dry run
# ---------------------------------------------------------------------------------------

@dataclass
class Deps:
    """What the commands need from outside: the prompt module, a loader of the splits and the keys the training data must not contain."""

    prompts: Any
    load_split: Callable[[str, str], SplitData]  # (model_id, "train" | "val") -> SplitData
    forbidden: Callable[[], dict[str, set[str]]]
    synthetic: bool = False


def load_split_from_b1(P: Any, model_id: str, which: str) -> SplitData:
    """The training or validation data of one model, as defined by the prompt module. ``which`` is "train" (800 items, 16 conditions each) or "val" (200 items, 8 each).

    Items: the ``mmlu_redux_cal`` pool split by the ``split_cal`` of the prompt module (the 200 smallest ``sha256("part3-b-split|" + key)`` are the validation set). Only the calibration pool is opened here; the
    evaluation and ARC pools are never loaded by the training script (their keys are compared in ``train``). z0: ``load_z0`` of the prompt module (the stored calibration runs, fixed temperature). Plan:
    ``plan_for_set`` of the prompt module. Every row's own payoffs are cross-checked against this script's exact A and the stored z0."""
    if which not in ("train", "val"):
        raise ValueError("which must be train or val")
    from nmi.data import mmlu_redux
    from nmi.data.registry import SplitRegistry
    from nmi.mechanism.data import MMLU_ROOT

    registry = SplitRegistry.load()
    cal = mmlu_redux.load_pool(registry, "mmlu_redux_cal", MMLU_ROOT)
    train_items, val_items = P.split_cal(cal)
    items = {"train": train_items, "val": val_items}
    z0, z0_info = P.load_z0(model_id, "mmlu_redux_cal")
    rows = P.plan_for_set(model_id, which, items, z0)
    conds = [(r["item_key"], P.as_cond(r)) for r in rows]
    for r, (_, c) in zip(rows, conds):
        if net_value(c) != r["payoff_help"] or z0[r["item_key"]] != r["payoff_self"] or z0[r["item_key"]] != r["z0"]:
            raise TrainPlanError("a plan row disagrees with the exact net value or the frozen z0")
    chosen = items[which]
    return SplitData(items={i.item_key: i for i in chosen}, z0={i.item_key: z0[i.item_key] for i in chosen}, conditions=conds, raw_rows=rows,
                     source={"z0": z0_info, "split": "part3-b-split|", "pool": "mmlu_redux_cal", "set": which, "seed": P.SEEDS[which], "per_item": P.PER_ITEM[which]})


def real_forbidden_keys() -> dict[str, set[str]]:
    """Keys that no training or validation item may carry: the evaluation pool, the holdout pool, the holdout ledger (keys only) and the ARC test split (keys only)."""
    from nmi.data.registry import SplitRegistry

    reg = SplitRegistry.load()
    out: dict[str, set[str]] = {"mmlu_redux_eval": set(reg.keys("mmlu_redux_eval")), "mmlu_redux_holdout": set(reg.keys("mmlu_redux_holdout"))}
    from nmi.provenance import note_skipped

    ledger_keys: set[str] = set()
    ledger_path = REPO / "results" / "holdout_ledger.jsonl"
    if ledger_path.is_file():
        for line in ledger_path.read_text().splitlines():
            key = json.loads(line).get("item_key")
            if key:
                ledger_keys.add(key)
    else:  # the ledger of the study is not part of this release
        note_skipped(ledger_path, "holdout ledger")
    out["holdout_ledger"] = ledger_keys
    arc_keys_path = REPO / "results" / "mechanism_20261016" / "data" / "test.keys"
    if arc_keys_path.is_file():
        out["arc_test"] = set(arc_keys_path.read_text().split())
    else:  # the frozen key list of the ARC test split is not part of this release
        note_skipped(arc_keys_path, "ARC test key list")
        out["arc_test"] = set()
    return out


def real_deps() -> Deps:
    P = importlib.import_module("_part3_b_prompts")
    return Deps(prompts=P, load_split=lambda model_id, which: load_split_from_b1(P, model_id, which), forbidden=real_forbidden_keys)


def assert_pairwise_disjoint(sets: Mapping[str, set[str]], own: Sequence[str] = ("train", "val")) -> None:
    """The training and validation key sets are disjoint from each other and from every other named set (the forbidden sets may overlap among themselves, e.g. holdout pool and ledger)."""
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if a not in own and b not in own:
                continue
            shared = sets[a] & sets[b]
            if shared:
                raise TrainPlanError(f"{a} and {b} share {len(shared)} item keys")


def synthetic_split(n_items: int, per_item: int, seed: int, tag: str = "syn", question_marker: str = "Which value was recorded for entity") -> SplitData:
    """Fake items, z0 and conditions with the shape of the training distribution, for the tests and the CPU dry run only. Not the real sampler of the prompt module."""
    from nmi.data.registry import McItem
    from nmi.mechanism.prompts import Condition

    rng = np.random.default_rng([seed, 17])
    items, z0, conds = {}, {}, []
    a_grid = [float(x) for x in TRAIN_A_H]
    for i in range(n_items):
        key = sha256_bytes(f"{tag}|{seed}|{i}".encode())
        choices = tuple(f"option {j} of {tag}{i:04d}" for j in range(1, 5))
        gt = 1 + i % 4
        items[key] = McItem(key, "synthetic", "synthetic", f"synthetic#{i}", None, f"{question_marker} {tag}{i:04d}?", choices, gt, choices[gt - 1], None, i)
        z0[key] = float(np.round(rng.uniform(0.3, 0.99), 4))
    for key in sorted(items):
        for j in range(per_item):
            c0 = float(rng.choice([0.0, 0.1, 0.2]))
            w_hi = 20 if abs(c0 - 0.1) < 1e-9 else 40
            w = round(int(rng.integers(-20, w_hi + 1)) / 100, 2)
            a_h = float(rng.choice(a_grid))
            conds.append((key, Condition(f"{tag}_{j:02d}", "id", round(a_h, 2), round(c0, 2), w, "neutral", int(rng.integers(0, 2)))))
    return SplitData(items=items, z0=z0, conditions=conds, source={"synthetic": True, "seed": seed})


def synthetic_deps(prompts: Any, n_items: int = 24, per_item: int = 4, seed: int = 0) -> Deps:
    def load(model_id: str, which: str) -> SplitData:
        return synthetic_split(n_items if which == "train" else max(8, n_items // 3), per_item, seed + (0 if which == "train" else 1), tag="syn" + which)

    return Deps(prompts=prompts, load_split=load, forbidden=lambda: {}, synthetic=True)


# ---------------------------------------------------------------------------------------
# manifest helpers
# ---------------------------------------------------------------------------------------

def plan_sha256(rows: Sequence[Row]) -> str:
    h = hashlib.sha256()
    for r in rows:
        c = r.cond
        h.update(json.dumps([r.item_key, f"{Decimal(str(c.a_h)):.2f}", f"{Decimal(str(c.c0)):.2f}", f"{Decimal(str(c.w)):.2f}", c.frame, c.domain, int(c.mapping_id)]).encode() + b"\n")
    return h.hexdigest()


def z0_sha256(rows: Sequence[Row]) -> str:
    h = hashlib.sha256()
    for key, z in sorted({r.item_key: r.z0 for r in rows}.items()):
        h.update(f"{key} {z!r}\n".encode())
    return h.hexdigest()


def prompt_ids_sha256(rows: Sequence[Row], renderer: Renderer) -> str:
    """sha256 over the token ids of every decision prompt (plan order) and every P-answer prompt (sorted by item key)."""
    h = hashlib.sha256()

    def feed(ids: np.ndarray) -> None:
        h.update(np.int64(len(ids)).tobytes())
        h.update(np.asarray(ids, dtype=np.int32).tobytes())

    for r in rows:
        feed(renderer.decision(r.item, r.cond)[0])
    items = {r.item_key: r.item for r in rows}
    for key in sorted(items):
        feed(renderer.panswer(items[key]))
    return h.hexdigest()


def decision_ids_sha256(rows: Sequence[Row], renderer: Renderer) -> str:
    """sha256 over the rendered token ids of the decision prompts of ``rows`` in plan order: the prompt hash of the tau file."""
    h = hashlib.sha256()
    for r in rows:
        ids = renderer.decision(r.item, r.cond)[0]
        h.update(np.int64(len(ids)).tobytes())
        h.update(np.asarray(ids, dtype=np.int32).tobytes())
    return h.hexdigest()


def model_identity(model_path: Path) -> dict[str, str]:
    """Hashes that identify the frozen base model: config.json and the weight index (the index file of a sharded checkpoint, else the safetensors header of the single file)."""
    index = model_path / "model.safetensors.index.json"
    if index.is_file():
        kind, digest = "model.safetensors.index.json", sha256_file(index)
    else:
        with open(model_path / "model.safetensors", "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            digest = sha256_bytes(fh.read(n))
        kind = "safetensors header of model.safetensors"
    return {"config_json_sha256": sha256_file(model_path / "config.json"), "weight_index_sha256": digest, "weight_index_kind": kind}


def hf_blob_name(path: Path) -> str | None:
    """The content-addressed name of a Hugging Face cache file, if ``path`` is a snapshot entry: ``snapshots/<rev>/<file>`` is a link to ``../../blobs/<name>``, and for a large (LFS) file that name is
    the sha256 of the content (the git blob id of a small file has 40 digits and is not a content hash). Only the DIRECT target of the link counts (``os.readlink``, never ``realpath``): the blob in
    the model's own ``blobs`` directory may itself be a link into a deduplicated store (``hub/blobs/<2 hex>/<another name>``), and that later hop is a storage detail whose name is not the content hash.
    None if ``path`` is not a symlink (a plain local model directory), if its direct target does not lie in a directory named ``blobs``, or if the name is not 64 hex digits."""
    p = Path(path)
    if not p.is_symlink():
        return None
    target = Path(os.readlink(p))
    if target.parent.name != "blobs":
        return None
    return target.name if re.fullmatch(r"[0-9a-f]{64}", target.name) else None


def file_sha256_checked(path: Path) -> str:
    """The sha256 over the ACTUAL BYTES of a file, never taken from a name. If the file is a snapshot entry that links directly to a Hugging Face cache blob named by a sha256 (``hf_blob_name``),
    the computed hash must equal that name; a mismatch raises (a damaged or replaced blob, or a name that was made up)."""
    digest = sha256_file(path)
    named = hf_blob_name(path)
    if named is not None and named != digest:
        raise TrainPlanError(f"{path}: the sha256 of its bytes is {digest}, but it is (a link to) a Hugging Face cache blob named {named}; the cache is damaged or the blob was replaced")
    return digest


def model_sources(model_path: Path) -> dict[str, Any]:
    """The content hashes of the base model files: sha256 over the bytes of every weight file (and of config, tokenizer and template files), computed here and nowhere copied from a file name.
    ``bytes`` is the size of a weight file. The structure is the one that the manifests and the tau file hold under ``model_files``."""
    files = {}
    for p in sorted(model_path.iterdir()):
        if not p.is_file():
            continue
        if p.suffix == ".safetensors":
            files[p.name] = {"bytes": p.stat().st_size, "sha256": file_sha256_checked(p)}
        elif p.suffix in (".json", ".jinja", ".model", ".txt"):
            files[p.name] = {"sha256": file_sha256_checked(p)}
    return files


def code_sources(extra: Sequence[Path] = ()) -> dict[str, str]:
    paths = [Path(__file__), REPO / "nmi" / "mechanism" / "prompts.py", REPO / "nmi" / "prompts.py", REPO / "nmi" / "readout" / "hf_backend.py", REPO / "nmi" / "readout" / "digits.py",
             PROMPT_BOOK_PATH, CONFIG_PATH, DESIGN_PATH, *extra]
    return {str(p): sha256_file(p) for p in paths if Path(p).is_file()}


def environment() -> dict[str, str]:
    import transformers

    out = {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__, "host": platform.node()}
    try:
        import peft

        out["peft"] = peft.__version__
    except Exception:  # pragma: no cover
        pass
    return out


# ---------------------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------------------

def parse_save_at(text: str) -> tuple[int, ...]:
    steps = tuple(sorted({int(x) for x in text.split(",") if x.strip()}))
    if not steps or steps[0] < 1:
        raise argparse.ArgumentTypeError("--save-at needs positive step numbers")
    return steps


def validate_train_args(args: argparse.Namespace) -> None:
    if args.time_only is not None:  # N steps of the formal configuration, one adapter, a throw-away output
        if args.time_only < 1:
            raise TrainPlanError("--time-only needs a positive number of steps")
        args.steps, args.save_at = args.time_only, (args.time_only,)
    if not args.allow_offgrid:
        if args.model not in MODEL_IDS:
            raise TrainPlanError(f"model {args.model!r} is not one of the frozen models {MODEL_IDS}")
        if not any(math.isclose(args.lr, x, rel_tol=1e-9) for x in GRID_LR):
            raise TrainPlanError(f"lr {args.lr} is not a frozen candidate {GRID_LR}")
        if not any(math.isclose(args.beta, x, rel_tol=1e-9) for x in GRID_BETA):
            raise TrainPlanError(f"beta {args.beta} is not a frozen candidate {GRID_BETA}")
        if args.objective in CONTROL_OBJECTIVES and (args.model != "qwen3-4b" or args.seed != 0):
            raise TrainPlanError(f"{args.objective} is run on qwen3-4b with seed 0 only")
    if args.steps < 1 or max(args.save_at) > args.steps:
        raise TrainPlanError("--save-at must lie inside 1..--steps")
    if args.batch < 2 or args.micro_batch < 1:
        raise TrainPlanError("--batch needs at least 2 rows and --micro-batch at least 1")
    if args.objective in TAU_OBJECTIVES and not args.tau_file:
        raise TrainPlanError(f"{args.objective} needs --tau-file: run the tau command of this model first")
    if args.objective not in TAU_OBJECTIVES and args.tau_file:
        raise TrainPlanError(f"{args.objective} does not use tau; --tau-file is refused")


def load_tau_file(path: str | Path, *, model_id: str, model_path: Path, model_files: Mapping[str, Any], decision_hash: str) -> dict:
    """Read a tau file and check it against this run. The model id, the CANONICAL prompt hash (the rendered token ids of the original routing train decision prompts of this model,
    ``decision_hash``, recomputed by the run from the sampler: also for fixed_cost and shuffled, whose own rewritten-prompt hash is recorded elsewhere and never compared) and the content hashes
    of every base file (``model_files``, the structure of the manifest) must match. Returns what the manifest copies."""
    p = Path(path)
    doc = read_json(p)
    if doc.get("schema") != TAU_SCHEMA:
        raise TrainPlanError(f"{p} is not a tau file of this program (schema {doc.get('schema')!r})")
    if doc["model_id"] != model_id:
        raise TrainPlanError(f"the tau file belongs to model {doc['model_id']!r}, not {model_id!r}")
    hashes = doc["hashes"]
    if hashes["decision_ids_sha256"] != decision_hash:
        raise TrainPlanError("the tau file was computed on other train decision prompts than the canonical original routing train prompts of this run (token id hashes differ)")
    if hashes["model_files"] != dict(model_files):
        differs = sorted(k for k in set(hashes["model_files"]) | set(model_files) if hashes["model_files"].get(k) != model_files.get(k))
        raise TrainPlanError(f"the tau file was computed on other base model files than this run's (they differ in {differs[:5]})")
    ident = model_identity(model_path)
    for key in ("config_json_sha256", "weight_index_sha256"):
        if hashes[key] != ident[key]:
            raise TrainPlanError(f"the tau file was computed on another base model ({key} differs)")
    tau, median = float(doc["tau"]), float(doc["median_abs_m"])
    if not (math.isfinite(tau) and tau >= 1.0 and abs(tau - max(1.0, median)) <= 1e-9 * max(1.0, tau)):
        raise TrainPlanError(f"the tau file is inconsistent: tau {tau}, median |m| {median}; the rule is tau = max(1, median)")
    return {"value": tau, "median_abs_m": median, "file": str(p.resolve()), "file_sha256": sha256_file(p), "model_id": doc["model_id"], "abs_m_quantiles": doc["abs_m_quantiles"],
            "wrong_side": doc["wrong_side"], "hashes": hashes}


def train(args: argparse.Namespace, deps: Deps) -> dict:
    device = resolve_device(args.device)  # first: nothing is read, written or loaded before the device is settled
    if deps.synthetic and device.type == "cuda":
        raise TrainPlanError("synthetic data is for the CPU dry run only")
    validate_train_args(args)
    out = assert_off_root(args.out)
    if out.exists() and any(out.iterdir()):
        raise TrainPlanError(f"{out} exists and is not empty; a run directory is never overwritten")
    requested = device
    device = prepare_device(requested)
    print(f"[device] {requested}" + (" (all GPUs hidden)" if device.type == "cpu" else f" -> {device} of CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']!r}: {torch.cuda.get_device_name(device)}"), flush=True)
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    amp = dtype == torch.bfloat16  # bf16 weights run under bf16 autocast (CPU autocast included, so that the CPU tests exercise the path of the GPU run); float32 runs without
    model_path = Path(args.model_path) if args.model_path else Path(model_entry(args.model)["path"])
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # data and its checks
    train_split = deps.load_split(args.model, "train")
    val_split = deps.load_split(args.model, "val")
    orig_rows = build_rows(train_split)  # the canonical train plan, rendered as for routing; fixed_cost trains on a rewritten copy
    rows = to_fixed_cost(orig_rows) if args.objective == "fixed_cost" else orig_rows
    check_training_support(rows, args.objective)
    n_items = len(train_split.items)
    if not deps.synthetic and (n_items != 800 or len(val_split.items) != 200 or len(rows) != 16 * n_items):
        raise TrainPlanError(f"expected 800 training items with 16 conditions each and 200 validation items, got {n_items} items, {len(rows)} rows, {len(val_split.items)} validation items")
    per_item = collections.Counter(r.item_key for r in rows)
    if len(set(per_item.values())) != 1:
        raise TrainPlanError("every training item must have the same number of conditions")
    forbidden = deps.forbidden()
    assert_pairwise_disjoint({"train": set(train_split.items), "val": set(val_split.items), **forbidden})

    # tokenizer, renderer, digit ids
    from transformers import AutoTokenizer

    from nmi.readout.digits import resolve_digit_token_ids

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    digit_ids = torch.tensor(resolve_digit_token_ids(tokenizer, DIGITS), dtype=torch.long)
    renderer = Renderer(deps.prompts, tokenizer, args.model)
    plan_hash, train_plan_hash, z0_hash = plan_sha256(rows), plan_sha256(orig_rows), z0_sha256(rows)
    ids_hash = prompt_ids_sha256(rows, renderer)
    canonical_hash = decision_ids_sha256(orig_rows, renderer)  # the canonical hash, the one the tau file carries
    run_hash = decision_ids_sha256(rows, renderer) if args.objective == "fixed_cost" else canonical_hash  # fixed_cost's own rewritten prompts; never compared with the tau file
    model_files = model_sources(model_path)
    tau_info = None
    if args.objective in TAU_OBJECTIVES:
        tau_info = load_tau_file(args.tau_file, model_id=args.model, model_path=model_path, model_files=model_files, decision_hash=canonical_hash)
    tau = tau_info["value"] if tau_info else TAU
    tau_text = f"tau {tau:.3f}" if tau_info else "tau -"  # sft does not use tau (its logged q uses 1)
    t_prepared = time.time()

    # model
    model = load_base_model(model_path, dtype, device, args.attn_implementation, gradient_checkpointing=not args.no_grad_checkpointing)
    n_targets = len(lora_target_names(model))
    model = add_lora(model, args.seed)
    params = [p for p in model.parameters() if p.requires_grad]
    n_lora = sum(p.numel() for p in params)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)
    t_model = time.time()

    manifest = {
        "schema": "part3_b.train/1",
        "program": PROGRAM,
        "status": "started",
        "timing_only": args.time_only is not None,
        "model_id": args.model,
        "model_path": str(model_path),
        "objective": args.objective,
        "seed": args.seed,
        "lr": args.lr,
        "beta": args.beta,
        "steps": args.steps,
        "save_at": sorted(set(args.save_at) | {args.steps}),
        "tau": tau_info,
        "settings": {
            "batch_decision": args.batch, "batch_panswer": args.batch, "micro_batch": args.micro_batch, "tau": tau if tau_info else None, "lambda_m": LAMBDA_M, "warmup_steps": args.warmup, "weight_decay": 0.0,
            "adamw_betas": [0.9, 0.999], "adamw_eps": 1e-8, "max_grad_norm": args.max_grad_norm, "lr_schedule": "linear warmup, then constant", "dtype": args.dtype, "autocast_bf16": amp,
            "gradient_checkpointing": not args.no_grad_checkpointing, "attn_implementation": args.attn_implementation, "device": str(requested), "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT, "targets": "language-model attention and MLP projections", "n_target_modules": n_targets, "n_params": n_lora},
            "loss": {
                "routing": "-mean(q A + (1-q) z0) + beta (KL1 + KL2) + lambda_M (L_M,decision + L_M,P-answer), q = sigmoid(m / tau)",
                "sft": "CE(target) + beta KL2 + lambda_M (L_M,decision + L_M,P-answer)",
                "fixed_cost": "routing with w = 0.20", "shuffled": "routing with the displayed terms rotated by one",
            }[args.objective],
            "mass_terms": "L_M = KL(Bern(M_ref) || Bern(M_theta)), M = total probability of the five digits (decision prompt) or of the four answer digits (P-answer prompt); logsumexp, never 1 - M",
            "kl1_in_loss": args.objective != "sft",
        },
        "data": {
            "source": "synthetic (CPU dry run)" if deps.synthetic else "B1 split loader",
            "n_items": n_items, "n_rows": len(rows), "conditions_per_item": next(iter(per_item.values())), "n_val_items": len(val_split.items),
            "plan_sha256": plan_hash, "train_plan_sha256": train_plan_hash, "z0_sha256": z0_hash, "prompt_ids_sha256": ids_hash, "canonical_decision_ids_sha256": canonical_hash,
            "run_decision_ids_sha256": run_hash, "split_source": train_split.source, "disjoint_from": sorted(forbidden),
        },
        "sources": {"code": code_sources([Path(deps.prompts.__file__)] if getattr(deps.prompts, "__file__", None) else []), "model_files": model_files},
        "nmi_source_sha256": None,
        "environment": environment(),
        "command": " ".join(sys.argv),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "adapters": [],
    }
    try:
        from nmi.provenance import read_git_head, source_hash

        manifest["nmi_source_sha256"] = source_hash()
        manifest["git_head"] = read_git_head().commit
    except Exception:  # pragma: no cover
        pass
    write_json(out / "manifest.json", manifest)

    stream = BatchStream(len(rows), args.batch, args.seed)
    ctx = StepContext(pad_id=pad_token_id(tokenizer), device=device, amp=amp, digit_ids=digit_ids, micro=args.micro_batch, tau=tau)
    log_path = out / "train_log.jsonl"
    last_stats: dict[str, float] = {}
    step_seconds: list[float] = []
    shown_hash, n_shown = hashlib.sha256(), 0  # the decision prompts actually shown, in training order (for shuffled these are the rotated prompts)
    with open(log_path, "w", encoding="utf-8") as log:
        for step in range(1, args.steps + 1):
            t_step = time.time()
            idx = stream.indices(step - 1)
            batch_rows = [rows[int(i)] for i in idx]
            shown = [r.cond for r in batch_rows]
            if args.objective == "shuffled":
                shown = rotate_terms(shown)
            batch = make_batch(batch_rows, shown, renderer)
            for ids_row in batch.dec_ids:
                shown_hash.update(np.int64(len(ids_row)).tobytes())
                shown_hash.update(np.asarray(ids_row, dtype=np.int32).tobytes())
            n_shown += len(batch.dec_ids)
            lr = args.lr * lr_factor(step, args.warmup)
            for group in optimizer.param_groups:
                group["lr"] = lr
            model.train()
            optimizer.zero_grad(set_to_none=True)
            stats = forward_backward(model, batch, args.objective, args.beta, ctx)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm if args.max_grad_norm > 0 else float("inf")))
            if not (math.isfinite(stats["loss"]) and math.isfinite(grad_norm)):
                raise TrainPlanError(f"non-finite loss or gradient at step {step}; the run is stopped, nothing is repaired silently")
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            sec = time.time() - t_step
            step_seconds.append(sec)
            record = {"step": step, "lr": lr, "grad_norm": grad_norm, "clipped": bool(args.max_grad_norm > 0 and grad_norm > args.max_grad_norm), "sec": round(sec, 3), **{k: sig(v) for k, v in stats.items()}}
            if device.type == "cuda":
                record["peak_gb"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 3)
            log.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            log.flush()
            last_stats = stats
            if step % 25 == 0 or step == 1 or step == args.steps:
                print(f"[{args.model} {args.objective} s{args.seed}] step {step}/{args.steps} loss {stats['loss']:.5f} q {stats['q']:.3f} kl1 {stats['kl1']:.5f} kl2 {stats['kl2']:.5f} "
                      f"LM {stats['lm_dec']:.3g}/{stats['lm_pa']:.3g} M {stats['M_dec']:.4g}/{stats['M_pa']:.4g} (min {stats['M_dec_min']:.4g}/{stats['M_pa_min']:.4g}) {tau_text} "
                      f"grad {grad_norm:.3f} {record['sec']}s" + (f" peak {record['peak_gb']}GB" if "peak_gb" in record else ""), flush=True)
            if step in manifest["save_at"]:
                t_save = time.time()
                entry = save_adapter(model, out, step, manifest)
                entry["save_seconds"] = round(time.time() - t_save, 3)
                manifest["adapters"].append(entry)
                write_json(out / "manifest.json", manifest)
    manifest["data"]["shown_decision_ids_sha256"], manifest["data"]["n_shown_decision_prompts"] = shown_hash.hexdigest(), n_shown
    manifest["status"] = "finished"
    manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest["elapsed_sec"] = round(time.time() - t0, 1)
    manifest["last_step_stats"] = {k: sig(v) for k, v in last_stats.items()}
    write_json(out / "manifest.json", manifest)
    if args.time_only is not None:
        stable = step_seconds[TIMING_FIRST_STABLE_STEP - 1:]
        timing = {
            "schema": "part3_b.timing/1", "timing_only": True, "model_id": args.model, "objective": args.objective, "seed": args.seed, "lr": args.lr, "beta": args.beta, "n_steps": len(step_seconds),
            "formal_configuration": {"batch_decision": args.batch, "batch_panswer": args.batch, "micro_batch": args.micro_batch, "gradient_checkpointing": not args.no_grad_checkpointing, "dtype": args.dtype,
                                     "autocast_bf16": amp, "device": str(requested), "reference_forward": True, "mass_terms": True, "tau": tau if tau_info else None},
            "data_and_hash_seconds": round(t_prepared - t0, 3), "model_load_seconds": round(t_model - t_prepared, 3), "first_step_seconds": round(step_seconds[0], 3),
            "initialisation_seconds": round(t_model - t_prepared + step_seconds[0], 3),
            "stable_steps": [TIMING_FIRST_STABLE_STEP, len(step_seconds)], "median_step_seconds": round(statistics.median(stable), 4) if stable else None,
            "step_seconds": [round(x, 4) for x in step_seconds], "adapter_save_seconds": manifest["adapters"][-1]["save_seconds"],
            "peak_cuda_allocated_gib": round(torch.cuda.max_memory_allocated(device) / 2**30, 3) if device.type == "cuda" else None,
            "peak_cuda_reserved_gib": round(torch.cuda.max_memory_reserved(device) / 2**30, 3) if device.type == "cuda" else None,
        }
        write_json(out / "timing.json", timing)
        manifest["timing"] = {"file": "timing.json", "median_step_seconds": timing["median_step_seconds"]}
        write_json(out / "manifest.json", manifest)
    return manifest


def save_adapter(model, out: Path, step: int, manifest: Mapping) -> dict:
    d = out / f"{ADAPTER_PREFIX}{step:04d}"
    model.save_pretrained(str(d))
    tau = manifest["tau"]
    meta = {
        "schema": "part3_b.adapter/1", "model_id": manifest["model_id"], "model_path": manifest["model_path"], "objective": manifest["objective"], "seed": manifest["seed"], "lr": manifest["lr"],
        "beta": manifest["beta"], "step": step, "plan_sha256": manifest["data"]["plan_sha256"], "run_name": out.name, "tag": f"{out.name}_step{step}", "timing_only": manifest["timing_only"],
        "tau": tau["value"] if tau else None, "tau_file_sha256": tau["file_sha256"] if tau else None, "adapter_model_sha256": sha256_file(d / "adapter_model.safetensors"),
    }
    write_json(d / "part3_b_meta.json", meta)
    return {"step": step, "dir": d.name, "adapter_model_sha256": meta["adapter_model_sha256"]}


# ---------------------------------------------------------------------------------------
# tau
# ---------------------------------------------------------------------------------------

def compute_tau(args: argparse.Namespace, deps: Deps) -> dict:
    """tau_model = max(1, median |m|) of the frozen base model over the 12,800 ORIGINAL ``train`` decision prompts (800 items x 16 conditions) as rendered for ``routing``.

    No gradient, eval mode, the numerics of the training forward pass (the weights in ``--dtype``, the same autocast), the training batch size in rows per forward, no adapter. Writes the JSON file:
    tau and the median, the 5, 25, 50, 75 and 95 % quantiles of |m|, the share of rows on the wrong side (m > 0 differs from A > z0), the share of the wrong-side rows with |m| / tau > 3, and the hashes of
    the rendered token ids, of config.json and the weight index of the base model, and of the code; and the sha256 over the bytes of every base file, ``model_files``."""
    requested = resolve_device(args.device)
    if deps.synthetic and requested.type == "cuda":
        raise TrainPlanError("synthetic data is for the CPU dry run only")
    if args.batch_size < 1:
        raise TrainPlanError("--batch-size must be positive")
    out = assert_off_root(args.out)
    if out.exists():
        raise TrainPlanError(f"{out} exists; a tau file is never overwritten")
    device = prepare_device(requested)
    print(f"[device] {requested}" + (" (all GPUs hidden)" if device.type == "cpu" else f" -> {device} of CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']!r}: {torch.cuda.get_device_name(device)}"), flush=True)
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    amp = dtype == torch.bfloat16
    model_path = Path(args.model_path) if args.model_path else Path(model_entry(args.model)["path"])
    t0 = time.time()
    model_files = model_sources(model_path)  # the bytes of every base file are hashed before anything else; a damaged cache blob stops the command here
    split = deps.load_split(args.model, "train")
    rows = build_rows(split)
    check_training_support(rows, "routing")
    n_items = len(split.items)
    if not deps.synthetic and (n_items != 800 or len(rows) != 16 * n_items):
        raise TrainPlanError(f"expected 800 training items with 16 conditions each, got {n_items} items and {len(rows)} rows")

    from transformers import AutoTokenizer

    from nmi.readout.digits import resolve_digit_token_ids

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    digit_ids = torch.tensor(resolve_digit_token_ids(tokenizer, DIGITS), dtype=torch.long)
    renderer = Renderer(deps.prompts, tokenizer, args.model)
    ids_hash = decision_ids_sha256(rows, renderer)
    model = load_base_model(model_path, dtype, device, args.attn_implementation, gradient_checkpointing=False)
    model.eval()
    pad_id, n = pad_token_id(tokenizer), len(rows)
    abs_m, wrong = np.empty(n), np.empty(n, dtype=bool)
    with torch.no_grad():
        for lo in range(0, n, args.batch_size):
            chunk = rows[lo:lo + args.batch_size]
            batch = make_batch(chunk, [r.cond for r in chunk], renderer)
            lp5 = digit_logprobs(last_logits(model, batch.dec_ids, pad_id, device, amp), digit_ids.to(device)).cpu()
            m, _, _ = margin_terms(lp5, batch.help_idx, batch.ans_idx, batch.A, batch.z0)
            abs_m[lo:lo + len(chunk)] = m.abs().double().numpy()
            wrong[lo:lo + len(chunk)] = ((m > 0) != (batch.A > batch.z0)).numpy()
            if (lo // args.batch_size) % 100 == 0:
                print(f"[tau {args.model}] {lo + len(chunk)}/{n} rows, {time.time() - t0:.0f}s", flush=True)
    if not np.isfinite(abs_m).all():
        raise TrainPlanError("a non-finite margin: the base model gives no usable readout on some train prompt")
    median = float(np.median(abs_m))
    tau = max(1.0, median)
    q5, q25, q50, q75, q95 = (float(x) for x in np.percentile(abs_m, [5, 25, 50, 75, 95]))
    n_wrong = int(wrong.sum())
    n_wrong_big = int((wrong & (abs_m / tau > 3.0)).sum())
    doc = {
        "schema": TAU_SCHEMA, "model_id": args.model, "model_path": str(model_path), "tau": tau, "median_abs_m": median,
        "rule": "tau = max(1, median |m|) of the frozen base model (no adapter) over the original train decision prompts, as rendered for routing",
        "abs_m_quantiles": {"5": q5, "25": q25, "50": q50, "75": q75, "95": q95},
        "wrong_side": {"n_rows": n, "n_wrong": n_wrong, "share": n_wrong / n, "n_wrong_with_abs_m_over_tau_gt_3": n_wrong_big,
                       "share_of_wrong_with_abs_m_over_tau_gt_3": (n_wrong_big / n_wrong) if n_wrong else None},
        "hashes": {"decision_ids_sha256": ids_hash, "model_files": model_files, **model_identity(model_path),
                   "code": code_sources([Path(deps.prompts.__file__)] if getattr(deps.prompts, "__file__", None) else [])},
        "settings": {"device": str(requested), "dtype": args.dtype, "autocast_bf16": amp, "batch_size": args.batch_size, "attn_implementation": args.attn_implementation, "eval_mode": True, "no_grad": True,
                     "adapter": None},
        "n_items": n_items, "environment": environment(), "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "elapsed_sec": round(time.time() - t0, 1),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, doc)
    return doc


# ---------------------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------------------

READ_SETS = ("val", "val_panswer")  # the only sets `select` opens
EXPECTED_VAL_ROWS, EXPECTED_VAL_ITEMS = 1600, 200  # 200 cal-val items x 8 conditions; 200 forced P-answers


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def condition_key(row: Mapping) -> str:
    """The identity of a planned trial: its ``trial_id`` when the readout carries one, else item, variant and condition."""
    if row.get("trial_id") is not None:
        return str(row["trial_id"])
    return json.dumps([row.get("item_key"), row.get("variant"), row.get("condition")], sort_keys=True)


def gate_units(rows: Sequence[Mapping], name: str = "") -> list[dict]:
    """The quality gate per gate unit of one readout set: the share of PLANNED trials with five-digit mass >= 0.5 must be >= 0.99; a missing trial counts as failing.
    The comparison is exact (fractions); the threshold is ``GATE_MIN_SHARE``. The unit of a row is its ``unit`` field (the readout writes it)."""
    threshold = Fraction(GATE_MIN_SHARE).limit_denominator(1000)  # exact: 297 of 300 is 0.99, not 0.98999...
    counts: dict[str, list[int]] = {}
    for r in rows:
        c = counts.setdefault(str(r.get("unit") or r.get("set") or name), [0, 0, 0])  # planned, pass, missing
        missing = bool(r.get("missing", False))
        c[0] += 1
        c[1] += int(bool(r.get("gate_pass", False)) and not missing)
        c[2] += int(missing)
    return [{"unit": u, "n_planned": n, "n_pass": k, "n_missing": miss, "share": k / n, "passes": bool(Fraction(k, n) >= threshold)} for u, (n, k, miss) in sorted(counts.items())]


def routing_regret(rows: Sequence[Mapping]) -> dict:
    """Validation routing regret: r = max(payoff_help, payoff_self) - payoff of the readout's decision (help iff margin > 0), averaged over each item's conditions, then over items.

    Returns ``{"regret", "n_items", "n_rows", "n_missing", "gate_pass_rate", "keys"}``; a row without a finite margin counts as missing and makes the regret ``None``.
    """
    per_item: dict[str, list[float]] = collections.defaultdict(list)
    keys, missing, gate = set(), 0, 0
    for r in rows:
        k = condition_key(r)
        if k in keys:
            raise TrainPlanError("a (item, condition) trial appears twice in the readout")
        keys.add(k)
        margin = r.get("margin")
        if r.get("missing") or margin is None or not math.isfinite(float(margin)):
            missing += 1
            continue
        asked = float(margin) > 0
        if "asked" in r and bool(r["asked"]) != asked:
            raise TrainPlanError("a readout row has asked != (margin > 0)")
        ph, ps = float(r["payoff_help"]), float(r["payoff_self"])
        per_item[r["item_key"]].append(max(ph, ps) - (ph if asked else ps))
        gate += bool(r.get("gate_pass", False))
    regret = None if missing or not per_item else float(np.mean([np.mean(v) for v in per_item.values()]))
    return {"regret": regret, "n_items": len(per_item), "n_rows": len(rows), "n_missing": missing, "gate_pass_rate": gate / max(1, len(rows)), "keys": keys, "gate_units": gate_units(rows, "val")}


def forced_accuracy(rows: Sequence[Mapping]) -> dict:
    """Forced P-answer accuracy on the cal-val items: counts of correct answers; a row without a correctness flag counts as missing."""
    seen, correct, missing, gate = set(), 0, 0, 0
    for r in rows:
        if r["item_key"] in seen:
            raise TrainPlanError("an item appears twice in the P-answer readout")
        seen.add(r["item_key"])
        flag = r.get("correct", r.get("is_correct"))
        if r.get("missing") or flag is None:
            missing += 1
        else:
            correct += bool(flag)
        gate += bool(r.get("gate_pass", False))
    return {"k": correct, "n": len(rows), "n_missing": missing, "keys": seen, "gate_pass_rate": gate / max(1, len(rows)), "gate_units": gate_units(rows, "val_panswer")}


def select_checkpoint(base: Mapping, candidates: Sequence[Mapping], max_drop_pp: float = MAX_ACCURACY_DROP_PP) -> dict:
    """The selection rule: lowest validation routing regret among the candidates whose forced P-answer accuracy falls by at most ``max_drop_pp`` percentage points from the base model;
    ties go to the smaller lr, then the smaller beta, then the earlier step. A candidate with missing rows is not eligible. Nothing is relaxed when none qualifies.

    ``base`` has ``k`` and ``n`` (correct answers and items); every candidate has ``tag, lr, beta, step, regret, k, n, n_missing``.
    """
    table = []
    for c in candidates:
        row = dict(c)
        if c["n"] != base["n"]:
            row.update(eligible=False, reason="item count differs from the base model")
        elif not c.get("readout_valid", True):  # the readout fails the quality gate, so the candidate is not ranked
            row.update(eligible=False, reason="readout invalid (not ranked): " + "; ".join(c.get("invalid_units", [])))
        elif c["n_missing"] or c["regret"] is None:
            row.update(eligible=False, reason="missing readout rows")
        else:
            drop_items = base["k"] - c["k"]
            row["drop_pp"] = 100.0 * drop_items / base["n"]
            if drop_items * 100.0 > max_drop_pp * base["n"] + 1e-9:  # counts, so that exactly one point is allowed
                row.update(eligible=False, reason=f"forced P-answer accuracy drops by {row['drop_pp']:.2f} points (> {max_drop_pp})")
            else:
                row.update(eligible=True, reason="")
        table.append(row)
    eligible = [r for r in table if r["eligible"]]
    selected = min(eligible, key=lambda r: (round(r["regret"], 12), r["lr"], r["beta"], r["step"])) if eligible else None
    return {"status": "selected" if selected else "no_selectable_checkpoint", "selected": selected, "table": table}


def check_readout_provenance(readout_dir: Path, tag: str, model_id: str, adapter_sha256: str | None) -> str:
    """Guard against a readout filed under the wrong tag. The ``read`` command of ``scripts/run_part3_b_eval.py`` writes ``manifest.json`` into the readout directory with the ``checkpoint`` it was given; for a candidate that path
    holds the ``merge_manifest.json`` of ``merge`` (tag and adapter hash). A definite mismatch raises; a missing manifest or a merged directory that has been deleted is reported, not an error."""
    mp = readout_dir / "manifest.json"
    if not mp.is_file():
        return "not verified: no read manifest"
    man = read_json(mp)
    if man.get("model") not in (None, model_id):
        raise TrainPlanError(f"{tag}: the readout belongs to model {man.get('model')!r}, not {model_id!r}")
    checkpoint = man.get("checkpoint")
    if tag == "base":
        if checkpoint != "base":
            raise TrainPlanError(f"the base readout was read from {checkpoint!r}, not from the base model")
        return "verified: read from the base model"
    if checkpoint in (None, "base"):
        raise TrainPlanError(f"{tag}: the readout was read from {checkpoint!r}, not from a merged checkpoint")
    mm = Path(checkpoint) / "merge_manifest.json"
    if not mm.is_file():
        return "not verified: the merged checkpoint is no longer on disk"
    merged = read_json(mm)
    if merged.get("tag") != tag or (adapter_sha256 is not None and merged.get("adapter_meta", {}).get("adapter_model_sha256") != adapter_sha256):
        raise TrainPlanError(f"{tag}: the readout was read from a checkpoint merged from another adapter ({merged.get('tag')!r})")
    return "verified: merged from this adapter"


def discover_candidates(run_dirs: Sequence[str | Path]) -> list[dict]:
    cands = []
    for rd in run_dirs:
        rd = Path(rd)
        man = read_json(rd / "manifest.json")
        if man.get("status") != "finished":
            raise TrainPlanError(f"{rd} is not a finished run")
        if man.get("timing_only"):
            raise TrainPlanError(f"{rd} is a timing-only run; it never enters a selection")
        for ad in man["adapters"]:
            cands.append({"run_dir": str(rd), "run_name": rd.name, "model_id": man["model_id"], "objective": man["objective"], "seed": man["seed"], "lr": man["lr"], "beta": man["beta"],
                          "step": ad["step"], "adapter_dir": str(rd / ad["dir"]), "adapter_sha256": ad.get("adapter_model_sha256"), "tag": f"{rd.name}_step{ad['step']}",
                          "tau_file_sha256": (man.get("tau") or {}).get("file_sha256"), "tau": (man.get("tau") or {}).get("value"), "tau_info": man.get("tau"),
                          "model_files": (man.get("sources") or {}).get("model_files")})
    return cands


def describe_failed_units(set_name: str, units: Sequence[Mapping]) -> list[str]:
    return [f"{set_name} unit {u['unit']}: {u['n_pass']} of {u['n_planned']} planned trials pass ({u['share']:.4f} < 0.99; {u['n_missing']} missing)" for u in units if not u["passes"]]


def check_tau_binding(cands: Sequence[Mapping], tau_file: str | Path | None) -> dict:
    """The base files of every candidate against the tau file of that model. A run that used tau names its tau file in the manifest; that file is read again (its sha256 must be the recorded one)
    and its ``model_files`` must equal the candidate's. ``tau_file`` (optional) names the tau file of the model explicitly; it is required to bind an sft search, whose runs use no tau."""
    model_id, record = cands[0]["model_id"], {"tau_file": None, "checked": False, "note": ""}
    docs: dict[str, dict] = {}
    if tau_file is not None:
        p = Path(tau_file)
        docs[str(p.resolve())] = read_json(p)
        record.update(tau_file=str(p.resolve()), tau_file_sha256=sha256_file(p))
    info = cands[0]["tau_info"]
    if info:
        p = Path(info["file"])
        if tau_file is not None and p.resolve() != Path(tau_file).resolve() and sha256_file(tau_file) != info["file_sha256"]:
            raise TrainPlanError("the tau file given to select is not the one the runs used")
        if p.is_file():
            if sha256_file(p) != info["file_sha256"]:
                raise TrainPlanError(f"{p} changed since the runs used it")
            docs.setdefault(str(p.resolve()), read_json(p))
            record.update(tau_file=str(p.resolve()), tau_file_sha256=info["file_sha256"])
        elif not docs:
            docs[str(p)] = {"model_id": info["model_id"], "hashes": info["hashes"]}
            record["note"] = "the tau file is no longer on disk; compared with the copy in the run manifest"
    for key, doc in docs.items():
        if doc["model_id"] != model_id:
            raise TrainPlanError(f"the tau file {key} belongs to model {doc['model_id']!r}, not {model_id!r}")
        for c in cands:
            if c["model_files"] != doc["hashes"]["model_files"]:
                raise TrainPlanError(f"{c['tag']}: the base model files of the run differ from those of the tau file {key}")
        record["checked"] = True
    if not record["checked"]:
        record["note"] = "no tau file: the runs use no tau (sft) and none was given; the candidates share the same base model files"
    return record


def run_select(run_dirs: Sequence[str | Path], val_out: str | Path, allow_partial: bool = False, expect_rows: tuple[int, int] | None = (EXPECTED_VAL_ROWS, EXPECTED_VAL_ITEMS),
               tau_file: str | Path | None = None) -> dict:
    """Select among the seed-0 runs of one (model, objective) search. ``expect_rows`` = (decision trials of ``val``, items of ``val_panswer``) that the base readout must hold; ``None`` skips the count."""
    val_out = Path(val_out)
    cands = discover_candidates(run_dirs)
    if not cands:
        raise TrainPlanError("no candidate checkpoint")
    for key in ("model_id", "objective", "seed", "tau_file_sha256"):
        if len({c[key] for c in cands}) != 1:
            raise TrainPlanError(f"select compares the runs of one (model, objective) search at seed 0 with one tau; the runs differ in {key}")
    if len({json.dumps(c["model_files"], sort_keys=True) for c in cands}) != 1:
        raise TrainPlanError("the runs of one search were trained on different base model files")
    tau_check = check_tau_binding(cands, tau_file)
    if cands[0]["seed"] != 0:
        raise TrainPlanError("the search runs with seed 0")
    if cands[0]["objective"] not in ("routing", "sft"):
        raise TrainPlanError("a search exists for routing and sft only; fixed_cost and shuffled inherit the routing selection")
    expected = {(lr, beta, step) for lr in GRID_LR for beta in GRID_BETA for step in GRID_STEPS}
    got = {(c["lr"], c["beta"], c["step"]) for c in cands}
    if not allow_partial and (got != expected or len(got) != len(cands)):
        raise TrainPlanError(f"the search needs exactly the frozen grid (lr x beta x step); missing {sorted(expected - got)}, extra {sorted(got - expected)}")
    inputs: dict[str, str] = {}

    def load(tag: str, name: str) -> list[dict]:
        if name not in READ_SETS:
            raise AssertionError(f"select reads only {READ_SETS}, not {name!r}")
        p = val_out / tag / name / "rows.jsonl"
        if not p.is_file():
            raise TrainPlanError(f"missing readout {p}")
        inputs[str(p)] = sha256_file(p)
        return read_jsonl(p)

    provenance = {"base": check_readout_provenance(val_out / "base", "base", cands[0]["model_id"], None)}
    base_dec, base_pa = routing_regret(load("base", "val")), forced_accuracy(load("base", "val_panswer"))
    if base_dec["n_missing"] or base_pa["n_missing"]:
        raise TrainPlanError("the base model's validation readout has missing rows")
    bad_base = describe_failed_units("val", base_dec["gate_units"]) + describe_failed_units("val_panswer", base_pa["gate_units"])
    if bad_base:
        raise TrainPlanError("the base model's validation readout fails the quality gate: " + "; ".join(bad_base))
    if expect_rows is not None and (base_dec["n_rows"], base_pa["n"]) != tuple(expect_rows):
        raise TrainPlanError(f"the base readout holds {base_dec['n_rows']} decision trials and {base_pa['n']} P-answer items, expected {tuple(expect_rows)}")
    results = []
    for c in cands:
        provenance[c["tag"]] = check_readout_provenance(val_out / c["tag"], c["tag"], c["model_id"], c["adapter_sha256"])
        dec, pa = routing_regret(load(c["tag"], "val")), forced_accuracy(load(c["tag"], "val_panswer"))
        if dec["keys"] != base_dec["keys"] or pa["keys"] != base_pa["keys"]:
            raise TrainPlanError(f"{c['tag']}: the trials differ from the base model's trials (every planned trial must appear once)")
        invalid = describe_failed_units("val", dec["gate_units"]) + describe_failed_units("val_panswer", pa["gate_units"])
        results.append({**{k: c[k] for k in ("tag", "lr", "beta", "step", "run_name", "adapter_dir")}, "regret": dec["regret"], "k": pa["k"], "n": pa["n"],
                        "n_missing": dec["n_missing"] + pa["n_missing"], "gate_pass_rate": dec["gate_pass_rate"], "pa_gate_pass_rate": pa["gate_pass_rate"], "readout_valid": not invalid,
                        "invalid_units": invalid, "gate_units": {"val": dec["gate_units"], "val_panswer": pa["gate_units"]}})
    verdict = select_checkpoint({"k": base_pa["k"], "n": base_pa["n"]}, results)
    sel = verdict["selected"]
    out = {
        "schema": "part3_b.selection/1", "status": verdict["status"], "model_id": cands[0]["model_id"], "objective": cands[0]["objective"], "search_seed": 0,
        "rule": "lowest validation routing regret among candidates whose forced P-answer accuracy on cal-val drops by at most 1 percentage point from the base model; ties: smaller lr, smaller beta, earlier step",
        "base": {"regret": base_dec["regret"], "accuracy_k": base_pa["k"], "n_items": base_pa["n"], "n_val_rows": base_dec["n_rows"]},
        "candidates": verdict["table"], "selected": sel, "inputs": inputs, "readout_provenance": provenance,
        "excluded_for_missing_rows": [c["tag"] for c in verdict["table"] if c["reason"] == "missing readout rows"],  # a statistic without its readout is technically undetermined, not zero
        "readout_invalid": [c["tag"] for c in verdict["table"] if not c.get("readout_valid", True)],  # not ranked
        "base_gate_units": {"val": base_dec["gate_units"], "val_panswer": base_pa["gate_units"]},
        "tau": cands[0]["tau"], "tau_file_sha256": cands[0]["tau_file_sha256"], "tau_binding": tau_check,
    }
    if sel:
        out["retrain"] = [{"seed": s, "lr": sel["lr"], "beta": sel["beta"], "steps": sel["step"], "save_at": [sel["step"]]} for s in (1, 2)]
        if cands[0]["model_id"] == "qwen3-4b" and cands[0]["objective"] == "routing":  # the controls inherit this selection, seed 0, no search of their own
            out["controls"] = [{"objective": o, "seed": 0, "lr": sel["lr"], "beta": sel["beta"], "steps": sel["step"], "save_at": [sel["step"]]} for o in CONTROL_OBJECTIVES]
    write_json(val_out / "selection.json", out)
    return out


def format_selection(sel: Mapping) -> str:
    lines = [f"{sel['model_id']} {sel['objective']}: {sel['status']}   base regret {sel['base']['regret']}, base correct {sel['base']['accuracy_k']}/{sel['base']['n_items']}"]
    lines.append(f"{'tag':48s} {'lr':>8s} {'beta':>5s} {'step':>5s} {'regret':>9s} {'drop_pp':>8s} {'gate':>6s} {'paGate':>6s} eligible")
    for c in sel["candidates"]:
        regret = "-" if c["regret"] is None else f"{c['regret']:.5f}"
        lines.append(f"{c['tag']:48s} {c['lr']:8.0e} {c['beta']:5.2f} {c['step']:5d} {regret:>9s} {c.get('drop_pp', float('nan')):8.2f} {c['gate_pass_rate']:6.3f} {c.get('pa_gate_pass_rate', float('nan')):6.3f} {c['eligible']} {c['reason']}")
    if sel["readout_invalid"]:
        lines.append(f"readout invalid (fails the quality gate, not ranked): {sel['readout_invalid']}")
    if sel["excluded_for_missing_rows"]:
        lines.append(f"technically undetermined (missing readout rows, not ranked): {sel['excluded_for_missing_rows']}")
    if sel["selected"]:
        s = sel["selected"]
        lines.append(f"selected: {s['tag']} (lr {s['lr']}, beta {s['beta']}, step {s['step']}); retrain seeds 1 and 2 to step {s['step']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------------------

_ST_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
_SKIP_COPY = {"README.md", ".gitattributes"}


def safetensors_header(path: str | Path) -> tuple[int, dict]:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return n, json.loads(fh.read(n))


def load_adapter_pairs(adapter_dir: Path) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], float, dict]:
    """module name -> (A, B) of a saved LoRA adapter, and its scaling alpha / r."""
    from safetensors.torch import load_file

    cfg = read_json(adapter_dir / "adapter_config.json")
    if cfg.get("peft_type") != "LORA" or cfg.get("use_dora") or cfg.get("use_rslora") or cfg.get("fan_in_fan_out") or cfg.get("bias", "none") != "none" or cfg.get("modules_to_save"):
        raise TrainPlanError("only a plain LoRA adapter (no DoRA, rsLoRA, bias or modules_to_save) can be merged by this command")
    scaling = float(cfg["lora_alpha"]) / float(cfg["r"])
    state = load_file(str(adapter_dir / "adapter_model.safetensors"))
    pairs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for key, tensor in state.items():
        m = re.fullmatch(r"base_model\.model\.(.+)\.lora_([AB])\.weight", key)
        if not m:
            raise TrainPlanError(f"unexpected adapter tensor {key!r}")
        slot = pairs.setdefault(m.group(1), [None, None])  # type: ignore[arg-type]
        slot[0 if m.group(2) == "A" else 1] = tensor
    if any(a is None or b is None for a, b in pairs.values()):
        raise TrainPlanError("an adapter module lacks lora_A or lora_B")
    return {k: (v[0], v[1]) for k, v in pairs.items()}, scaling, cfg


def fixed_cpu_path(fn):
    """The merge and the weight check run on one fixed CPU path, ``torch.set_num_threads(1)``, so that the float32 matrix product is reproducible bit for bit."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            return fn(*args, **kwargs)
        finally:
            torch.set_num_threads(previous)

    return wrapper


def merged_weight(w: torch.Tensor, a: torch.Tensor, b: torch.Tensor, scaling: float) -> torch.Tensor:
    """In exactly this order: Wf = W.to(float32); D = (B.to(float32) @ A.to(float32)) * s; W' = (Wf + D).to(dtype of W). The dtype is bfloat16 in the checkpoints; a float32 test model keeps float32.
    ``s`` is the scaling of the adapter's own adapter_config.json. The caller runs under ``fixed_cpu_path`` (one thread)."""
    wf = w.to(torch.float32)
    d = (b.to(torch.float32) @ a.to(torch.float32)) * scaling
    return (wf + d).to(w.dtype)


@fixed_cpu_path
def merge_weights(base_dir: Path, out_dir: Path, pairs: Mapping[str, tuple[torch.Tensor, torch.Tensor]], scaling: float) -> dict:
    """Copy the base checkpoint and rewrite, in place, only the byte ranges of the adapted weights with ``merged_weight``.
    Header, tensor order, all other tensors, the shard layout and the index stay byte-identical, so a multimodal wrapper (Gemma-4) keeps the loading structure of the original."""
    todo = {f"{name}.weight": ab for name, ab in pairs.items()}
    done: dict[str, dict] = {}
    shards = sorted(p for p in base_dir.iterdir() if p.suffix == ".safetensors")
    if not shards:
        raise TrainPlanError(f"no safetensors file in {base_dir}")
    for shard in shards:
        target = out_dir / shard.name
        shutil.copyfile(shard, target)
        n, header = safetensors_header(target)
        data_start = 8 + n
        with open(target, "r+b") as fh:
            for name, entry in header.items():
                if name == "__metadata__" or name not in todo:
                    continue
                dtype = _ST_DTYPES.get(entry["dtype"])
                if dtype is None:
                    raise TrainPlanError(f"{name}: unsupported dtype {entry['dtype']}")
                begin, end = entry["data_offsets"]
                fh.seek(data_start + begin)
                w = torch.frombuffer(bytearray(fh.read(end - begin)), dtype=dtype).reshape(entry["shape"])
                a, b = todo[name]
                if (b.shape[0], a.shape[1]) != tuple(w.shape):
                    raise TrainPlanError(f"{name}: LoRA delta {(b.shape[0], a.shape[1])} does not fit the weight {tuple(w.shape)}")
                new = merged_weight(w, a, b, scaling).contiguous()
                fh.seek(data_start + begin)
                fh.write(new.view(torch.uint8).numpy().tobytes())
                done[name] = {"shard": shard.name, "max_abs_delta": float(((b.to(torch.float32) @ a.to(torch.float32)) * scaling).abs().max()), "changed_elements": int((new.float() != w.float()).sum())}
    missing = sorted(set(todo) - set(done))
    if missing:
        raise TrainPlanError(f"{len(missing)} adapted weights are not in the base checkpoint, e.g. {missing[:3]}")
    return {"patched": done, "shards": [s.name for s in shards]}


def copy_non_weight_files(base_dir: Path, out_dir: Path) -> list[str]:
    copied = []
    for p in sorted(base_dir.iterdir()):
        if p.is_file() and p.suffix != ".safetensors" and p.name not in _SKIP_COPY and p.suffix != ".ipynb":
            shutil.copyfile(p, out_dir / p.name)
            copied.append(p.name)
    return copied


def merge_check_prompts(deps: Deps, model_id: str, tokenizer: Any, k: int = 4) -> list[tuple[list[int], str, str]]:
    """Check 2: the first k cal-val items by item_key, each with the first `val` condition of the plan. Returns (token ids, help label, item key) per prompt."""
    val = deps.load_split(model_id, "val")
    first: dict[str, Any] = {}
    for key, cond in val.conditions:
        first.setdefault(key, cond)
    keys = sorted(first)[:k]
    if len(keys) < k:
        raise TrainPlanError("fewer than four validation items for the merge check")
    renderer = Renderer(deps.prompts, tokenizer, model_id)
    out = []
    for key in keys:
        ids, _, help_label = renderer.decision(val.items[key], first[key])
        out.append((ids.tolist(), help_label, key))
    return out


def _logits_for(model, prompts: Sequence[Sequence[int]], pad_id: int, device: torch.device, amp: bool = False) -> list[torch.Tensor]:
    """Float32 logits (V,) of the readout position of every prompt, one prompt per forward pass, eval mode, no gradient, on the CPU. ``amp`` is the autocast switch of training (``last_logits``)."""
    model.eval()
    with torch.no_grad():
        return [last_logits(model, [p], pad_id, device, amp=amp)[0].cpu() for p in prompts]


def _release(device: torch.device) -> None:
    """Give a model back before the next one is loaded: collect the reference cycles of the library objects and empty the CUDA cache."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def float_order(t: torch.Tensor) -> torch.Tensor:
    """Map the bit patterns of a float tensor to integers that increase with the value (+0 and -0 coincide), so that |order(a) - order(b)| is the distance in representable steps."""
    flat = t.contiguous().reshape(-1)
    if flat.dtype in (torch.bfloat16, torch.float16):
        bits, sign, mag_mask = flat.view(torch.int16).to(torch.int64) & 0xFFFF, 0x8000, 0x7FFF
    elif flat.dtype == torch.float32:
        bits, sign, mag_mask = flat.view(torch.int32).to(torch.int64) & 0xFFFFFFFF, 0x80000000, 0x7FFFFFFF
    else:
        raise TrainPlanError(f"unsupported dtype {flat.dtype} for an ulp distance")
    mag = bits & mag_mask
    return torch.where((bits & sign) != 0, -mag, mag)


def ulp_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per element, the number of representable values between a and b of the same dtype (0 = bitwise equal up to the sign of zero)."""
    if a.dtype != b.dtype or a.shape != b.shape:
        raise TrainPlanError("ulp distance needs equal dtype and shape")
    return (float_order(a) - float_order(b)).abs()


def _int_bits(t: torch.Tensor) -> torch.Tensor:
    """The bit pattern of each element as an integer of the same width (so that +0 and -0 differ and a NaN compares by its bits)."""
    flat = t.contiguous().reshape(-1)
    if flat.dtype in (torch.bfloat16, torch.float16):
        return flat.view(torch.int16)
    if flat.dtype == torch.float32:
        return flat.view(torch.int32)
    raise TrainPlanError(f"unsupported dtype {flat.dtype} for a bitwise comparison")


@fixed_cpu_path
def check_merge_weights(base_dir: Path, adapter_dir: Path, merged_dir: Path) -> dict:
    """Check 1 (binding), on the weights, bitwise, and independent of the merge code: the base safetensors, the saved adapter files (with the scaling of their own adapter_config.json) and the
    merged safetensors are loaded from disk, and for every adapted module W' is recomputed on the fixed CPU path (one thread; Wf = W.to(float32); D = (B.to(float32) @ A.to(float32)) * s;
    W' = (Wf + D).to(dtype of W)). Every element of every adapted tensor must be bitwise equal to W'. A tolerance would let a missed merge pass whenever the correct value is one ulp from the base.

    * reported per module, without a threshold: the number of elements where W' differs from W_base, which shows that the merge changed the weights;
    * reported if an adapted tensor is not bitwise equal to W': the count and the largest ulp distance (a diagnosis, not a tolerance);
    * every other tensor bitwise equal to the base tensor (for a multimodal wrapper such as Gemma-4 this includes everything outside the language model, and no adapted tensor may lie outside it);
    * the same shard files, tensor names, shapes and dtypes as the base."""
    from safetensors import safe_open

    problems: list[str] = []
    cfg = read_json(adapter_dir / "adapter_config.json")
    if cfg.get("use_dora") or cfg.get("use_rslora") or cfg.get("bias", "none") != "none":
        problems.append("an adapter variant that this check does not know (DoRA, rsLoRA or bias)")
    s = float(cfg["lora_alpha"]) / float(cfg["r"])
    adapted: dict[str, dict[str, torch.Tensor]] = {}
    with safe_open(str(adapter_dir / "adapter_model.safetensors"), framework="pt") as f:
        for key in f.keys():
            found = re.fullmatch(r"base_model\.model\.(.+)\.lora_([AB])\.weight", key)
            if not found:
                problems.append(f"unexpected adapter tensor {key}")
                continue
            adapted.setdefault(found.group(1) + ".weight", {})[found.group(2)] = f.get_tensor(key)
    for name, ab in adapted.items():
        if set(ab) != {"A", "B"}:
            problems.append(f"{name}: lora_A or lora_B is missing")
    base_files = sorted(p.name for p in base_dir.glob("*.safetensors"))
    files_equal = base_files == sorted(p.name for p in merged_dir.glob("*.safetensors"))
    if not files_equal:
        problems.append("the merged checkpoint has other weight files than the base")
    modules: list[dict] = []
    n_elems = n_unequal = n_other = n_outside_lm = n_changed = 0
    worst, worst_name, seen, other_ok, schema_ok, wrapper = 0, None, set(), True, True, False
    for fname in base_files:
        if not (merged_dir / fname).is_file():
            continue
        with safe_open(str(base_dir / fname), framework="pt") as fb, safe_open(str(merged_dir / fname), framework="pt") as fm:
            names_b, names_m = set(fb.keys()), set(fm.keys())
            if names_b != names_m:
                schema_ok = False
                problems.append(f"{fname}: tensor names differ ({len(names_b ^ names_m)} not in both)")
            wrapper = wrapper or any(".language_model." in n for n in names_b)
            for name in sorted(names_b & names_m):
                sb, sm = fb.get_slice(name), fm.get_slice(name)
                if sb.get_shape() != sm.get_shape() or sb.get_dtype() != sm.get_dtype():
                    schema_ok = False
                    problems.append(f"{name}: shape or dtype differs from the base")
                    continue
                tb, tm = fb.get_tensor(name), fm.get_tensor(name)
                if name in adapted and set(adapted[name]) == {"A", "B"}:
                    seen.add(name)
                    a, b = adapted[name]["A"], adapted[name]["B"]
                    if (b.shape[0], a.shape[1]) != tuple(tb.shape):
                        problems.append(f"{name}: the LoRA product {(b.shape[0], a.shape[1])} does not fit the weight {tuple(tb.shape)}")
                        continue
                    wf = tb.to(torch.float32)
                    d = (b.to(torch.float32) @ a.to(torch.float32)) * s
                    ref = (wf + d).to(tb.dtype)
                    n_diff_base = int((_int_bits(ref) != _int_bits(tb)).sum())
                    n_bad = int((_int_bits(tm) != _int_bits(ref)).sum())
                    entry = {"name": name, "n_elements": tb.numel(), "n_changed_vs_base": n_diff_base}
                    n_elems += tb.numel()
                    n_changed += n_diff_base
                    if n_bad:
                        top = int(ulp_distance(tm, ref).max())
                        entry.update(n_not_bitwise_equal=n_bad, largest_ulp_distance=top)
                        n_unequal += n_bad
                        if top > worst:
                            worst, worst_name = top, name
                        problems.append(f"{name}: {n_bad} of {tb.numel()} elements are not bitwise equal to the recomputed W' (largest ulp distance {top})")
                    modules.append(entry)
                else:
                    n_other += 1
                    if ".language_model." not in name:
                        n_outside_lm += 1
                    if not torch.equal(tb.reshape(-1).view(torch.uint8), tm.reshape(-1).view(torch.uint8)):
                        other_ok = False
                        problems.append(f"{name}: not bitwise equal to the base (it is not an adapted tensor)")
    for name in sorted(set(adapted) - seen):
        problems.append(f"{name}: adapted module not found in the base checkpoint")
    if wrapper:
        outside = [n for n in adapted if ".language_model." not in n]
        if outside:
            problems.append(f"{len(outside)} adapted tensors lie outside the language model, e.g. {outside[0]}")
    return {
        "passed": not problems, "rule": "bitwise equality with W' = (W_base.float() + (B.float() @ A.float()) * s).to(dtype) on one CPU thread", "scaling": s, "n_adapted_tensors": len(seen), "n_adapted_elements": n_elems,
        "n_elements_changed_by_merge": n_changed, "merge_changed_weights": n_changed > 0, "n_modules_unchanged_by_merge": sum(1 for m in modules if m["n_changed_vs_base"] == 0), "modules": modules,
        "n_not_bitwise_equal": n_unequal, "largest_ulp_distance": worst if n_unequal else None, "worst_tensor": worst_name if n_unequal else None,
        "n_other_tensors": n_other, "other_tensors_bitwise_equal": other_ok, "names_shapes_dtypes_equal": schema_ok and files_equal, "multimodal_wrapper": wrapper,
        "n_tensors_outside_language_model": n_outside_lm if wrapper else None, "problems": problems[:20], "n_problems": len(problems),
    }


def _readout_of(logits: torch.Tensor, help_idx: int, ids: Sequence[int]) -> dict:
    """What check 2 reports for one computation on one prompt: the margin (the help digit minus the best of the item's four answer digits, from the log-softmax over the full vocabulary in float64),
    the digit with the highest logit among the five, the decision m > 0 (True = ask) and the log of the five-digit mass."""
    ids = list(ids)
    lp = torch.log_softmax(logits.double(), -1)[ids]
    others = [j for j in range(len(ids)) if j != help_idx]
    margin = float(lp[help_idx] - lp[others].max())
    return {"margin": margin, "digit_argmax": DIGITS[int(logits[ids].argmax())], "decision": margin > 0, "log_mass": float(torch.logsumexp(lp, 0))}


def compare_merge_logits(logits_a: Sequence[torch.Tensor], logits_b: Sequence[torch.Tensor], logits_c: Sequence[torch.Tensor], help_labels: Sequence[str], digit_ids: Sequence[int]) -> dict:
    """Check 2, a report without any threshold, on the full-vocabulary logits of the readout position (one tensor per prompt and computation) with the item's own labels.
    The computations are A (merged bf16 checkpoint upcast), B (adapter on, training numerics) and C (adapter on, float32); see ``CHECK2_COMPUTATIONS``.

    ``per_prompt`` rows: ``help_label``; ``margin_{A,B,C}``, ``digit_argmax_{A,B,C}`` (the digit "1".."5" with the highest logit), ``decision_{A,B,C}`` (m > 0) and ``log_mass_{A,B,C}`` (log of the
    five-digit mass); for the pairs AB (the merge against the model as trained) and AC (the precision sensitivity of the trained update) ``abs_margin_diff_``, ``max_abs_logit_diff_`` (over the
    whole vocabulary), ``abs_log_mass_diff_``, ``digit_argmax_equal_`` and ``decision_equal_``. The summary repeats the pair keys over all prompts (max of the differences, all of the equalities)."""
    ids = list(digit_ids)
    per_prompt = []
    for k, (a, b, c) in enumerate(zip(logits_a, logits_b, logits_c)):
        h = int(help_labels[k]) - 1
        got = {"A": _readout_of(a, h, ids), "B": _readout_of(b, h, ids), "C": _readout_of(c, h, ids)}
        row: dict[str, Any] = {"help_label": str(help_labels[k])}
        for key, r in got.items():
            row[f"margin_{key}"], row[f"digit_argmax_{key}"], row[f"decision_{key}"], row[f"log_mass_{key}"] = r["margin"], r["digit_argmax"], r["decision"], r["log_mass"]
        for pair, other in (("AB", b), ("AC", c)):
            x, y = got["A"], got[pair[1]]
            row[f"abs_margin_diff_{pair}"] = abs(x["margin"] - y["margin"])
            row[f"max_abs_logit_diff_{pair}"] = float((a - other).abs().max())
            row[f"abs_log_mass_diff_{pair}"] = abs(x["log_mass"] - y["log_mass"])
            row[f"digit_argmax_equal_{pair}"] = x["digit_argmax"] == y["digit_argmax"]
            row[f"decision_equal_{pair}"] = x["decision"] == y["decision"]
        per_prompt.append(row)
    out: dict[str, Any] = {"n_prompts": len(per_prompt), "per_prompt": per_prompt}
    for pair in ("AB", "AC"):
        out[f"max_abs_margin_diff_{pair}"] = max(p[f"abs_margin_diff_{pair}"] for p in per_prompt)
        out[f"max_abs_logit_diff_{pair}"] = max(p[f"max_abs_logit_diff_{pair}"] for p in per_prompt)
        out[f"max_abs_log_mass_diff_{pair}"] = max(p[f"abs_log_mass_diff_{pair}"] for p in per_prompt)
        out[f"digit_argmax_equal_{pair}"] = all(p[f"digit_argmax_equal_{pair}"] for p in per_prompt)
        out[f"decision_equal_{pair}"] = all(p[f"decision_equal_{pair}"] for p in per_prompt)
    return out


def _logits_merged_upcast(merged_dir: Path, prompts: Sequence[Sequence[int]], pad_id: int, device: torch.device) -> list[torch.Tensor]:
    """Computation A: the merged bf16 checkpoint reloaded from disk and upcast to float32. The model is local to this function and goes out of scope when it returns."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(str(merged_dir), dtype=torch.float32, local_files_only=True).to(device)
    return _logits_for(model, prompts, pad_id, device, amp=False)


def _logits_adapter_on(base_path: Path, adapter_dir: Path, dtype: torch.dtype, amp: bool, prompts: Sequence[Sequence[int]], pad_id: int, device: torch.device) -> list[torch.Tensor]:
    """Computations B (``dtype`` bfloat16, ``amp`` True: the training numerics, the base loaded in bf16, the LoRA in float32, eval mode, the autocast of ``last_logits``) and C (float32, no autocast).
    The model is local to this function and goes out of scope when it returns."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(str(base_path), dtype=dtype, local_files_only=True).to(device)
    on = PeftModel.from_pretrained(base, str(adapter_dir), torch_device=str(device))  # never the library's default device (cuda when any GPU is visible)
    if any(p.dtype != torch.float32 for name, p in on.named_parameters() if "lora_" in name):
        raise TrainPlanError("the LoRA parameters of the adapter-on model must be float32, as in training")
    return _logits_for(on, prompts, pad_id, device, amp=amp)


def check_merge_logits(adapter_dir: Path, merged_dir: Path, base_path: Path, prompts: Sequence[Sequence[int]], help_labels: Sequence[str], pad_id: int, digit_ids: Sequence[int],
                       device: torch.device) -> dict:
    """Check 2, a report on the 4 fixed prompts with the item's own labels, and no acceptance test (acceptance is check 1 alone). Three computations (``CHECK2_COMPUTATIONS``):
    A the merged bf16 checkpoint upcast to float32, B the adapter-on model in the training numerics (bf16 base, float32 LoRA, bf16 autocast, eval mode, on ``device``), C the adapter-on model in
    float32. For each: margin, digit argmax and decision; for A against B (the merge against the model as trained) and A against C (the precision sensitivity of the trained update): the
    differences (``compare_merge_logits`` lists the keys). The models are loaded one at a time and released before the next one, so that no two float32 copies of a base model are ever held."""
    logits_a = _logits_merged_upcast(merged_dir, prompts, pad_id, device)
    _release(device)
    logits_b = _logits_adapter_on(base_path, adapter_dir, torch.bfloat16, True, prompts, pad_id, device)
    _release(device)
    logits_c = _logits_adapter_on(base_path, adapter_dir, torch.float32, False, prompts, pad_id, device)
    _release(device)
    return {**compare_merge_logits(logits_a, logits_b, logits_c, help_labels, digit_ids), "role": "report only (R15.2): acceptance rests on check 1", "computations": dict(CHECK2_COMPUTATIONS),
            "device": str(device)}


def run_merge(adapter_dir: str | Path, out_dir: str | Path, deps: Deps, device: str = "cpu", verify: bool = True, hash_merged: bool = False) -> dict:
    """Merge and check. ``accepted`` is check 1 alone (the weights, bitwise, on the CPU); check 2 is a report of three computations on the 4 fixed prompts, on ``device``, and never
    changes the verdict. The verdict is written to the manifest before the report is computed. The adapter of a timing-only run may be merged (timed)."""
    requested = resolve_device(device)
    adapter_dir, out_dir = Path(adapter_dir), assert_off_root(out_dir)
    dev = prepare_device(requested)
    meta = read_json(adapter_dir / "part3_b_meta.json")
    base_dir = Path(meta["model_path"])
    if out_dir.exists() and any(out_dir.iterdir()):
        raise TrainPlanError(f"{out_dir} exists and is not empty")
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pairs, scaling, _ = load_adapter_pairs(adapter_dir)
    result = merge_weights(base_dir, out_dir, pairs, scaling)
    copied = copy_non_weight_files(base_dir, out_dir)
    manifest = {
        "schema": "part3_b.merge/3", "adapter_dir": str(adapter_dir), "adapter_meta": meta, "base_path": str(base_dir), "scaling": scaling, "formula": "W' = round(fp32(W_base) + s * fp32(B) @ fp32(A))",
        "n_adapted_weights": len(result["patched"]), "shards": result["shards"], "copied_files": copied, "max_abs_delta": max(v["max_abs_delta"] for v in result["patched"].values()),
        "merged_sha256": {sh: sha256_file(out_dir / sh) for sh in result["shards"]} if hash_merged else None, "tag": meta["tag"], "timing_only_adapter": bool(meta.get("timing_only")),
        "timing": {"merge_seconds": round(time.time() - t0, 3)},
    }
    write_json(out_dir / "merge_manifest.json", manifest)
    if verify:
        from transformers import AutoTokenizer

        from nmi.readout.digits import resolve_digit_token_ids

        t1 = time.time()
        manifest["check1_weights"] = check_merge_weights(base_dir, adapter_dir, out_dir)
        manifest["timing"]["check1_seconds"] = round(time.time() - t1, 3)
        manifest["accepted"] = bool(manifest["check1_weights"]["passed"])  # acceptance rests on check 1 alone
        write_json(out_dir / "merge_manifest.json", manifest)
        t2 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(str(out_dir), local_files_only=True)
        checks = merge_check_prompts(deps, meta["model_id"], tokenizer)
        manifest["check2_prompts"] = [{"item_key": c[2], "help_label": c[1], "n_tokens": len(c[0]), "ids_sha256": sha256_bytes(np.asarray(c[0], dtype="<i4").tobytes())} for c in checks]
        try:
            manifest["check2_logits"] = check_merge_logits(adapter_dir, out_dir, base_dir, [c[0] for c in checks], [c[1] for c in checks], pad_token_id(tokenizer),
                                                          resolve_digit_token_ids(tokenizer, DIGITS), dev)
        except Exception as exc:  # the report could not be produced: the verdict of check 1 stands and is on disk, the failure is recorded and raised, never swallowed
            manifest["check2_logits"] = {"error": f"{type(exc).__name__}: {exc}"}
            write_json(out_dir / "merge_manifest.json", manifest)
            raise
        manifest["timing"]["check2_seconds"] = round(time.time() - t2, 3)
    write_json(out_dir / "merge_manifest.json", manifest)
    return manifest


# ---------------------------------------------------------------------------------------
# GPU memory estimate
# ---------------------------------------------------------------------------------------

def estimate_memory(model_path: str | Path, batch: int = BATCH, micro: int | None = None, tokens: int = 400, checkpointing: bool = True, overhead_gb: float = 1.5) -> dict:
    """Peak GPU memory of one training step in GB, from the configuration (meta-device module shapes) and the weight file sizes. An estimate within about +-30 %, not a measurement.

    Per micro-batch of k decision plus k P-answer prompts of ``tokens`` tokens, with gradient checkpointing: the layer inputs of both graphs (bf16), plus the recomputed activations of one
    layer (bf16 base activations and, for every adapted linear, a float32 cast, the dropout output, the dropout mask and the autocast bf16 copy of its input: 11 bytes per input element).
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    model_path = Path(model_path)
    micro = micro or batch
    cfg = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    names = lora_target_names(model)
    n_lora = sum(LORA_R * (model.get_submodule(n).in_features + model.get_submodule(n).out_features) for n in names)
    layers: dict[int, dict[str, torch.nn.Linear]] = collections.defaultdict(dict)
    for n in names:
        found = re.search(r"layers\.(\d+)\.(?:self_attn|mlp)\.(\w+)$", n)
        layers[int(found.group(1))][found.group(2)] = model.get_submodule(n)
    text_cfg = cfg.get_text_config()
    hidden, n_layers, vocab = int(text_cfg.hidden_size), int(text_cfg.num_hidden_layers), int(text_cfg.vocab_size)
    layer_base, layer_lora = [], []
    for li in sorted(layers):
        mods = layers[li]
        q_out, inter = mods["q_proj"].out_features, mods["gate_proj"].out_features
        kv_out = mods["k_proj"].out_features if "k_proj" in mods else 0  # the shared-KV layers of Gemma-4 have no k and v projection
        # saved bf16 elements per token: the seven linear outputs, the attention context, the activation and the gate-up product, norm outputs and residual stream, normed and rotated q and k
        elems = sum(m.out_features for m in mods.values()) + q_out + 2 * inter + 4 * hidden + 2 * q_out + 2 * kv_out
        layer_base.append(elems * 2)
        layer_lora.append(sum(m.in_features for m in mods.values()) * 11)
    weights = sum(p.stat().st_size for p in model_path.iterdir() if p.suffix == ".safetensors")
    toks_graph = micro * tokens  # one graph (decision or P-answer) of a micro-batch
    boundaries = 2 * toks_graph * n_layers * hidden * 2
    peak_layer = toks_graph * (max(layer_base) + max(layer_lora))
    all_layers = 2 * toks_graph * sum(b + lo for b, lo in zip(layer_base, layer_lora))
    activations = (boundaries + peak_layer) if checkpointing else all_layers
    adam = n_lora * 16  # float32 parameter, gradient and the two Adam moments
    logits = 2 * micro * vocab * 4 * 6
    total = weights + adam + activations * 1.15 + logits + overhead_gb * 2**30
    gb = lambda x: round(x / 2**30, 2)  # noqa: E731
    return {"model_path": str(model_path), "batch": batch, "micro_batch": micro, "tokens_per_prompt": tokens, "gradient_checkpointing": checkpointing, "lora_params_millions": round(n_lora / 1e6, 2),
            "weights_gb": gb(weights), "lora_adam_gb": gb(adam), "activations_gb": gb(activations * 1.15), "logits_gb": gb(logits), "overhead_gb": overhead_gb, "total_gb": gb(total)}


# ---------------------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train one LoRA run")
    t.add_argument("--model", required=True)
    t.add_argument("--objective", required=True, choices=OBJECTIVES)
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--lr", type=float, required=True)
    t.add_argument("--beta", type=float, required=True)
    t.add_argument("--steps", type=int, default=1000)
    t.add_argument("--save-at", type=parse_save_at, default=parse_save_at("250,500,1000"))
    t.add_argument("--out", required=True)
    t.add_argument("--model-path", default=None, help="override the model directory of the configuration (the tiny test model)")
    t.add_argument("--device", default="cpu", help="cpu (default) or an explicit cuda:N; a bare 'cuda' is refused. A CPU run hides all GPUs from the process.")
    t.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    t.add_argument("--batch", type=int, default=BATCH, help="decision prompts and P-answer prompts per step")
    t.add_argument("--micro-batch", type=int, default=DEFAULT_MICRO_BATCH, help="rows per forward and backward pass; the optimiser step is over the whole batch")
    t.add_argument("--warmup", type=int, default=WARMUP_STEPS)
    t.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM, help="0 disables clipping")
    t.add_argument("--no-grad-checkpointing", action="store_true")
    t.add_argument("--attn-implementation", default="sdpa")
    t.add_argument("--allow-offgrid", action="store_true", help="allow an lr, beta, model or seed outside the frozen candidates (tests and dry runs only)")
    t.add_argument("--tau-file", default=None, help="the tau file of this model (the tau command); required for routing, fixed_cost and shuffled, refused for sft")
    t.add_argument("--time-only", type=int, default=None, metavar="N", help="N steps of the formal configuration and one adapter save, timed; writes timing.json, marks the manifest timing_only; select refuses the directory")
    t.add_argument("--dry-run-synthetic", type=int, default=0, metavar="N_ITEMS", help="CPU dry run on N fake items instead of the real data; never for results")

    u = sub.add_parser("tau", help="tau = max(1, median |m|) of the frozen base model over the original train decision prompts")
    u.add_argument("--model", required=True)
    u.add_argument("--out", required=True, help="the tau file (JSON); never overwritten")
    u.add_argument("--device", default="cpu", help="cpu (default) or an explicit cuda:N; a bare 'cuda' is refused. A CPU run hides all GPUs from the process.")
    u.add_argument("--batch-size", type=int, default=BATCH, help="rows per forward pass (the training batch size)")
    u.add_argument("--model-path", default=None, help="override the model directory of the configuration (the tiny test model)")
    u.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    u.add_argument("--attn-implementation", default="sdpa")
    u.add_argument("--dry-run-synthetic", type=int, default=0, metavar="N_ITEMS", help="CPU dry run on N fake items instead of the real data; never for results")

    s = sub.add_parser("select", help="choose a checkpoint of one (model, objective) search from the validation readouts")
    s.add_argument("--runs", nargs="+", required=True, help="the seed-0 run directories of the 4 (lr, beta) cells")
    s.add_argument("--val-out", required=True, help="root of the output of scripts/run_part3_b_eval.py read: <val-out>/<tag>/val/rows.jsonl and <tag>/val_panswer/rows.jsonl, tag = base | <run name>_step<N>; selection.json is written here")
    s.add_argument("--allow-partial", action="store_true", help="accept an incomplete grid (tests only)")
    s.add_argument("--tau-file", default=None, help="the tau file of the model: the base model files of every candidate must equal its model_files; needed to bind an sft search")

    m = sub.add_parser("merge", help="write a merged bf16 checkpoint for vLLM; accepted iff the bitwise weight check passes; then report three computations on the 4 fixed prompts")
    m.add_argument("--adapter", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--device", default="cpu", help="cpu (default) or an explicit cuda:N for the report of check 2 (three computations, one model at a time); check 1 always runs on the CPU. A bare 'cuda' is refused.")
    m.add_argument("--no-verify", action="store_true")
    m.add_argument("--hash-merged", action="store_true", help="also record the sha256 of the merged weight files (a minute for 16 GB)")

    e = sub.add_parser("estimate-memory", help="GPU memory estimate from the model configuration")
    e.add_argument("--model", required=True)
    e.add_argument("--batch", type=int, default=BATCH)
    e.add_argument("--micro-batch", type=int, default=None)
    e.add_argument("--tokens", type=int, default=400)
    e.add_argument("--no-grad-checkpointing", action="store_true")
    return p


def main(argv: Sequence[str] | None = None, deps: Deps | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in ("select", "estimate-memory"):
        prepare_device(torch.device("cpu"))  # these commands never need a GPU
    elif resolve_device(args.device).type == "cpu":
        prepare_device(torch.device("cpu"))  # before the prompt module and the libraries are imported; a GPU run settles its card inside train and merge
    if args.command == "estimate-memory":
        print(json.dumps(estimate_memory(model_entry(args.model)["path"], args.batch, args.micro_batch, args.tokens, not args.no_grad_checkpointing), indent=2))
        return 0
    if args.command == "select":
        sel = run_select(args.runs, args.val_out, args.allow_partial, expect_rows=None if args.allow_partial else (EXPECTED_VAL_ROWS, EXPECTED_VAL_ITEMS), tau_file=args.tau_file)
        print(format_selection(sel))
        return 0 if sel["status"] == "selected" else EXIT_NO_SELECTABLE
    if deps is None:
        if args.command in ("train", "tau") and args.dry_run_synthetic:
            deps = synthetic_deps(importlib.import_module("_part3_b_prompts"), n_items=args.dry_run_synthetic, per_item=4)
        else:
            deps = real_deps()
    if args.command == "tau":
        doc = compute_tau(args, deps)
        print(json.dumps({"out": args.out, "tau": doc["tau"], "median_abs_m": doc["median_abs_m"], "abs_m_quantiles": doc["abs_m_quantiles"], "wrong_side": doc["wrong_side"]}, indent=2))
        return 0
    if args.command == "train":
        manifest = train(args, deps)
        print(json.dumps({"status": manifest["status"], "out": args.out, "adapters": manifest["adapters"], "elapsed_sec": manifest["elapsed_sec"]}))
        return 0
    if args.command == "merge":
        manifest = run_merge(args.adapter, args.out, deps, args.device, verify=not args.no_verify, hash_merged=args.hash_merged)
        print(json.dumps({"out": args.out, "n_adapted_weights": manifest["n_adapted_weights"], "accepted": manifest.get("accepted"), "check1_weights": manifest.get("check1_weights"),
                          "check2_logits": manifest.get("check2_logits"), "timing": manifest["timing"]}, indent=2))
        return 0 if manifest.get("accepted", True) else EXIT_MERGE_REJECTED
    raise AssertionError(args.command)


if __name__ == "__main__":
    sys.exit(main())
