#!/usr/bin/env python3
"""Evaluation harness of the training experiments: pre-specified plans, token inspection, the vLLM readout and the numeric cross-check.

    python scripts/run_part3_b_eval.py build      --out DIR [--models ...]
    python scripts/run_part3_b_eval.py inspect    --model ID --plans DIR
    python scripts/run_part3_b_eval.py read       --model ID --checkpoint base|DIR --plans DIR --sets S1,S2,... --out RUN --gpu-mem-util X [--device N]
    python scripts/run_part3_b_eval.py crosscheck --model ID --checkpoint DIR --plans DIR --out DIR --reference-device cuda|cpu --gpu-mem-util X [--device N]

    build       CPU. The plan of every set for every model (``DIR/<model>/<set>.jsonl``), their hashes, the price-extrapolation feasibility, the single-action check of ``unseen_combo``.
                The plans are built before any training.
    inspect     CPU + tokenizer. Digit tokens, labels, lengths <= 4095, the rendering of the mechanism runtime, the P-answer rendering.
    read        GPU. Offline vLLM, bf16, seed 0, max_model_len 4096. One row per planned trial.
    crosscheck  GPU or CPU float32 reference against vLLM bf16, in its five-digit, item-label form.

Nothing here prints or stores question text. Outputs hold item keys, labels, numbers and sha256 of token ids only.

DEVICE RULE. No command ever picks a GPU by default. ``build`` and ``inspect`` hide every GPU (``CUDA_VISIBLE_DEVICES=""`` is set inside the process). ``read`` and ``crosscheck`` refuse
to run unless the device is chosen explicitly: ``--device N`` (or ``cuda:N``), or ``CUDA_VISIBLE_DEVICES`` already set to exactly one device in the environment. The choice is made before any
CUDA-capable library is imported.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for _p in (str(REPO), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import _part3_b_prompts as P  # noqa: E402

SCHEMA_PLANS = "nmi.part3_b.plans/1"
SCHEMA_ROWS = "nmi.part3_b.rows/1"
SCHEMA_CROSSCHECK = "nmi.part3_b.crosscheck/1"
MAX_PROMPT_TOKENS = 4095  # max_model_len 4096 and one generated token
DIGIT_MASS_GATE = 0.5
CHUNK = 2048  # prompts per engine call
CODE_FILES = (
    "scripts/_part3_b_prompts.py", "scripts/run_part3_b_eval.py", "nmi/mechanism/prompts.py", "nmi/prompts.py", "nmi/mechanism/runtime.py", "nmi/readout/vllm_backend.py",
    "nmi/readout/hf_backend.py", "nmi/readout/digits.py", "configs/prompts/ask_v1_1.yaml", "configs/v6_confirm.yaml",
)
# the numeric cross-check; the numbers are those of configs/v6_confirm.yaml readout.hf_cross_check
CC_EVAL_KEY = (9, 2, 9001, 0xA0)
CC_CAL_KEY = (9, 8, 9001, 0xA0)
CC_TARGET = 150
CC_MAX_DRAWN = 300
CC_N_CAL = 100
CC_MULTIPLIER, CC_STEP, CC_FLOOR = 1.5, 0.25, 0.25
CC_CAL_CONDITION = dict(a_h=0.80, c0=0.0, w=0.20)  # the condition values of the noise-calibration prompt, in the neutral decision prompt (mapping 0, labels rotate by item)


# ---------------------------------------------------------------------------------------
# the device rule: a GPU is never chosen by default
# ---------------------------------------------------------------------------------------

ENVIRON = os.environ  # replaceable by a plain dict for testing, so that no test touches the real environment
_DEVICE_RE = re.compile(r"^(?:\d+|GPU-[0-9a-fA-F-]+|MIG-[0-9a-zA-Z/-]+)$")


def hide_gpus(environ: Any = None) -> None:
    """The CPU commands (``build``, ``inspect``) and the analysis never need a GPU: hide all of them inside this process."""
    env = ENVIRON if environ is None else environ
    env["CUDA_VISIBLE_DEVICES"] = ""


def parse_device(text: str) -> str:
    """One device: ``N``, ``cuda:N`` or a single GPU/MIG uuid. A list, ``all``, a negative number or anything else is refused."""
    t = str(text).strip()
    if t.startswith("cuda:"):
        t = t[len("cuda:"):]
    if not _DEVICE_RE.match(t):
        raise SystemExit(f"refused: {text!r} is not exactly one GPU (give N or cuda:N; no list, no 'all', no default)")
    return t


def choose_device(device: str | None, environ: Any = None) -> str:
    """The device of a GPU command, chosen by the caller and never by default. ``--device`` wins; otherwise ``CUDA_VISIBLE_DEVICES`` must already name exactly one
    device. An unset or empty variable, or a variable that contradicts ``--device``, stops the command BEFORE anything CUDA-capable is imported. ``--device`` is written into the environment of this process."""
    env = ENVIRON if environ is None else environ
    current = env.get("CUDA_VISIBLE_DEVICES")
    if device is not None:
        chosen = parse_device(device)
        if current not in (None, "") and parse_device(current) != chosen:
            raise SystemExit(f"refused: --device {device} contradicts CUDA_VISIBLE_DEVICES={current}")
        env["CUDA_VISIBLE_DEVICES"] = chosen
        return chosen
    if current is None:
        raise SystemExit("refused: no device chosen. Pass --device N, or set CUDA_VISIBLE_DEVICES to the one card you were assigned. No GPU is ever picked by default.")
    if current.strip() == "":
        raise SystemExit("refused: CUDA_VISIBLE_DEVICES is empty, so no GPU is visible. Pass --device N to choose the card explicitly.")
    return parse_device(current)


# ---------------------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------------------

def ensure_data_path(path: str | Path) -> Path:
    """Outputs must lie below NMI_OUTPUT_ROOT (default: unrestricted)."""
    p = Path(path).resolve()
    if not p.is_relative_to(os.environ.get("NMI_OUTPUT_ROOT", "/")):
        raise ValueError(f"output path {p} is not under NMI_OUTPUT_ROOT")
    return p


def atomic_json(path: str | Path, value: Any) -> None:
    p = ensure_data_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".partial")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(value, fh, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(p)


def json_safe(v: Any) -> Any:
    """NaN and infinities become None (``allow_nan=False`` otherwise refuses them)."""
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, (np.floating,)):
        return json_safe(float(v))
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, Mapping):
        return {str(k): json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [json_safe(x) for x in v]
    return v


def write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    p = ensure_data_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = ("\n".join(json.dumps(json_safe(r), sort_keys=True, separators=(",", ":"), allow_nan=False) for r in rows) + "\n").encode()
    tmp = p.with_name(p.name + ".partial")
    tmp.write_bytes(data)
    tmp.replace(p)
    return {"n": len(rows), "sha256": hashlib.sha256(data).hexdigest()}


def code_hashes() -> dict[str, str]:
    return {name: P.sha256_file(REPO / name) for name in CODE_FILES}


def keys_digest(keys: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest()


def load_tokenizer(model_id: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(P.model_path(model_id), local_files_only=True)


def spec_from_config() -> Any:
    from nmi.readout.digits import ReadoutSpec

    return ReadoutSpec.from_config(P._config())


def items_by_key(items: Mapping[str, Sequence[Any]] | None = None) -> dict[str, Any]:
    items = items if items is not None else P.load_items()
    out: dict[str, Any] = {}
    for pool in items.values():
        for it in pool:
            out[it.item_key] = it
    return out


def all_plan_rows(plans_dir: str | Path, model_id: str, sets: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
    return {s: P.load_plan(plans_dir, model_id, s) for s in sets}


# ---------------------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------------------

def cmd_build(args: argparse.Namespace) -> int:
    hide_gpus()
    out = ensure_data_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    models = args.models.split(",") if args.models else list(P.MODEL_IDS)
    t0 = time.time()
    if not P.DESIGN_PATH.is_file():  # the design text is not part of this release
        print(f"note: {P.DESIGN_PATH} not found; the design file check is skipped", file=sys.stderr)
    elif hashlib.md5(P.DESIGN_PATH.read_bytes()).hexdigest() != P.DESIGN_MD5:
        raise SystemExit(f"the design file {P.DESIGN_PATH} does not have the expected md5 ({P.DESIGN_MD5})")
    items = P.load_items()
    pools_doc = {name: {"n": len(v), "item_keys_sha256": keys_digest([i.item_key for i in v])} for name, v in items.items()}
    models_doc: dict[str, Any] = {}
    plan_hashes: dict[str, dict[str, str]] = {}
    for model in models:
        plans, report = P.build_model_plans(model, items)
        files = {}
        for set_name, rows in plans.items():
            info = P.write_plan_file(P.plan_path(out, model, set_name), rows)
            files[set_name] = {"n": info["n"], "sha256": info["sha256"], "n_items": len({r["item_key"] for r in rows})}
        plan_hashes[model] = {s: f["sha256"] for s, f in files.items()}
        models_doc[model] = {**report, "files": files}
        print(f"[build] {model}: {sum(f['n'] for f in files.values())} trials in {len(files)} sets ({time.time() - t0:.0f} s)", flush=True)
    plans_sha = hashlib.sha256(json.dumps(plan_hashes, sort_keys=True).encode()).hexdigest()
    manifest = {
        "schema": SCHEMA_PLANS, "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "plans_sha256": plans_sha,
        "design": {"path": str(P.DESIGN_PATH.relative_to(REPO)), "md5": P.DESIGN_MD5, "sections": "4 (and 3, 11)"},
        "code_sha256": code_hashes(),
        "pools": pools_doc, "pool_checks": getattr(P, "_ITEM_CACHE", {}).get("report"), "split": {"salt": P.SPLIT_SALT, "n_val": P.VAL_N, "prereg_salt": P.PREREG_SALT},
        "seeds": P.SEEDS, "template_date_text": P.TEMPLATE_DATE_TEXT, "assistant_prefix": P.ASSISTANT_PREFIX, "population": {
            "paraphrase": P.POPULATION_PARAPHRASE, "a_h": P.POPULATION_A_H, "c0": P.POPULATION_C0, "w": list(P.POPULATION_W),
            "source": "configs/v6_confirm.yaml grid.operating_curve; a_H = 0.85; paraphrase a00",
        },
        "models": models_doc, "python": platform.python_version(), "numpy": np.__version__,
    }
    atomic_json(out / "plans_manifest.json", manifest)
    atomic_json(out / "feasibility.json", {m: {"price_extrap": d["price_extrap"], "unseen_combo": {k: v for k, v in d["unseen_combo"].items()}} for m, d in models_doc.items()})
    print(json.dumps({"plans_sha256": plans_sha, "out": str(out)}), flush=True)
    for m, d in models_doc.items():
        pe, uc = d["price_extrap"], d["unseen_combo"]
        steps = "; ".join(f"w_max {t['w_max']:.2f}: {t['share_help_optimal']:.3f}" for t in pe["tried"])
        verdict = f"testable at w_max {pe['w_max']:.2f}" if pe["testable"] else "NOT TESTABLE"
        print(f"[feasibility] {m}: price extrapolation {verdict} (help-optimal share {steps}); unseen_combo help-optimal share {uc['share_help_optimal']:.3f}"
              f"{' SINGLE-ACTION' if uc['single_action'] else ''}; sweep tie points {d['sweep']['n_tie_points']}", flush=True)
    return 0


# ---------------------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------------------

def sample_indices(n: int, k: int, salt: str) -> list[int]:
    """A deterministic sample of ``k`` indices of ``n`` (by a hash of the index; no random state)."""
    if n <= k:
        return list(range(n))
    order = sorted(range(n), key=lambda i: hashlib.sha256(f"{salt}|{i}".encode()).hexdigest())
    return sorted(order[:k])


def inspect_rows(
    model_id: str, tokenizer: Any, rows_by_set: Mapping[str, Sequence[Mapping[str, Any]]], items: Mapping[str, Any], *, n_continuation: int = 25, n_part2: int = 8, batch: int = 512,
) -> dict[str, Any]:
    """The checks of ``inspect`` on plan rows (CPU, tokenizer). Raises ``AssertionError`` on the first failure of a hard check; softer facts go into the report."""
    from nmi.readout.digits import resolve_digit_token_ids

    report: dict[str, Any] = {"model": model_id, "sets": {}}
    digit_ids = resolve_digit_token_ids(tokenizer)  # raises when a digit is not one clean token
    report["digit_token_ids"] = list(digit_ids)
    report["digits_single_token"] = True
    tmpl = getattr(tokenizer, "chat_template", None)
    report["template_sha256"] = hashlib.sha256(tmpl.encode()).hexdigest() if isinstance(tmpl, str) else None
    report["template_reads_clock"] = bool(isinstance(tmpl, str) and "strftime_now" in tmpl)
    report["template_date_pinned"] = P.TEMPLATE_DATE_TEXT if report["template_reads_clock"] else None
    from nmi.mechanism.runtime import Runtime  # read only: the rendering of the mechanism runtime, used as the reference below

    stub = SimpleNamespace(tokenizer=tokenizer, chat_kwargs=P.template_kwargs(tokenizer, model_id), assistant_answer_prefix=P.ASSISTANT_PREFIX)
    worst = 0
    for set_name, rows in rows_by_set.items():
        texts: list[str] = []
        prefixes: list[str] = []
        label_issues = 0
        for row in rows:
            item = items[row["item_key"]]
            text, prefix = P.plan_text_and_prefix(item, row)
            texts.append(text)
            prefixes.append(prefix)
            opts, help_ = P.labels_of_row(row)
            ok = (list(opts) == list(row["option_labels"]) and help_ == row["help_label"] and help_ in P.DIGITS and sorted(list(opts) + [help_]) == list(P.DIGITS))
            label_issues += int(not ok)
        assert label_issues == 0, f"{set_name}: {label_issues} rows have inconsistent labels (help label must be one of 1..5 and the five labels distinct)"
        lengths: list[int] = []
        tails = Counter()
        sample = set(sample_indices(len(rows), n_continuation, f"{model_id}|{set_name}|cont"))
        part2 = set(sample_indices(len(rows), n_part2, f"{model_id}|{set_name}|part2")) if rows and rows[0]["family"] == "mech" else set()
        n_cont = n_p2 = 0
        for start in range(0, len(rows), batch):
            idx = range(start, min(start + batch, len(rows)))
            rendered = [P.render_text_for_model(tokenizer, texts[i], model_id, prefixes[i]) for i in idx]
            enc = tokenizer(rendered, add_special_tokens=False)["input_ids"]
            for i, text_i, ids in zip(idx, rendered, enc):
                ids = [int(t) for t in ids]
                lengths.append(len(ids))
                tails[(prefixes[i], tuple(ids[-6:]))] += 1
                assert prefixes[i] == "" or text_i.endswith(prefixes[i]), "the assistant prefix is not at the end of the rendered text"
                if i in sample:  # the digit continues the prefix as ONE token (as in the label_token_ids check of the mechanism runtime)
                    for d, did in zip(P.DIGITS, digit_ids):
                        full = [int(t) for t in tokenizer(text_i + d, add_special_tokens=False)["input_ids"]]
                        assert full[: len(ids)] == ids and len(full) == len(ids) + 1 and full[-1] == did, f"digit {d} does not continue the prompt as one token"
                    n_cont += 1
                if i in part2:  # the same ids as Runtime.prepare of the mechanism runtime
                    row = rows[i]
                    d = P.decision_prompt(items[row["item_key"]], row)
                    prep = Runtime.prepare(stub, {"text": d["text"], "spans": d["spans"], "option_labels": d["option_labels"], "help_label": d["help_label"]})
                    assert prep["input_ids"] == ids, "ids differ from Runtime.prepare of the mechanism runtime"
                    assert prep["label_ids"] == [int(digit_ids[int(x) - 1]) for x in list(d["option_labels"]) + [d["help_label"]]], "label token ids differ"
                    n_p2 += 1
        mx = max(lengths) if lengths else 0
        worst = max(worst, mx)
        assert mx <= MAX_PROMPT_TOKENS, f"{set_name}: a prompt has {mx} tokens (limit {MAX_PROMPT_TOKENS})"
        report["sets"][set_name] = {
            "n": len(rows), "family": rows[0]["family"] if rows else None, "max_tokens": mx, "mean_tokens": float(np.mean(lengths)) if lengths else None,
            "distinct_tails": len(tails), "tail_token_ids": [list(t[1]) for t in tails][:3], "prefix": sorted({t[0] for t in tails}), "continuation_checked": n_cont, "part2_prepare_checked": n_p2,
            "labels_ok": True,
        }
    report["max_tokens_all_sets"] = worst
    report["max_tokens_limit"] = MAX_PROMPT_TOKENS
    return report


def inspect_panswer(model_id: str, tokenizer: Any, items: Mapping[str, Sequence[Any]], n: int | None = None) -> dict[str, Any]:
    """The P-answer rendering equals ``calibrated_answer_chat_text`` token for token, on every validation, evaluation and ARC item (``n`` limits it for tests)."""
    out = {}
    for pool, domain in (("val", "id"), ("eval", "id"), ("arc", "ood")):
        its = list(items[pool])[: n or None]
        for it in its:
            P.assert_panswer_matches_calibrated(tokenizer, it, model_id, domain)
        out[pool] = len(its)
    return out


def cmd_inspect(args: argparse.Namespace) -> int:
    hide_gpus()
    t0 = time.time()
    tok = load_tokenizer(args.model)
    items = P.load_items()
    by_key = items_by_key(items)
    rows_by_set = all_plan_rows(args.plans, args.model, P.expand_sets(args.sets) if args.sets else P.ALL_SETS)
    report = inspect_rows(args.model, tok, rows_by_set, by_key)
    report["panswer_equals_calibrated_answer_chat_text"] = inspect_panswer(args.model, tok, items)
    report["seconds"] = round(time.time() - t0, 1)
    manifest = json.loads((Path(args.plans) / "plans_manifest.json").read_text())
    report["plans_sha256"] = manifest["plans_sha256"]
    out = Path(args.out) if args.out else Path(args.plans) / f"inspect_{args.model}.json"
    atomic_json(out, json_safe(report))
    summary = {s: (d["n"], d["max_tokens"], d["distinct_tails"]) for s, d in report["sets"].items()}
    print(json.dumps({"model": args.model, "digit_token_ids": report["digit_token_ids"], "max_tokens_all_sets": report["max_tokens_all_sets"], "sets (n, max_tokens, distinct_tails)": summary,
                      "panswer_checked": report["panswer_equals_calibrated_answer_chat_text"], "template_date_pinned": report["template_date_pinned"], "out": str(out)}, indent=1), flush=True)
    return 0


# ---------------------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------------------

def condition_of(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("domain", "a_h", "c0", "w", "x", "frame", "variant", "mapping_id", "wait_seconds", "expr", "group", "net", "tie_point", "k", "paraphrase")
    return {k: row.get(k) for k in keys if k in row}


def decision_result(plan_row: Mapping[str, Any], logprobs: Sequence[float] | None, ids: Sequence[int]) -> dict[str, Any]:
    """The output row of one decision trial. The margin uses the ITEM'S own labels; a trial without a readout is ``missing`` (no value is invented)."""
    row = {
        "schema": SCHEMA_ROWS, "trial_id": plan_row["trial_id"], "set": plan_row["set"], "item_key": plan_row["item_key"], "family": plan_row["family"], "unit": plan_row["unit"],
        "variant": plan_row["variant"], "help_label": plan_row["help_label"], "option_labels": plan_row["option_labels"], "condition": condition_of(plan_row),
        "z0": plan_row["z0"], "payoff_help": plan_row["payoff_help"], "payoff_self": plan_row["payoff_self"], "optimal": plan_row["optimal"],
        "ids_sha256": P.ids_digest(ids), "n_tokens": len(ids),
    }
    ok = logprobs is not None and len(logprobs) == 5 and all(isinstance(v, (int, float)) and not math.isnan(v) and v != math.inf for v in logprobs)
    if not ok:
        row.update(missing=True, logprobs=None, mass=None, gate_pass=False, margin=None, asked=None)
        return row
    lp = [float(v) for v in logprobs]
    mass = P.digit_mass(lp)
    margin = P.route_margin(lp, plan_row["help_label"])
    row.update(missing=False, logprobs=lp, mass=mass, gate_pass=bool(mass >= DIGIT_MASS_GATE), margin=None if math.isnan(margin) else margin, asked=bool(margin > 0))
    return row


def panswer_result(plan_row: Mapping[str, Any], logprobs: Sequence[float] | None, ids: Sequence[int], temp: float) -> dict[str, Any]:
    """The output row of one P-answer trial: five digits for the mass and the gate, the four answer digits for pred, z and correctness."""
    row = {
        "schema": SCHEMA_ROWS, "trial_id": plan_row["trial_id"], "set": plan_row["set"], "item_key": plan_row["item_key"], "family": "panswer", "unit": plan_row["unit"],
        "gt_position": plan_row["gt_position"], "z0": plan_row.get("z0"), "temperature": temp, "ids_sha256": P.ids_digest(ids), "n_tokens": len(ids),
    }
    ok = logprobs is not None and len(logprobs) == 5 and all(isinstance(v, (int, float)) and not math.isnan(v) and v != math.inf for v in logprobs)
    if not ok:
        row.update(missing=True, logprobs=None, mass=None, gate_pass=False, pred=None, z=None, correct=None)
        return row
    lp = [float(v) for v in logprobs]
    r = P.panswer_readout(lp, temp, plan_row["gt_position"])
    row.update(missing=False, logprobs=lp, mass=r["mass"], gate_pass=r["gate_pass"], pred=r["pred"], z=r["z"], correct=r["correct"])
    return row


def read_rows(
    backend: Any, model_id: str, tokenizer: Any, plan_rows: Sequence[Mapping[str, Any]], by_key: Mapping[str, Any], *, chunk: int = CHUNK, log: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Read one plan with ``backend.readouts(list of token id lists)`` (the offline vLLM backend, or any object with the same method). A chunk that raises is read prompt by
    prompt; a prompt that still raises is ``missing``. Nothing is dropped and nothing is filled in."""
    temp = P.temperature(model_id)
    out: list[dict[str, Any]] = []
    for start in range(0, len(plan_rows), chunk):
        part = plan_rows[start:start + chunk]
        ids = []
        for row in part:
            text, prefix = P.plan_text_and_prefix(by_key[row["item_key"]], row)
            ids.append(P.render_for_model(tokenizer, text, model_id, prefix))
        lps: list[Sequence[float] | None]
        try:
            lps = [tuple(r.readout.logprobs) for r in backend.readouts(ids)]
            if len(lps) != len(ids):
                raise RuntimeError("wrong number of readouts")
        except Exception as error:  # noqa: BLE001 - isolate the failing prompt(s)
            if log:
                log(f"chunk at {start} failed ({type(error).__name__}); reading prompt by prompt")
            lps = []
            for one in ids:
                try:
                    lps.append(tuple(backend.readouts([one])[0].readout.logprobs))
                except Exception:  # noqa: BLE001
                    lps.append(None)
        for row, i, lp in zip(part, ids, lps):
            out.append(panswer_result(row, lp, i, temp) if row["family"] == "panswer" else decision_result(row, lp, i))
        if log:
            log(f"{len(out)}/{len(plan_rows)}")
    return out


def checkpoint_identity(model_id: str, checkpoint: str, hash_weights: bool = False) -> dict[str, Any]:
    base = Path(P.model_path(model_id))
    path = base if checkpoint == "base" else Path(checkpoint)
    files = {}
    for f in sorted(path.iterdir()):
        if f.is_file() and (f.suffix in (".json", ".jinja", ".model") or f.suffix == ".safetensors"):
            entry: dict[str, Any] = {"bytes": f.resolve().stat().st_size}
            if f.suffix != ".safetensors" or hash_weights:
                entry["sha256"] = P.sha256_file(f)
            files[f.name] = entry
    ident = {"checkpoint": checkpoint, "model_dir": str(path.resolve()), "files": files}
    if checkpoint != "base":  # the tokenizer and the chat template must be those of the base model (a merged checkpoint carries them over unchanged)
        diffs = {}
        for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json", "generation_config.json"):
            a, b = base / name, path / name
            if a.exists() or b.exists():
                diffs[name] = bool(a.exists() and b.exists() and P.sha256_file(a) == P.sha256_file(b))
        ident["tokenizer_files_equal_to_base"] = diffs
    return ident


def shutdown_engine(engine: Any) -> None:
    """Stop an offline vLLM engine and its engine-core process (an orphaned ``VLLM::EngineCore`` keeps ~28 GB)."""
    try:
        core = getattr(getattr(engine, "llm_engine", None), "engine_core", None)
        if core is not None:
            core.shutdown()
    except Exception as error:  # noqa: BLE001
        print(f"engine shutdown: {type(error).__name__}: {error}", flush=True)
    del engine
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def open_vllm_backend(model_id: str, checkpoint: str, gpu_mem_util: float) -> tuple[Any, Any]:
    """Offline vLLM, bf16, seed 0, max_model_len 4096 (the same engine settings as in the other readout scripts). The tokenizer files are those of the BASE model, whatever the checkpoint holds."""
    if not 0.0 < gpu_mem_util <= 1.0:
        raise SystemExit("--gpu-mem-util must be in (0, 1]")
    from vllm import LLM

    from nmi.readout.vllm_backend import VLLMBackend

    base = P.model_path(model_id)
    model_dir = base if checkpoint == "base" else str(Path(checkpoint).resolve())
    tok = load_tokenizer(model_id)
    spec = spec_from_config()
    engine = LLM(model=model_dir, tokenizer=base, dtype="bfloat16", max_model_len=4096, seed=0, gpu_memory_utilization=float(gpu_mem_util))
    return VLLMBackend(engine, tok, spec), tok


def cmd_read(args: argparse.Namespace) -> int:
    device = choose_device(getattr(args, "device", None))  # before anything CUDA-capable is imported
    try:
        sets = P.expand_sets(args.sets)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    out = ensure_data_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plans_manifest = json.loads((Path(args.plans) / "plans_manifest.json").read_text())
    by_key = items_by_key()
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"schema": SCHEMA_ROWS, "sets": {}}
    if manifest.get("model") not in (None, args.model) or manifest.get("checkpoint") not in (None, args.checkpoint):
        raise SystemExit(f"{manifest_path} belongs to another model or checkpoint")
    manifest.update(
        model=args.model, checkpoint=args.checkpoint, plans_sha256=plans_manifest["plans_sha256"], code_sha256=code_hashes(), identity=checkpoint_identity(args.model, args.checkpoint, args.hash_weights),
        settings={"engine": "vllm offline", "dtype": "bfloat16", "seed": 0, "max_model_len": 4096, "gpu_memory_utilization": args.gpu_mem_util, "cuda_visible_devices": device, "assistant_prefix": P.ASSISTANT_PREFIX,
                  "template_date_text": P.TEMPLATE_DATE_TEXT, "digit_mass_gate": DIGIT_MASS_GATE, "temperature": P.temperature(args.model)},
    )
    pending = []
    for s in sets:
        plan = P.load_plan(args.plans, args.model, s)
        want = plans_manifest["models"][args.model]["files"][s]["sha256"]
        done = manifest["sets"].get(s)
        if done and done.get("status") == "complete" and done.get("plan_sha256") == want and (out / s / "rows.jsonl").exists():
            print(f"[read] {s}: already complete, skipped", flush=True)
            continue
        pending.append((s, plan, want))
    if not pending:
        atomic_json(manifest_path, json_safe(manifest))
        return 0
    backend, tok = open_vllm_backend(args.model, args.checkpoint, args.gpu_mem_util)
    try:
        for s, plan, want in pending:
            t0 = time.time()
            rows = read_rows(backend, args.model, tok, plan, by_key, log=lambda m, s=s: print(f"[read] {s}: {m}", flush=True))
            info = write_jsonl(out / s / "rows.jsonl", rows)
            n_missing = sum(r["missing"] for r in rows)
            manifest["sets"][s] = {"status": "complete", "n": info["n"], "sha256": info["sha256"], "plan_sha256": want, "n_missing": n_missing, "seconds": round(time.time() - t0, 1),
                                   "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            atomic_json(manifest_path, json_safe(manifest))
            print(f"[read] {s}: {info['n']} rows, {n_missing} missing, {time.time() - t0:.0f} s", flush=True)
    finally:
        shutdown_engine(backend.engine)
    return 0


# ---------------------------------------------------------------------------------------
# crosscheck: HF float32 reference against offline vLLM bf16, five digits, the item's own labels
# ---------------------------------------------------------------------------------------

def tolerance(e_max: float, multiplier: float = CC_MULTIPLIER, step: float = CC_STEP, floor: float = CC_FLOOR) -> float:
    """delta = step * max(floor / step, ceil(multiplier / step * e_max)): 0.25 * max(1, ceil(6 e_max)) for the numbers used here."""
    if not math.isfinite(e_max) or e_max < 0:
        raise ValueError("the largest deviation must be finite and not negative")
    return step * max(round(floor / step), math.ceil(multiplier / step * e_max))


def deviations(f_lp: np.ndarray, v_lp: np.ndarray, help_labels: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """``e_m = |m_V - m_F|`` and ``e_opt = max over the pairs of answer digits |(l_a - l_b)_V - (l_a - l_b)_F|``, per prompt, with the prompt's OWN answer digits and help digit.
    A non-finite logprob on either side gives NaN for that prompt (a technical failure for the caller)."""
    f = np.asarray(f_lp, dtype=np.float64)
    v = np.asarray(v_lp, dtype=np.float64)
    if f.shape != v.shape or f.ndim != 2 or f.shape[1] != 5 or len(help_labels) != len(f):
        raise ValueError("both sides need [n, 5] digit logprobs and one help label per prompt")
    e_m = np.full(len(f), np.nan)
    e_opt = np.full(len(f), np.nan)
    for i, h in enumerate(help_labels):
        if not (np.isfinite(f[i]).all() and np.isfinite(v[i]).all()):
            continue
        e_m[i] = abs(P.route_margin(v[i], h) - P.route_margin(f[i], h))
        ans = [int(d) - 1 for d in P.answer_labels(h)]
        best = 0.0
        for a in range(4):
            for b in range(a + 1, 4):
                best = max(best, abs((v[i][ans[a]] - v[i][ans[b]]) - (f[i][ans[a]] - f[i][ans[b]])))
        e_opt[i] = best
    return e_m, e_opt


def walk_k0_4(
    f_lp: np.ndarray, v_lp: np.ndarray, help_labels: Sequence[str], delta_m: float, delta_opt: float, *, target: int = CC_TARGET, max_drawn: int = CC_MAX_DRAWN, max_disagreements: int = 0,
) -> dict[str, Any]:
    """The agreement judgement over the drawn prompts in draw order. A prompt is taken when ``|m_F| >= delta_m`` (NaN is not taken); the walk stops at the ``target``-th taken prompt or
    after ``max_drawn`` prompts; fewer taken prompts than the target is not passed. Ask is ``m > 0``. A disagreement: the ask decisions differ, or both answer, F's two largest answer
    logprobs differ by at least ``delta_opt`` and the chosen answer labels differ (a tie goes to the lowest label). Zero disagreements pass."""
    f = np.asarray(f_lp, dtype=np.float64)
    v = np.asarray(v_lp, dtype=np.float64)
    limit = min(len(f), max_drawn)
    scanned = taken = both_answer = checkable = 0
    ask_pos: list[int] = []
    opt_pos: list[int] = []
    excluded: list[int] = []
    for pos in range(limit):
        h = help_labels[pos]
        scanned = pos + 1
        m_f = P.route_margin(f[pos], h)
        if abs(m_f) >= delta_m:  # NaN compares false
            taken += 1
            ask_f, ask_v = m_f > 0, P.route_margin(v[pos], h) > 0
            if ask_f != ask_v:
                ask_pos.append(pos)
            elif not ask_f:
                both_answer += 1
                if P.option_gap(f[pos], h) >= delta_opt:
                    checkable += 1
                    if P.top_answer_label(f[pos], h) != P.top_answer_label(v[pos], h):
                        opt_pos.append(pos)
        else:
            excluded.append(pos)
        if taken >= target:
            break
    n_bad = len(ask_pos) + len(opt_pos)
    enough = taken >= target
    passed = bool(enough and n_bad <= max_disagreements)
    if not enough:
        reason = f"only {taken} prompts with |m_F| >= delta_m ({delta_m:g}) in {scanned} scanned (target {target}): not passed"
    else:
        reason = f"{n_bad} disagreements on {taken} prompts with |m_F| >= delta_m ({delta_m:g}) (allowed {max_disagreements}; delta_opt {delta_opt:g}; {checkable} of {both_answer} answered prompts had comparable options)"
    return {
        "n_scanned": scanned, "n_taken": taken, "target": target, "delta_m": delta_m, "delta_opt": delta_opt, "n_disagree_ask": len(ask_pos), "n_disagree_option": len(opt_pos),
        "ask_positions": ask_pos, "option_positions": opt_pos, "n_excluded_scanned": len(excluded), "n_both_answer_taken": both_answer, "n_options_checkable": checkable,
        "max_disagreements": max_disagreements, "passed": passed, "reason": reason,
    }


def calibrate_tolerances(f_lp: np.ndarray, v_lp: np.ndarray, help_labels: Sequence[str]) -> dict[str, Any]:
    """The noise calibration on the calibration prompts: delta_m and delta_opt from the largest deviations. A prompt with a non-finite value is a technical failure."""
    e_m, e_opt = deviations(f_lp, v_lp, help_labels)
    bad = [int(i) for i in np.flatnonzero(~(np.isfinite(e_m) & np.isfinite(e_opt)))]
    doc: dict[str, Any] = {"n_items": len(e_m), "failures": bad, "status": "technical_failure" if bad else "ok"}
    if not bad:
        doc.update(e_m_max=float(e_m.max()), e_opt_max=float(e_opt.max()), delta_m=tolerance(float(e_m.max())), delta_opt=tolerance(float(e_opt.max())))
    return doc


def crosscheck_core(
    cal_f: np.ndarray, cal_v: np.ndarray, cal_help: Sequence[str], ev_f: np.ndarray, ev_v: np.ndarray, ev_help: Sequence[str],
) -> dict[str, Any]:
    """Calibrate on the calibration prompts, lock the tolerances, then judge the evaluation prompts (pure arrays; the commands below produce the readings)."""
    cal = calibrate_tolerances(cal_f, cal_v, cal_help)
    if cal["status"] != "ok":
        return {"noise_calibration": cal, "k0_4": {"passed": False, "reason": "technical failure of the noise calibration: " + str(cal["failures"][:5])}}
    k = walk_k0_4(ev_f, ev_v, ev_help, cal["delta_m"], cal["delta_opt"])
    e_m, e_opt = deviations(ev_f, ev_v, ev_help)
    with np.errstate(invalid="ignore"):
        overall = {
            "n_items": int(len(ev_f)), "margin_deviation_max": float(np.nanmax(e_m)) if np.isfinite(e_m).any() else None, "option_deviation_max": float(np.nanmax(e_opt)) if np.isfinite(e_opt).any() else None,
            "n_margin_deviation_above_delta_m": int((e_m > cal["delta_m"]).sum()), "n_option_deviation_above_delta_opt": int((e_opt > cal["delta_opt"]).sum()),
        }
    return {"noise_calibration": cal, "k0_4": k, "all_items": overall}


def hf_float32_backend(model_id: str, checkpoint: str, device: str) -> tuple[Any, Any]:
    """The reference side: HF float32, eval mode, batch size 1, no padding, TF32 and autocast off. Loads the CHECKPOINT (the merged model)."""
    import _c2_common as common
    from nmi.readout.hf_backend import HFBackend

    numerics = common.assert_strict_float32(common.apply_strict_float32())
    base = P.model_path(model_id)
    path = base if checkpoint == "base" else str(Path(checkpoint).resolve())
    model, loader = common.load_hf_model(path, dtype="float32", device=device)
    tok = load_tokenizer(model_id)
    return HFBackend(model, tok, spec_from_config(), device=device, batch_size=1, sort_by_length=False), {"loader": loader, "numerics": numerics}


def registry_pool_keys(pool: str) -> tuple[str, ...]:
    """The key list of a pool in registry order (the order over which the permutations are drawn)."""
    from nmi.data.registry import SplitRegistry

    return SplitRegistry.load().keys(pool)


def crosscheck_draws(cal_keys: Sequence[str], eval_keys: Sequence[str]) -> tuple[list[str], list[str]]:
    """The item keys of the calibration prompts (the first 100 of ``seeds.rng((9, 8, 9001, 0xA0)).permutation(n_cal)``) and of the evaluation prompts (the first 300 of
    ``seeds.rng((9, 2, 9001, 0xA0)).permutation(n_eval)``), in draw order, over the pools in registry order."""
    from nmi import seeds

    cal = seeds.rng(CC_CAL_KEY).permutation(len(cal_keys))[:CC_N_CAL]
    ev = seeds.rng(CC_EVAL_KEY).permutation(len(eval_keys))[:CC_MAX_DRAWN]
    return [cal_keys[int(i)] for i in cal], [eval_keys[int(i)] for i in ev]


def crosscheck_rows(model_id: str, plans_dir: str | Path, cal_keys: Sequence[str], ev_keys: Sequence[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The calibration rows (the fixed condition, neutral decision prompt, mapping 0; the help label still rotates by item) and the evaluation rows (the first ``in_dist`` trial of
    every drawn item: random conditions and both mappings, so the help label is not always 5)."""
    ev_plan = {r["item_key"]: r for r in P.load_plan(plans_dir, model_id, "in_dist") if r["k"] == 0}
    cal_rows = [{**CC_CAL_CONDITION, "frame": "neutral", "variant": "base", "mapping_id": 0, "domain": "id", "x": None, "wait_seconds": None, "item_key": k} for k in cal_keys]
    return cal_rows, [ev_plan[k] for k in ev_keys]


def cmd_crosscheck(args: argparse.Namespace) -> int:
    device = choose_device(getattr(args, "device", None))  # before anything CUDA-capable is imported
    import _c2_common as common

    t0 = time.time()
    out = ensure_data_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    by_key = items_by_key()
    tok = load_tokenizer(args.model)
    cal_keys, ev_keys = crosscheck_draws(registry_pool_keys("mmlu_redux_cal"), registry_pool_keys("mmlu_redux_eval"))
    cal_rows, ev_rows = crosscheck_rows(args.model, args.plans, cal_keys, ev_keys)

    def prompts(rows: Sequence[Mapping[str, Any]]) -> tuple[list[list[int]], list[str]]:
        ids, helps = [], []
        for r in rows:
            d = P.decision_prompt(by_key[r["item_key"]], r)
            ids.append(P.render_for_model(tok, d["text"], args.model))
            helps.append(d["help_label"])
        return ids, helps

    cal_ids, cal_help = prompts(cal_rows)
    ev_ids, ev_help = prompts(ev_rows)
    print(f"[crosscheck] {len(cal_ids)} calibration and {len(ev_ids)} evaluation prompts", flush=True)
    hf, hf_info = hf_float32_backend(args.model, args.checkpoint, args.reference_device)
    try:
        hf_cal = np.asarray([r.readout.logprobs for r in hf.readouts_from_ids(cal_ids)], dtype=np.float64)
        hf_ev = np.asarray([r.readout.logprobs for r in hf.readouts_from_ids(ev_ids)], dtype=np.float64)
    finally:
        del hf
        common.free_gpu_memory()
    backend, _ = open_vllm_backend(args.model, args.checkpoint, args.gpu_mem_util)
    try:
        v_cal = np.asarray([r.readout.logprobs for r in backend.readouts(cal_ids)], dtype=np.float64)
        v_ev = np.asarray([r.readout.logprobs for r in backend.readouts(ev_ids)], dtype=np.float64)
    finally:
        shutdown_engine(backend.engine)
    result = crosscheck_core(hf_cal, v_cal, cal_help, hf_ev, v_ev, ev_help)
    plans_manifest = json.loads((Path(args.plans) / "plans_manifest.json").read_text())
    doc = {
        "schema": SCHEMA_CROSSCHECK, "rule": "amendment v1.1 2.2 (150 of at most 300 prompts with |m_F| >= delta_m, zero disagreements), five digits, the item's own labels (R7)",
        "model": args.model, "checkpoint": args.checkpoint, "identity": checkpoint_identity(args.model, args.checkpoint),
        "reference": {"device": args.reference_device, "dtype": "float32", "batch_size": 1, **hf_info}, "checked": {"backend": "vllm_offline", "dtype": "bfloat16", "seed": 0, "max_model_len": 4096},
        "cuda_visible_devices": device,
        "calibration_key": list(CC_CAL_KEY), "evaluation_key": list(CC_EVAL_KEY), "calibration_condition": CC_CAL_CONDITION, "plans_sha256": plans_manifest["plans_sha256"],
        "code_sha256": code_hashes(), "passed": bool(result["k0_4"]["passed"]), "seconds": round(time.time() - t0, 1), **result,
    }
    atomic_json(out / "crosscheck.json", json_safe(doc))
    write_jsonl(out / "crosscheck_items.jsonl", [
        {"part": part, "pos": i, "item_key": rows[i]["item_key"], "help_label": helps[i], "hf_float32": hf_lp[i].tolist(), "vllm_bf16": v_lp[i].tolist()}
        for part, rows, helps, hf_lp, v_lp in (("calibration", cal_rows, cal_help, hf_cal, v_cal), ("evaluation", ev_rows, ev_help, hf_ev, v_ev)) for i in range(len(rows))
    ])
    print(json.dumps({"passed": doc["passed"], "reason": result["k0_4"].get("reason"), "delta_m": result["noise_calibration"].get("delta_m"), "delta_opt": result["noise_calibration"].get("delta_opt")}), flush=True)
    return 0 if doc["passed"] else 1


# ---------------------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------------------

def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="pre-specified plans for every model and set (CPU)")
    b.add_argument("--out", required=True)
    b.add_argument("--models", default=None, help="comma list (default: the three models)")
    i = sub.add_parser("inspect", help="tokenizer checks of the plans (CPU)")
    i.add_argument("--model", required=True, choices=list(P.MODEL_IDS))
    i.add_argument("--plans", required=True)
    i.add_argument("--sets", default=None)
    i.add_argument("--out", default=None)
    r = sub.add_parser("read", help="offline vLLM readout of planned sets (GPU)")
    r.add_argument("--model", required=True, choices=list(P.MODEL_IDS))
    r.add_argument("--checkpoint", required=True, help="'base' or a merged checkpoint directory")
    r.add_argument("--plans", required=True)
    r.add_argument("--sets", required=True, help="comma list of sets and groups. Sets: " + ",".join(P.ALL_SETS) + ". Groups: " + ", ".join(f"{k} = {','.join(v)}" for k, v in P.SET_GROUPS.items() if k != "all") + "; all = every set")
    r.add_argument("--out", required=True, help="RUN directory: results/part3_b_<date>/<model>/<checkpoint>")
    r.add_argument("--gpu-mem-util", type=float, required=True)
    r.add_argument("--device", default=None, help="the card: N or cuda:N. Required unless CUDA_VISIBLE_DEVICES already names exactly one device; no GPU is ever chosen by default")
    r.add_argument("--hash-weights", action="store_true", help="also record the sha256 of the weight files (slow)")
    c = sub.add_parser("crosscheck", help="HF float32 against offline vLLM bf16, five digits")
    c.add_argument("--model", required=True, choices=list(P.MODEL_IDS))
    c.add_argument("--checkpoint", required=True)
    c.add_argument("--plans", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--reference-device", default="cuda", choices=["cuda", "cpu"], help="where the float32 HF reference runs; the vLLM side needs the chosen card in either case")
    c.add_argument("--gpu-mem-util", type=float, required=True)
    c.add_argument("--device", default=None, help="the card: N or cuda:N. Required unless CUDA_VISIBLE_DEVICES already names exactly one device; no GPU is ever chosen by default")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    return {"build": cmd_build, "inspect": cmd_inspect, "read": cmd_read, "crosscheck": cmd_crosscheck}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
