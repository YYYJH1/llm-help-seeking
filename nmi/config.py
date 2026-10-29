"""Loader and validator for the run configuration (``configs/v6_confirm.yaml``) and for its base versions killtest_v1, killtest_v1_1 and killtest_v1_2 (not distributed).

``load_config`` reads the file, rejects duplicate keys, validates EVERY key against a schema
(types, ranges, allowed values, unknown and missing keys), checks the cross-section
relations the protocol depends on (run counts, duplicated parameters, pool and
model references), parses the few numbers that the config states inside prose strings
(``Derived``), and returns a read-only ``Config`` that carries the md5 of the file.

Validation is by schema, not by the particular numbers of one file: a configuration with other
values loads as long as it is consistent. Every mapping key of the YAML must be visited by
the schema (``ValidationReport.visited`` records the keys it visited).

The base configuration killtest_v1_1 adds OPTIONAL keys, so that killtest_v1 loads and means what it
meant: ``paths.prereg_amendments`` (the amendment files a run binds in its manifest), ``readout.hf_cross_check.reference`` and
``.noise_calibration`` (the float32 reference and the calibrated tolerances of the HF cross-check; both or neither) and
``steering.hf_precision`` (float32 HF readouts in the steering stage). A config without them behaves as killtest_v1.

The base configuration killtest_v1_2 adds one OPTIONAL key, ``models.substitute_devices``: the entries
of the models that ``models.device_substitution_order`` names, in the format of ``models.devices``. They are looked up by id only (``Config.device`` asks
``models.devices`` first, then the substitutes) and are never part of ``models.devices``: the code that works on the whole set of devices (the number of models of the DES and of C4, the
full run of ``run_sim`` (``registered_run``), the default models of the psychometric stage) does not see them. ``loop_devices`` and ``steering_device`` can name a device of ``models.devices`` only.
A config without the key behaves as killtest_v1.

Guard: ``load_config(expected_md5=...)`` raises ``ConfigHashMismatch`` when the md5 of the file differs from the expected one.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from nmi.data.keys import ITEM_KEY_RULE, NORMALIZE_RULE
from nmi.data.registry import POOL_BANK, POOL_NAMES
from nmi.paths import (
    DEFAULT_CONFIG,
    HOLDOUT_LEDGER_NAME,
    HOLDOUT_MARKER_NAME,
    REPO_ROOT,
    ROOT_PARTITION_PREFIXES,
)
from nmi.provenance import bytes_md5
from nmi.seeds import SALT


class ConfigError(ValueError):
    """The configuration file is malformed or inconsistent."""


class ConfigHashMismatch(ConfigError):
    """The configuration file does not have the expected md5."""


# ---------------------------------------------------------------------------------------
# YAML loading without silent key overrides
# ---------------------------------------------------------------------------------------

class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that refuses a mapping with a repeated key (PyYAML keeps the last one)."""


def _construct_unique_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False):
    loader.flatten_mapping(node)
    seen: set[Any] = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise ConfigError(f"duplicate key {key!r} (line {key_node.start_mark.line + 1})")
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


# ---------------------------------------------------------------------------------------
# schema combinators
# ---------------------------------------------------------------------------------------

class _Ctx:
    def __init__(self) -> None:
        self.visited: set[str] = set()


Check = Callable[[Any, str, _Ctx], None]


def _norm(path: str) -> str:
    return re.sub(r"\[\d+\]", "[]", path)


def _join(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _fail(path: str, message: str) -> None:
    raise ConfigError(f"{path or '<root>'}: {message}")


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def S(
    *,
    pattern: str | None = None,
    one_of: Sequence[str] | None = None,
    equals: str | None = None,
    nonempty: bool = True,
) -> Check:
    """A string; optionally non-empty, matching a regex, one of a set, or a given text."""
    rx = re.compile(pattern) if pattern else None

    def check(v: Any, path: str, ctx: _Ctx) -> None:
        ctx.visited.add(_norm(path))
        if not isinstance(v, str):
            _fail(path, f"expected a string, got {type(v).__name__} {v!r}")
        if nonempty and not v.strip():
            _fail(path, "must not be empty")
        if rx is not None and not rx.fullmatch(v):
            _fail(path, f"{v!r} does not match {pattern!r}")
        if one_of is not None and v not in one_of:
            _fail(path, f"{v!r} is not one of {sorted(one_of)}")
        if equals is not None and v != equals:
            _fail(path, f"must be {equals!r}, got {v!r}")

    return check


def N(
    lo: float | None = None,
    hi: float | None = None,
    *,
    integer: bool = False,
    lo_open: bool = False,
    hi_open: bool = False,
) -> Check:
    """A finite number (an int when ``integer``), optionally bounded."""

    def check(v: Any, path: str, ctx: _Ctx) -> None:
        ctx.visited.add(_norm(path))
        if integer:
            if isinstance(v, bool) or not isinstance(v, int):
                _fail(path, f"expected an integer, got {type(v).__name__} {v!r}")
        elif not _is_number(v):
            _fail(path, f"expected a number, got {type(v).__name__} {v!r}")
        if not math.isfinite(v):
            _fail(path, f"must be finite, got {v!r}")
        if lo is not None and (v < lo or (lo_open and v == lo)):
            _fail(path, f"{v!r} must be {'>' if lo_open else '>='} {lo}")
        if hi is not None and (v > hi or (hi_open and v == hi)):
            _fail(path, f"{v!r} must be {'<' if hi_open else '<='} {hi}")

    return check


def I(lo: int | None = None, hi: int | None = None) -> Check:  # noqa: E743
    return N(lo, hi, integer=True)


def B(equals: bool | None = None) -> Check:
    def check(v: Any, path: str, ctx: _Ctx) -> None:
        ctx.visited.add(_norm(path))
        if not isinstance(v, bool):
            _fail(path, f"expected true or false, got {type(v).__name__} {v!r}")
        if equals is not None and v != equals:
            _fail(path, f"must be {equals}")

    return check


def _visit_all(v: Any, path: str, ctx: _Ctx) -> None:
    ctx.visited.add(_norm(path))
    if isinstance(v, Mapping):
        for key, value in v.items():
            _visit_all(value, _join(path, str(key)), ctx)
    elif isinstance(v, list):
        for i, value in enumerate(v):
            _visit_all(value, f"{path}[{i}]", ctx)


def ANY() -> Check:
    """Any YAML value (still counted as visited)."""

    def check(v: Any, path: str, ctx: _Ctx) -> None:
        _visit_all(v, path, ctx)

    return check


def L(
    item: Check,
    *,
    length: int | None = None,
    min_len: int = 0,
    max_len: int | None = None,
    unique: bool = False,
    increasing: bool = False,
) -> Check:
    """A list whose items satisfy ``item``."""

    def check(v: Any, path: str, ctx: _Ctx) -> None:
        ctx.visited.add(_norm(path))
        if not isinstance(v, list):
            _fail(path, f"expected a list, got {type(v).__name__}")
        if length is not None and len(v) != length:
            _fail(path, f"expected {length} items, got {len(v)}")
        if len(v) < min_len:
            _fail(path, f"needs at least {min_len} items, got {len(v)}")
        if max_len is not None and len(v) > max_len:
            _fail(path, f"allows at most {max_len} items, got {len(v)}")
        for i, element in enumerate(v):
            item(element, f"{path}[{i}]", ctx)
        if unique:
            try:
                if len(set(map(_hashable, v))) != len(v):
                    _fail(path, "items must be unique")
            except TypeError:
                pass
        if increasing and any(not (a < b) for a, b in zip(v, v[1:], strict=False)):
            _fail(path, "items must be strictly increasing")

    return check


def _hashable(v: Any) -> Any:
    return tuple(_hashable(x) for x in v) if isinstance(v, list) else v


def O(fields: Mapping[str, Check], *, optional: Sequence[str] = ()) -> Check:
    """A mapping with exactly these keys (``optional`` ones may be absent)."""
    optional_set = set(optional)

    def check(v: Any, path: str, ctx: _Ctx) -> None:
        ctx.visited.add(_norm(path))
        if not isinstance(v, dict):
            _fail(path, f"expected a mapping, got {type(v).__name__}")
        unknown = [k for k in v if k not in fields]
        if unknown:
            _fail(path, f"unknown keys {unknown}; the schema knows {sorted(fields)}")
        missing = [k for k in fields if k not in v and k not in optional_set]
        if missing:
            _fail(path, f"missing keys {missing}")
        for key, sub in fields.items():
            if key in v:
                sub(v[key], _join(path, key), ctx)

    return check


def M(value: Check, *, min_len: int = 0) -> Check:
    """A mapping with free string keys and values satisfying ``value``."""

    def check(v: Any, path: str, ctx: _Ctx) -> None:
        ctx.visited.add(_norm(path))
        if not isinstance(v, dict):
            _fail(path, f"expected a mapping, got {type(v).__name__}")
        if len(v) < min_len:
            _fail(path, f"needs at least {min_len} entries")
        for key, element in v.items():
            if not isinstance(key, str):
                _fail(path, f"key {key!r} must be a string")
            value(element, _join(path, key), ctx)

    return check


def OR(*options: Check) -> Check:
    def check(v: Any, path: str, ctx: _Ctx) -> None:
        errors = []
        for option in options:
            try:
                option(v, path, ctx)
                return
            except ConfigError as error:
                errors.append(str(error))
        _fail(path, "matches none of the allowed forms: " + " | ".join(errors))

    return check


def _tmp_root_check(v: Any, path: str, ctx: _Ctx) -> None:
    S(pattern=r"/.+")(v, path, ctx)
    for prefix in ROOT_PARTITION_PREFIXES:
        if v == prefix or v.startswith(prefix + "/"):
            _fail(path, f"{v!r} is under {prefix}, where temporary files are refused")


# ---------------------------------------------------------------------------------------
# the schema of the run configurations
# ---------------------------------------------------------------------------------------

_ID = S(pattern=r"[A-Za-z0-9][A-Za-z0-9_.\-]*")
_TEXT = S()
_ABS = S(pattern=r"/.+")
_PROB = N(0, 1)
_PROB_POS = N(0, 1, lo_open=True)  # (0, 1]
_POS = N(0, lo_open=True)
_NONNEG = N(0)
_UINT = I(0)
_PINT = I(1)
_SEED_LIST = L(_UINT, min_len=1, unique=True)
_SEED_RANGE = L(_UINT, length=2)
_POOL = S(one_of=sorted(POOL_NAMES))
_LABELS = ("STB", "OSC", "SAT", "OSC+SAT", "IND", "SAT/IND")

_DEVICE = O(
    {
        "id": _ID,
        "path": _ABS,
        "hf_repo": S(),
        "family": S(one_of=("qwen3", "gemma4")),
        "chat_template_kwargs": M(ANY()),
        "layers": _PINT,
    },
    optional=("path", "hf_repo"),
)
# ``chat_template_kwargs`` is optional for a helper (older files do not have it); when it is given the
# cross-section check below requires ``enable_thinking: false``.
_HELPER = O(
    {
        "id": _ID,
        "path": _ABS,
        "role": S(one_of=("main", "weak_exploratory")),
        "chat_template_kwargs": M(ANY()),
    },
    optional=("chat_template_kwargs",),
)
_CONDITION = L(_NONNEG, length=3)

# The reference side of the HF cross-check (HF, float32, one prompt at a time, no autocast, no TF32) and the calibration of its tolerances.
_HF_REFERENCE = O(
    {
        "dtype": S(equals="float32"),
        "batch_size": I(1, 1),
        "autocast": B(False),
        "tf32": B(False),
        "devices": L(S(one_of=("cuda", "cpu")), min_len=1, unique=True),
    }
)
_NOISE_CALIBRATION = O(
    {
        "pool": _POOL,
        "seed_cell": _ID,
        "rep": _UINT,
        "n_items": _PINT,
        "prompt": O(
            {
                "paraphrase": S(pattern=r"a\d\d"),
                "framing": S(equals="f2"),
                "layout": S(equals="standard"),
                "a_H": _PROB_POS,
                "c0": _NONNEG,
                "w": _NONNEG,
            }
        ),
        "multiplier": _POS,
        "step": _POS,
        "floor": _POS,
    }
)
# The precision of every HF readout of the steering stage.
_HF_PRECISION = O({"dtype": S(equals="float32"), "autocast": B(False), "tf32": B(False)})

# r_hat(z, w): the main estimator is logistic, the isotonic fit is kept as the sensitivity analysis.
# The ``logistic`` block is optional so that files that still say ``method: isotonic`` keep loading.
R_HAT_METHODS = ("logistic", "isotonic")
LOGISTIC_NA_CODES = frozenset(
    {
        "beta_not_positive",
        "target_outside_unit_interval",
        "threshold_outside_observed_range",
        "plateau",
        "theta_star_too_small",
        "fit_not_converged",
    }
)
_LOGISTIC = O(
    {
        "model": _TEXT,
        "z_clip": L(N(), length=2),  # [lower, upper]; the order and the range are checked in _semantic
        "threshold": _TEXT,
        "derivatives": O({"r_z": _TEXT, "r_w": _TEXT}),
        "na": L(S(), min_len=1, unique=True),  # the set is checked in _semantic
        "shape_check": _TEXT,
    }
)

_CASE = O(
    {
        "name": _ID,
        "rate": OR(_NONNEG, S()),
        "expected": L(S(one_of=_LABELS), min_len=1),
        "model": S(),
        "integrator": O({"method": S(equals="euler_maruyama"), "dt": _POS, "initial_history": _NONNEG}),
        "sigma": S(),
        "max_osc_count": _UINT,
        "base": _ID,
        "timeout_rate": _PROB,
    },
    optional=("rate", "model", "integrator", "sigma", "max_osc_count", "base", "timeout_rate"),
)

SCHEMA: Check = O(
    {
        "version": S(pattern=r"killtest_v\d+"),
        "revision": S(pattern=r"R\d+(-v\d+\.\d+)?"),  # "R" and a number; an amended file adds a version suffix
        "paths": O(
            {
                "results_root": S(),
                "evidence_root": S(),
                "tmp_root": _tmp_root_check,
                "prereg_amendments": L(S(pattern=r"protocol/[A-Za-z0-9][A-Za-z0-9_.\-]*\.md"), min_len=1, unique=True),
            },
            optional=("prereg_amendments",),
        ),
        "execution": O({"code_writing": _TEXT, "gpu_runs": _TEXT, "read_only_review": _TEXT}),
        "models": O(
            {
                "devices": L(_DEVICE, min_len=1),
                "device_substitution_order": L(_ID, min_len=1, unique=True),
                "substitute_devices": L(_DEVICE, min_len=1),  # the entries of the substitutes, looked up by id only (checked in _semantic)
                "loop_devices": L(_ID, min_len=1, unique=True),
                "steering_device": _ID,
                "helpers": L(_HELPER, min_len=1),
            },
            optional=("substitute_devices",),
        ),
        "data": O(
            {
                "item_key": S(equals=ITEM_KEY_RULE),
                "normalize": S(equals=NORMALIZE_RULE),
                "simpleqa_mc": O(
                    {
                        "source": _ABS,
                        "calibration_file": S(pattern=r".+\.csv"),
                        "evaluation_file": S(pattern=r".+\.csv"),
                        "full_file": S(pattern=r".+\.csv"),
                        # the PHASE files only; in the full file gt_answer is a 1-based option position
                        "ground_truth_column": S(equals="choice_1"),
                        "arrangement_seed": _UINT,
                    }
                ),
                "mmlu_redux": O(
                    {
                        "source": _ABS,
                        "keep_error_type": L(S(), min_len=1, unique=True),
                        "calibration_n": _PINT,
                        "evaluation_n": _PINT,
                        "stratify_by": S(equals="subject"),
                        "split_seed": _UINT,
                        "shuffled_copy_seed": _UINT,
                    }
                ),
                "holdout": O(
                    {
                        "mmlu_redux_holdout": O({"rule": _TEXT, "n": _PINT, "role": _TEXT}),
                        "simpleqa_mc_holdout": O({"rule": _TEXT, "n": _PINT, "role": _TEXT}),
                        "sealing": O(
                            {
                                "script": S(),
                                "sole_key_generator_and_pool_assigner": B(True),
                                "outputs": L(S(), min_len=1, unique=True),
                                "loaders": _TEXT,
                                "holdout_content_never_enters_experiments": B(True),
                            }
                        ),
                        "unseal_marker": S(),
                        "consumption_ledger": O(
                            {
                                "path": S(),
                                "unit": S(equals="item_key"),
                                "consumed_is_permanent": B(True),
                                "batch_may_include_multiple_models_and_arms": B(True),
                            }
                        ),
                    }
                ),
            }
        ),
        "prompts": O(
            {
                "file": S(),
                "ask_keep_max": _PINT,
                "primary_paraphrase": S(),
                "screening": O(
                    {
                        "split": S(),
                        "diagnostic_conditions": L(_CONDITION, min_len=1),
                        "middle_condition": _CONDITION,
                        "min_gate_pass_rate": _PROB_POS,
                        "ask_rate_bounds": L(_PROB, length=2),
                        "max_spearman_w": N(-1, 0, hi_open=True),
                        "rank_by": S(),
                    }
                ),
                "block_position_check": O({"paraphrase": S(), "split": S(), "conditions": S(), "role": S()}),
            }
        ),
        "grid": O(
            {
                "operating_curve": O(
                    {
                        "a_H": OR(S(equals="operating"), _PROB_POS),
                        "c0": L(_NONNEG, min_len=1, unique=True),
                        "w": L(_NONNEG, min_len=2, increasing=True),
                    }
                ),
                "psychometric": O(
                    {
                        "a_H": L(_PROB_POS, min_len=1, unique=True),
                        "c0": L(_NONNEG, min_len=1, unique=True),
                        "w": L(_NONNEG, min_len=1, increasing=True),
                    }
                ),
                "f1_paired": O(
                    {
                        "f1": O(
                            {
                                "a_H": L(_PROB_POS, min_len=1, unique=True),
                                "c0": L(_NONNEG, min_len=1, unique=True),
                                "kappa": L(_POS, min_len=1, unique=True),
                                "D_hat_s": L(_UINT, min_len=1, unique=True),
                            }
                        ),
                        "f2_matches": O(
                            {
                                "a_H": L(_PROB_POS, min_len=1, unique=True),
                                "c0": L(_NONNEG, min_len=1, unique=True),
                                "w": L(_NONNEG, min_len=1, increasing=True),
                            }
                        ),
                    }
                ),
                "displayed_w": O({"quantum": _POS, "cap": _POS}),
            }
        ),
        "readout": O(
            {
                "device_tokens": L(S(pattern=r"[0-9]"), length=5, unique=True),
                "ask_token": S(pattern=r"[0-9]"),
                "decision": S(equals="digit_conditional_argmax"),
                "vllm_offline": O(
                    {"max_tokens": _PINT, "temperature": _NONNEG, "logprob_token_ids": S(equals="device_tokens")}
                ),
                "vllm_http": O(
                    {
                        "max_tokens": _PINT,
                        "temperature": _NONNEG,
                        "logprobs": _PINT,
                        "logprob_token_ids": S(equals="device_tokens"),
                    }
                ),
                "record": L(S(), min_len=1, unique=True),
                "digit_mass_gate": _PROB_POS,
                "min_gate_pass_rate": _PROB_POS,
                "hf_cross_check": O(
                    {
                        "pool": _POOL,
                        "seed_cell": _ID,
                        "target_non_tie_items": _PINT,
                        "max_items_drawn": _PINT,
                        "near_tie_abs_margin": _POS,
                        "max_disagreements_non_tie": _UINT,
                        "reference": _HF_REFERENCE,
                        "noise_calibration": _NOISE_CALIBRATION,
                    },
                    optional=("reference", "noise_calibration"),
                ),
                "helper_cot": O(
                    {
                        "temperature": _NONNEG,
                        "answer_regex": S(),
                        "truncation_preflight": O(
                            {
                                "n_items": _PINT,
                                "split": S(),
                                "max_length_finish_rate": _PROB,
                                "max_tokens_options": L(_PINT, min_len=1, increasing=True),
                            }
                        ),
                        "max_tokens": S(equals="from_truncation_preflight"),
                    }
                ),
            }
        ),
        "calibration": O(
            {
                "method": S(equals="temperature_scaling"),
                "ece_bins": _PINT,
                "correctness_map": O(
                    {
                        "object": _TEXT,
                        "method": S(one_of=R_HAT_METHODS),
                        "sensitivity_method": S(one_of=R_HAT_METHODS),
                        "logistic": _LOGISTIC,
                        "target": S(),
                        "split": S(),
                        "fixed": O({"a_H": _PROB_POS, "c0": _NONNEG}),
                        "fitted_w": L(_NONNEG, min_len=2, increasing=True),
                        "r_w_zero_if_max_abs_diff_vs_w02_le": _POS,
                        "if_r_w_zero": S(equals="nearest_fitted_w"),
                        "else": S(),
                        "threshold": _TEXT,
                        "r_z": O(
                            {
                                "difference": S(equals="central"),
                                "step": _POS,
                                "at_support_edge": S(equals="one_sided"),
                                "na_if_le": _NONNEG,
                            }
                        ),
                        "r_w": _TEXT,
                        "equidistant_nearest_tie": S(equals="smaller_w"),
                        "theta_star": _TEXT,
                    },
                    optional=("sensitivity_method", "logistic"),
                ),
            }
        ),
        "psychometrics": O(
            {
                "model": _TEXT,
                "responsiveness": O({"theta_w": S(), "theta_c": S(), "theta_h": S()}),
                "rationality_ratio": _TEXT,
                "bootstrap": O(
                    {
                        "unit": S(equals="item"),
                        "n": _PINT,
                        "ci": N(0, 1, lo_open=True, hi_open=True),
                        "method": S(equals="percentile"),
                        "refit_r_hat": B(),
                    }
                ),
                "misfit": O({"bins": _TEXT, "severe_if": _TEXT}),
                "consistency": O({"measure": S(), "sensitivity_with_z_ask_if_below": _PROB_POS}),
            }
        ),
        "demand": O(
            {
                "empirical": O(
                    {
                        "name": S(equals="f_emp"),
                        "method": S(),
                        "lattice": _POS,
                        "smoothing": S(equals="none"),
                        "used_by": L(S(), min_len=1),
                    }
                ),
                "analytic": O(
                    {
                        "name": S(equals="f_hat"),
                        "steps": L(S(), length=6),
                        "u": S(),
                        "u_grid": O({"from": S(), "to": S(), "step": _POS}),
                        "bandwidth_h": _POS,
                        "kernel": O({"type": S(equals="gaussian"), "truncate_at": S(), "renormalize": B(True)}),
                        "derivative_w": _TEXT,
                        "sensitivity_h": L(_POS, min_len=1, unique=True),
                        "used_by": L(S(), min_len=1),
                    }
                ),
                "operating_point": O(
                    {
                        "rule": _TEXT,
                        "solver": O({"method": S(equals="bisection"), "tolerance": _POS}),
                        "light_if": _TEXT,
                        "overload_if": _TEXT,
                        "na_if": L(S(), min_len=1),
                    }
                ),
                "elasticity": _TEXT,
                "nonmonotone_item": _TEXT,
                "c1b": O({"role": S(), "report": S()}),
            }
        ),
        "steering": O(
            {
                "model": S(equals="steering_device"),
                "hf_precision": _HF_PRECISION,
                "primary_vector": O(
                    {
                        "prompt": S(),
                        "split": S(),
                        "pool": S(),
                        "select_by": S(),
                        "n_high": _PINT,
                        "n_low": _PINT,
                        "candidate_multiplier": _PINT,
                        "max_choice_share": _PROB_POS,
                        "max_choice_share_fallback": _PROB_POS,
                        "residual_norm_fraction": _POS,
                        "position": S(),
                    }
                ),
                "secondary_vector": O(
                    {
                        "prompt": S(),
                        "condition": _CONDITION,
                        "margin": _TEXT,
                        "margin_selection_temp": _POS,
                        "n_high": _PINT,
                        "n_low": _PINT,
                        "candidate_multiplier": _PINT,
                        "max_choice_share": _PROB_POS,
                        "residual_norm_fraction": _POS,
                        "role": S(),
                    }
                ),
                "layers": L(_UINT, min_len=1, increasing=True),
                "alphas": L(_POS, min_len=1, increasing=True),
                "directions": L(S(), min_len=1, unique=True),
                "items": O(
                    {"pool": _POOL, "n": _PINT, "stratify_by": S(), "seed": _UINT, "select": _PINT, "confirm": _PINT}
                ),
                "conditions": O({"w": L(_NONNEG, min_len=1, increasing=True), "a_H": _PROB_POS, "c0": _NONNEG}),
                "effect_sign": _TEXT,
                "selection": O(
                    {
                        "objective": S(),
                        "constraint": _TEXT,
                        "tie_break": L(S(), min_len=1, unique=True),
                        "none_feasible": S(),
                    }
                ),
                "random_control": O({"n_directions": _PINT, "seeds": _SEED_RANGE, "norm": S(), "role": S()}),
                "logit_bias_control": O(
                    {
                        "token": S(pattern=r"[0-9]"),
                        "sign": S(),
                        "search": O({"method": S(equals="bisection"), "range": L(N(), length=2), "tolerance_pp": _POS}),
                        "calibrated_on": S(),
                        "unreachable": S(),
                        "affects_k4": B(False),
                    }
                ),
                "bootstrap": O({"unit": S(equals="item"), "n": _PINT, "method": S(equals="percentile")}),
                "k4m": O({"model": S(), "report": L(S(), min_len=1)}),
                "permutation_null": O({"n_vectors": _PINT, "role": S()}),
                "gate": O(
                    {
                        "pass_all": L(S(), min_len=1),
                        "fail_any": L(S(), min_len=1),
                        "inconclusive": L(S(), min_len=1),
                    }
                ),
            },
            optional=("hf_precision",),
        ),
        "server_calibration": O(
            {
                "helper": _ID,
                "kv_cache_dtype": S(),
                "max_num_seqs": L(_PINT, min_len=1, unique=True),
                "concurrency_levels": L(_PINT, min_len=2, increasing=True),
                "seconds_per_level": _POS,
                "itl_w3": _TEXT,
                "interpolation": O(
                    {
                        "between_measured": S(equals="linear"),
                        "below_1": S(equals="value_at_1"),
                        "above_max": S(equals="value_at_max"),
                    }
                ),
                "record": L(S(), min_len=1),
                "decision_latency": O(
                    {"measure_under_loop_like_load": B(), "classes": _PINT, "class_value": S(equals="within_class_mean")}
                ),
                "itl_approximation_check": O({"when": S(), "max_rel_deviation": N(0, 1, lo_open=True, hi_open=True)}),
            }
        ),
        "des": O(
            {
                "pool": _POOL,
                "robustness_pool": _POOL,
                "n_devices": _PINT,
                "deadline_s": _POS,
                "window_s": _POS,
                "grid": O({"rho": S(), "kappa": L(_POS, min_len=1, increasing=True), "broadcast_period_s": L(_POS, min_len=1, increasing=True)}),
                "run": O({"duration_s": _POS, "warmup_s": _NONNEG}),
                "seeds": _SEED_LIST,
                "point_label": _TEXT,
                "scale_transform": O(
                    {"rule": _TEXT, "assert_equilibrium_rel_diff_le": N(0, 1, lo_open=True), "applied_to": S()}
                ),
                "pointwise_comparison": O({"unit": S(), "not_evaluable_if_either_IND": B(True)}),
                "transition_localisation": O({"method": S(), "role": S()}),
                "run_cap_total": _PINT,
                "cpu": O(
                    {
                        "max_processes": I(1, 32),
                        "threads_per_process": I(1, 1),
                        "nice": I(0, 19),
                        "timing_pilot": _TEXT,
                        "projection": _TEXT,
                        "report_to_author_if_projected_wall_h_above": _POS,
                        "no_silent_reduction": B(True),
                    }
                ),
                "c4_matched": O(
                    {
                        "system": S(equals="continuous_surrogate"),
                        "w_quantisation": S(equals="none"),
                        "feasibility": L(S(), min_len=1),
                        "infeasible": S(),
                        "service_length": S(),
                        "decision_latency": S(),
                        "deadline_s": _POS,
                        "broadcast_period_s": L(_POS, min_len=1, increasing=True),
                        "seeds": _SEED_LIST,
                        "labels": S(),
                        "runs": O(
                            {
                                "llm_pairs": _PINT,
                                "non_llm_control": _PINT,
                                "replay": _PINT,
                                "seeds_per_condition": _PINT,
                                "total": _PINT,
                                "scale_checks_max": _PINT,
                                "infeasible_not_run": B(True),
                            }
                        ),
                        "kappa_rule": _TEXT,
                        "D_target_s": L(_POS, min_len=1, increasing=True),
                        "rho": L(_POS, min_len=1, increasing=True),
                        "non_llm_control": O(
                            {"family": S(), "elasticities_geometric": L(_POS, length=2, increasing=True), "n": _PINT}
                        ),
                        "replay_version": O({"role": S(), "acceptance": _TEXT}),
                    }
                ),
                "mechanism_switches": L(S(), min_len=1, unique=True),
                "fresh_signal_control": B(),
            }
        ),
        "fluid": O(
            {
                "model": S(),
                "length_atoms": O(
                    {"n": _PINT, "value": _TEXT, "mass": S(), "conditioned_on": S(), "escalation_n": _PINT}
                ),
                "decision_latency_classes": O({"n": _PINT, "value": S(equals="within_class_mean")}),
                "arrivals": _TEXT,
                "dt_s": _POS,
                "conservation_tolerance": _POS,
                "broadcast": S(),
                "labels": _TEXT,
                "analytic": _TEXT,
            }
        ),
        "regimes": O(
            {
                "sample": _TEXT,
                "ask_time_for_binning": S(),
                "sat_dimension": O({"timeout_rate": _TEXT, "mean_latency": _TEXT, "rule": _TEXT}),
                "osc_dimension": O(
                    {
                        "bins": O(
                            {
                                "first_s": _POS,
                                "fallback_s": _POS,
                                "min_mean_count": _POS,
                                "ind_if_still_below": B(True),
                            }
                        ),
                        "detrend": _TEXT,
                        "cv_excess": _TEXT,
                        "period": O(
                            {
                                "method": S(equals="fft_no_window"),
                                "range_s": L(_POS, length=2, increasing=True),
                                "undetectable_if_peak_at_range_edge": B(True),
                                "min_cycles": _PINT,
                            }
                        ),
                        "acf": O({"formula": _TEXT, "lag_bins": _TEXT}),
                        "rule": L(S(), length=3),
                        "des_scale_check": O({"min_cv_ratio": _PROB_POS, "otherwise": S(equals="IND")}),
                    }
                ),
                "combined": M(S(one_of=_LABELS), min_len=6),
                "predicted_labels": O(
                    {
                        "representation": _TEXT,
                        "sat_IND_source": _TEXT,
                        "display_sat_IND": S(equals="IND*"),
                        "distance_and_eligibility_use": S(equals="tuple"),
                        "des_and_loop_sat_IND": S(equals="never"),
                    }
                ),
                "loop_label": S(),
                "sensitivity_cv": L(_POS, length=2, increasing=True),
                "validation": O(
                    {
                        "replicates": _PINT,
                        "warmup_s": _NONNEG,
                        "measure_s": _POS,
                        "arrivals": S(),
                        "base_rate_per_s": _POS,
                        "non_timeout_latency_s": _POS,
                        "timeout_latency_s": _POS,
                        "seed_cell": _ID,
                        "cases": L(_CASE, min_len=1),
                        "scaled_version": _TEXT,
                        "min_expected_share": _PROB_POS,
                    }
                ),
            }
        ),
        "loop": O(
            {
                "helper": O(
                    {
                        "model": _ID,
                        "kv_cache_dtype": S(),
                        "max_tokens": S(equals="from_truncation_preflight"),
                        "enable_log_requests": B(True),
                    }
                ),
                "device_models": S(equals="loop_devices"),
                "n_devices": _PINT,
                "warmup_s": _NONNEG,
                "measure_s": _POS,
                "deadline_s": _POS,
                "window_s": _POS,
                "reachable": O(
                    {
                        "kappa": L(_POS, min_len=1, increasing=True),
                        "broadcast_period_s": L(_POS, min_len=1, increasing=True),
                        "exclude_if_predicted_mean_count_per_10s_below": _POS,
                    }
                ),
                "distance": _TEXT,
                "conditions": O({"feasibility": _TEXT, "paired": _TEXT, "tie_break": L(S(), min_len=1, unique=True)}),
                "connection_pool": S(equals="unlimited"),
                "max_decision_latency_p95_s": _POS,
                "max_send_gap_p99_ms": _POS,
                "interference_check": O(
                    {
                        "poll_every_s": _POS,
                        "window_s": _POS,
                        "expected_itl": _TEXT,
                        "observed_itl": S(),
                        "bad_window_if": L(S(), min_len=1),
                        "skip_window_if": L(S(), min_len=1),
                        "invalid_window_if": L(S(), min_len=1),
                        "run_invalid_if": L(S(), min_len=1),
                        "prerequisite": _TEXT,
                    }
                ),
                "max_helper_error_rate": _PROB_POS,
                "follow_up_after_measure": S(),
                "timestamps": O({"client": L(S(), min_len=1, unique=True), "server": L(S(), min_len=1, unique=True)}),
                "timing": O({"preferred_max_load": _POS, "client_cpu_cores": _PINT, "record_load": B()}),
                "arrival_seeds": _SEED_LIST,
                "run_order_seed": _UINT,
            }
        ),
        "gates": O(
            {
                "K0": O(
                    {
                        "digit_mass_gate": _PROB_POS,
                        "min_gate_pass_rate": _PROB_POS,
                        "min_kept_paraphrases": _PINT,
                        "max_nonmonotone_item_share": _PROB_POS,
                        "hf_cross_check": S(),
                    }
                ),
                "K1": O({"min_models": _PINT, "ci": N(0, 1, lo_open=True, hi_open=True), "require": L(S(), min_len=1)}),
                "K2": O(
                    {
                        "i": _TEXT,
                        "ii": O(
                            {
                                "per_dimension": B(True),
                                "min_evaluable_share": _PROB_POS,
                                "min_agreement": _PROB_POS,
                                "inconclusive_if_coverage_below": B(True),
                            }
                        ),
                    }
                ),
                "K3": O(
                    {
                        "min_correct": _PINT,
                        "of": _PINT,
                        "ind_and_sat_ind_count_as_mismatch": B(True),
                        "role": S(),
                        "eligible": _TEXT,
                        "fewer_than_3_eligible_for_a_model": S(),
                    }
                ),
                "E3": O({"STB": L(S(one_of=_LABELS), min_len=1), "OSC": L(S(one_of=_LABELS), min_len=1), "SAT": L(S(one_of=_LABELS), min_len=1)}),
                "EP": O(
                    {
                        "eligibility": S(),
                        "orders": O({"osc": S(), "sat": S()}),
                        "rule": _TEXT,
                        "osc_ind_observed": S(),
                        "untestable_if_no_condition": B(True),
                    }
                ),
                "K4": S(equals="steering.gate"),
                "decision_order": L(S(), min_len=1),
            }
        ),
        "w1c_ladder": O(
            {
                "test_matrix": O(
                    {
                        "demand": _TEXT,
                        "rho": S(),
                        "kappa": L(_POS, min_len=1, increasing=True),
                        "broadcast_period_s": L(_POS, min_len=1, increasing=True),
                        "points_per_level": L(_PINT, length=6),
                        "service_curve": _TEXT,
                        "fixed_service_tokens": _POS,
                        "per_item_service": O({"dist": S(equals="lognormal"), "median_tokens": _POS, "sigma_log": _POS}),
                        "decision_latency": O({"dist": S(equals="gamma"), "mean_s": _POS, "shape": _POS}),
                        "seeds_per_point": _PINT,
                        "seed_cell": _ID,
                    }
                ),
                "runs": O(
                    {
                        "per_level": L(_PINT, length=6),
                        "total": _PINT,
                        "scale_checks_max": _PINT,
                        "reuse_across_levels": S(),
                        "fluid_only_recompute": L(S(), min_len=1),
                        "des_rerun": S(),
                    }
                ),
                "run_cap": _PINT,
                "comparison": S(),
                "levels": L(S(), length=6, unique=True),
                "level1_min_pointwise_agreement": _PROB_POS,
                "later_levels_min_pointwise_agreement": _PROB_POS,
                "cancellation_timeout_rate": O(
                    {
                        "des_point_estimate": _TEXT,
                        "fluid_point_estimate": _TEXT,
                        "set": _TEXT,
                        "metric": _TEXT,
                        "max_pp": _POS,
                        "empty_set": _TEXT,
                        "escalate_atoms_to": _PINT,
                        "still_above": _TEXT,
                    }
                ),
            }
        ),
        "seed_namespace": O(
            {
                "key": _TEXT,
                "stages": M(_UINT, min_len=1),
                "cells_stage_9": M(_UINT, min_len=1),
                "des": _SEED_LIST,
                "loop_arrivals": _SEED_LIST,
                "steering_random_directions": _SEED_RANGE,
                "development_pilot": _SEED_RANGE,
                "run_order": _UINT,
            }
        ),
        "gpu_budget_hours": _POS,
    }
)


# ---------------------------------------------------------------------------------------
# numbers stated inside prose strings
# ---------------------------------------------------------------------------------------

_NUM = r"(\d+(?:\.\d+)?)"
_RHO_RE = re.compile(rf"^\s*{_NUM}\s*\*\s*{_NUM}\*\*\(j/(\d+)\)\s*,\s*j\s*=\s*0\.\.(\d+)\s*$")
_SAT_RULE_RE = re.compile(rf"^SAT if timeout rate >= {_NUM} or mean latency >= {_NUM} T_dl;")
_NOOSC_RE = re.compile(rf"^NOOSC if CV_ex < {_NUM}$")
_OSC_RE = re.compile(
    rf"^OSC if CV_ex >= {_NUM}, period detectable, acf\(L1\) >= {_NUM} and acf\(L3\) >= {_NUM}$"
)
_DEVIATION_RE = re.compile(rf"^deviation > {_NUM}$")
_MISSING_RE = re.compile(rf"^missing polls > {_NUM}% of seconds$")
_VALID_RE = re.compile(rf"^valid windows < {_NUM}% \(monitoring insufficient\)$")
_BAD_RE = re.compile(rf"^bad windows > {_NUM}% of valid windows$")
_SET_RE = re.compile(rf"max\(fluid, des\) >= {_NUM}")
_BAND_RE = re.compile(rf"\[{_NUM}%,\s*{_NUM}%\]")
_U_RE = re.compile(rf"^log\(w \+ {_NUM}\)$")
_U_FROM_RE = re.compile(rf"^log\({_NUM}\) - {_NUM}h$")
_U_TO_RE = re.compile(rf"^log\({_NUM}\) \+ {_NUM}h$")
_TRUNCATE_RE = re.compile(rf"^{_NUM}h$")
_LADDER_S_RE = re.compile(r"s in \{([^}]*)\}")
_SEED_KEY_RE = re.compile(r"^SeedSequence\(\[stage_id, cell_id, rep, (0x[0-9A-Fa-f]+)\]\)$")

_EXPECTED_COMBINED = {
    "NOOSC,NOSAT": "STB",
    "NOOSC,SAT": "SAT",
    "OSC,NOSAT": "OSC",
    "OSC,SAT": "OSC+SAT",
    "IND,NOSAT": "IND",
    "IND,SAT": "SAT/IND",
}


@dataclass(frozen=True)
class Derived:
    """Numbers the config states inside prose strings, parsed and cross-checked."""

    rho_grid: tuple[float, ...]
    sat_timeout_rate: float  # SAT if timeout rate >= this
    sat_mean_latency_frac: float  # ... or mean latency >= this * T_dl
    cv_ex_threshold: float  # NOOSC below, OSC at or above
    acf_l1_min: float
    acf_l3_min: float
    interference_bad_deviation: float  # a 30 s window is bad above this relative ITL deviation
    interference_missing_poll_frac: float  # window invalid when more than this share of polls is missing
    interference_min_valid_window_frac: float  # run invalid when valid windows fall below this share
    interference_max_bad_window_frac: float  # run invalid when bad windows exceed this share of valid ones
    ladder_set_min_rate: float  # the screening set S needs max(fluid, des) timeout rate >= this
    ladder_ind_band: tuple[float, float]  # fluid timeout rate band mapped to SAT-dimension IND
    u_offset: float  # u = log(w + offset)
    ladder_demand_scales: tuple[float, ...]  # s in the ladder's logistic demand curves


def _first(rx: re.Pattern[str], text: str, path: str, *, search: bool = False) -> tuple[str, ...]:
    match = rx.search(text) if search else rx.match(text)
    if match is None:
        _fail(path, f"cannot read the expected numbers from {text!r}")
    return match.groups()


def _req(condition: bool, path: str, message: str) -> None:
    if not condition:
        _fail(path, message)


def _close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def _semantic(raw: Mapping[str, Any]) -> Derived:
    """Cross-section consistency and the numbers stated in prose."""
    models, data, des, loop = raw["models"], raw["data"], raw["des"], raw["loop"]
    fluid, regimes, ladder, seeds = raw["fluid"], raw["regimes"], raw["w1c_ladder"], raw["seed_namespace"]
    grid, readout, steering = raw["grid"], raw["readout"], raw["steering"]

    # --- models -------------------------------------------------------------------------
    device_ids = [d["id"] for d in models["devices"]]
    _req(len(set(device_ids)) == len(device_ids), "models.devices", "device ids must be unique")
    for i, device in enumerate(models["devices"]):
        _req(("path" in device) != ("hf_repo" in device), f"models.devices[{i}]", "give exactly one of path and hf_repo")
    _req(set(models["loop_devices"]) <= set(device_ids), "models.loop_devices", "must name devices listed in models.devices")
    _req(models["steering_device"] in device_ids, "models.steering_device", "must name a device in models.devices")
    if "substitute_devices" in models:  # the entries of the models of device_substitution_order; they are never part of models.devices
        substitute_ids = [d["id"] for d in models["substitute_devices"]]
        order = list(models["device_substitution_order"])
        _req(len(set(substitute_ids)) == len(substitute_ids), "models.substitute_devices", "device ids must be unique")
        _req(set(substitute_ids) == set(order), "models.substitute_devices", f"the ids must be exactly those of models.device_substitution_order {order}, got {substitute_ids}")
        shared = sorted(set(substitute_ids) & set(device_ids))
        _req(not shared, "models.substitute_devices", f"must not share an id with models.devices: {shared}")
        for i, device in enumerate(models["substitute_devices"]):
            _req(("path" in device) != ("hf_repo" in device), f"models.substitute_devices[{i}]", "give exactly one of path and hf_repo")
    helper_ids = [h["id"] for h in models["helpers"]]
    _req(len(set(helper_ids)) == len(helper_ids), "models.helpers", "helper ids must be unique")
    _req(sum(h["role"] == "main" for h in models["helpers"]) == 1, "models.helpers", "exactly one helper has role main")
    for i, helper in enumerate(models["helpers"]):
        if "chat_template_kwargs" in helper:  # Qwen3's template thinks by default; helpers must not
            _req(
                helper["chat_template_kwargs"].get("enable_thinking") is False,
                f"models.helpers[{i}].chat_template_kwargs.enable_thinking",
                "must be present and false: the kill test runs every helper in non-thinking mode",
            )
    _req(raw["server_calibration"]["helper"] in helper_ids, "server_calibration.helper", "must name a helper")
    _req(loop["helper"]["model"] in helper_ids, "loop.helper.model", "must name a helper")

    # --- data and holdout -----------------------------------------------------------------
    holdout = data["holdout"]
    _req(holdout["sealing"]["script"] == "scripts/seal_holdout.py", "data.holdout.sealing.script", "must be scripts/seal_holdout.py")
    results_root = raw["paths"]["results_root"]
    _req(
        holdout["unseal_marker"] == f"{results_root}/{HOLDOUT_MARKER_NAME}",
        "data.holdout.unseal_marker",
        f"must be {results_root}/{HOLDOUT_MARKER_NAME}",
    )
    _req(
        holdout["consumption_ledger"]["path"] == f"{results_root}/{HOLDOUT_LEDGER_NAME}",
        "data.holdout.consumption_ledger.path",
        f"must be {results_root}/{HOLDOUT_LEDGER_NAME}",
    )

    # --- prompts and grid -----------------------------------------------------------------
    screening = raw["prompts"]["screening"]
    _req(
        screening["middle_condition"] in screening["diagnostic_conditions"],
        "prompts.screening.middle_condition",
        "must be one of the diagnostic conditions",
    )
    _req(screening["ask_rate_bounds"][0] < screening["ask_rate_bounds"][1], "prompts.screening.ask_rate_bounds", "lower bound must be below the upper one")
    cap = grid["displayed_w"]["cap"]
    quantum = grid["displayed_w"]["quantum"]
    _req(abs(round(1.0 / quantum) * quantum - 1.0) < 1e-9, "grid.displayed_w.quantum", "must be the reciprocal of an integer")
    for name, ws in (
        ("grid.operating_curve.w", grid["operating_curve"]["w"]),
        ("grid.psychometric.w", grid["psychometric"]["w"]),
        ("grid.f1_paired.f2_matches.w", grid["f1_paired"]["f2_matches"]["w"]),
    ):
        _req(ws[-1] <= cap, name, f"values must not exceed the displayed cap {cap}")
    _req(grid["operating_curve"]["w"][0] == 0, "grid.operating_curve.w", "must start at 0")
    for kappa in grid["f1_paired"]["f1"]["kappa"]:
        _req(abs(round(kappa, 2) - kappa) < 1e-9, "grid.f1_paired.f1.kappa", f"{kappa} cannot be shown with two decimals")
    for name, cs in (
        ("grid.operating_curve.c0", grid["operating_curve"]["c0"]),
        ("grid.psychometric.c0", grid["psychometric"]["c0"]),
        ("grid.f1_paired.f1.c0", grid["f1_paired"]["f1"]["c0"]),
    ):
        for c0 in cs:
            _req(abs(round(c0, 2) - c0) < 1e-9, name, f"{c0} cannot be shown with two decimals")

    # --- readout ---------------------------------------------------------------------------
    tokens = readout["device_tokens"]
    _req(readout["ask_token"] in tokens, "readout.ask_token", "must be one of readout.device_tokens")
    _req(readout["vllm_http"]["logprobs"] == len(tokens), "readout.vllm_http.logprobs", "must equal the number of device tokens (vLLM requires it)")
    _req(steering["logit_bias_control"]["token"] == readout["ask_token"], "steering.logit_bias_control.token", "must be the ask token")
    check = readout["hf_cross_check"]
    _req(check["target_non_tie_items"] <= check["max_items_drawn"], "readout.hf_cross_check", "target must not exceed max_items_drawn")
    _req(check["seed_cell"] in seeds["cells_stage_9"], "readout.hf_cross_check.seed_cell", "must be a cell of seed_namespace.cells_stage_9")
    # the float32 reference and the calibrated tolerances come together, or not at all
    _req(("reference" in check) == ("noise_calibration" in check), "readout.hf_cross_check", "reference and noise_calibration must be given together or both left out")
    if "noise_calibration" in check:
        noise = check["noise_calibration"]
        where = "readout.hf_cross_check.noise_calibration"
        _req(noise["seed_cell"] in seeds["cells_stage_9"], f"{where}.seed_cell", "must be a cell of seed_namespace.cells_stage_9")
        _req(noise["seed_cell"] != check["seed_cell"], f"{where}.seed_cell", "must differ from the seed cell of the evaluation draw")
        _req(noise["pool"].endswith("_cal"), f"{where}.pool", "the tolerances are measured on a calibration pool")
        _req(noise["pool"] != check["pool"], f"{where}.pool", "must differ from the pool of the evaluation draw")
        _req(POOL_BANK[noise["pool"]] == POOL_BANK[check["pool"]], f"{where}.pool", "must be a pool of the bank of the evaluation draw")
        if noise["pool"] == "mmlu_redux_cal":
            _req(noise["n_items"] <= data["mmlu_redux"]["calibration_n"], f"{where}.n_items", "must not exceed data.mmlu_redux.calibration_n")
        _req(_close(round(noise["floor"] / noise["step"]) * noise["step"], noise["floor"]) and noise["floor"] >= noise["step"], f"{where}.floor",
             "must be a whole number (at least one) of steps")
        _req(noise["prompt"]["paraphrase"] in {f"a{i:02d}" for i in range(10)}, f"{where}.prompt.paraphrase", "must be one of the ten registered paraphrases")
        _req(noise["prompt"]["w"] <= grid["displayed_w"]["cap"], f"{where}.prompt.w", "must not exceed the displayed cap")
    # a file listed as an amendment gets its own manifest key; two files must not share one
    if "prereg_amendments" in raw["paths"]:
        keys = [amendment_key(f) for f in raw["paths"]["prereg_amendments"]]
        _req(len(set(keys)) == len(keys), "paths.prereg_amendments", f"two files give the same manifest key {keys}")
    try:
        regex = re.compile(readout["helper_cot"]["answer_regex"])
    except re.error as error:
        _fail("readout.helper_cot.answer_regex", f"is not a valid regular expression: {error}")
    _req(regex.groups == 1, "readout.helper_cot.answer_regex", "must have exactly one capture group")

    # --- calibration -------------------------------------------------------------------------
    cmap = raw["calibration"]["correctness_map"]
    _req(0.2 in cmap["fitted_w"], "calibration.correctness_map.fitted_w", "must contain 0.2 (the reference curve of the r_w test)")
    where = "calibration.correctness_map"
    method, sensitivity = cmap["method"], cmap.get("sensitivity_method")
    if sensitivity is not None:
        _req(sensitivity != method, f"{where}.sensitivity_method", f"must differ from method ({method})")
    if method == "logistic":
        _req("logistic" in cmap, f"{where}.logistic", "is required when method is logistic")
        _req(
            sensitivity == "isotonic",
            f"{where}.sensitivity_method",
            "must be isotonic when method is logistic (the isotonic fit is the sensitivity analysis)",
        )
    if "logistic" in cmap:  # validated whenever it is present, whichever method is the main one
        logistic = cmap["logistic"]
        lower, upper = logistic["z_clip"]
        _req(
            0.25 <= lower < upper < 1,
            f"{where}.logistic.z_clip",
            f"must satisfy 0.25 <= lower < upper < 1 (z is a four-option confidence), got [{lower}, {upper}]",
        )
        codes = set(logistic["na"])
        missing, unexpected = sorted(LOGISTIC_NA_CODES - codes), sorted(codes - LOGISTIC_NA_CODES)
        _req(
            not missing and not unexpected,
            f"{where}.logistic.na",
            f"must be exactly the six reason codes {sorted(LOGISTIC_NA_CODES)}; missing {missing}, unexpected {unexpected}",
        )

    # --- demand ------------------------------------------------------------------------------
    demand = raw["demand"]
    _req(_close(demand["empirical"]["lattice"], quantum), "demand.empirical.lattice", "must equal grid.displayed_w.quantum")
    (u_offset,) = _first(_U_RE, demand["analytic"]["u"], "demand.analytic.u")
    (trunc,) = _first(_TRUNCATE_RE, demand["analytic"]["kernel"]["truncate_at"], "demand.analytic.kernel.truncate_at")
    from_arg, from_mult = _first(_U_FROM_RE, demand["analytic"]["u_grid"]["from"], "demand.analytic.u_grid.from")
    to_arg, to_mult = _first(_U_TO_RE, demand["analytic"]["u_grid"]["to"], "demand.analytic.u_grid.to")
    offset = float(u_offset)
    _req(_close(float(from_arg), offset), "demand.analytic.u_grid.from", "log argument must equal the offset of u")
    _req(_close(float(to_arg), cap + offset), "demand.analytic.u_grid.to", "log argument must equal cap + offset")
    _req(
        _close(float(from_mult), float(trunc)) and _close(float(to_mult), float(trunc)),
        "demand.analytic.u_grid",
        "grid margins must equal the kernel truncation",
    )

    # --- steering ----------------------------------------------------------------------------
    primary = steering["primary_vector"]
    _req(primary["max_choice_share"] <= primary["max_choice_share_fallback"], "steering.primary_vector", "fallback share must not be below the first share")
    for share in (primary["max_choice_share"], primary["max_choice_share_fallback"]):
        for n in (primary["n_high"], primary["n_low"]):
            _req(4 * math.floor(share * n + 1e-9) >= n, "steering.primary_vector", f"a share of {share} cannot supply {n} items over 4 options")
    layer_count = next(d["layers"] for d in models["devices"] if d["id"] == models["steering_device"])
    _req(max(steering["layers"]) < layer_count, "steering.layers", f"must be below the {layer_count} layers of the steering device")
    items = steering["items"]
    _req(items["n"] == items["select"] + items["confirm"], "steering.items", "n must equal select + confirm")
    random_control = steering["random_control"]
    lo, hi = random_control["seeds"]
    _req(lo <= hi and hi - lo + 1 == random_control["n_directions"], "steering.random_control", "seeds [first, last] must span n_directions")
    _req(random_control["seeds"] == seeds["steering_random_directions"], "steering.random_control.seeds", "must equal seed_namespace.steering_random_directions")
    bias_lo, bias_hi = steering["logit_bias_control"]["search"]["range"]
    _req(bias_lo < bias_hi, "steering.logit_bias_control.search.range", "lower end must be below the upper end")

    # --- server calibration and fluid ----------------------------------------------------------
    _req(
        raw["server_calibration"]["decision_latency"]["classes"] == fluid["decision_latency_classes"]["n"],
        "fluid.decision_latency_classes.n",
        "must equal server_calibration.decision_latency.classes",
    )
    atoms = fluid["length_atoms"]
    _req(atoms["escalation_n"] > atoms["n"], "fluid.length_atoms.escalation_n", "must exceed n")
    _req(atoms["mass"] == f"1/{atoms['n']}", "fluid.length_atoms.mass", f"must be '1/{atoms['n']}'")
    _req(f"/{atoms['n']}" in atoms["value"], "fluid.length_atoms.value", "must state p_c over the same number of atoms")
    _req(ladder["cancellation_timeout_rate"]["escalate_atoms_to"] == atoms["escalation_n"], "w1c_ladder.cancellation_timeout_rate.escalate_atoms_to", "must equal fluid.length_atoms.escalation_n")

    # --- DES ---------------------------------------------------------------------------------
    for name in ("pool", "robustness_pool"):
        _req(des[name] in POOL_NAMES, f"des.{name}", "must be a known pool")
    _req(des["n_devices"] == loop["n_devices"], "des.n_devices", "must equal loop.n_devices")
    _req(des["deadline_s"] == loop["deadline_s"], "des.deadline_s", "must equal loop.deadline_s")
    _req(des["window_s"] == loop["window_s"], "des.window_s", "must equal loop.window_s")
    _req(des["seeds"] == seeds["des"], "des.seeds", "must equal seed_namespace.des")
    _req(des["c4_matched"]["seeds"] == seeds["des"], "des.c4_matched.seeds", "must equal seed_namespace.des")
    _req(des["run"]["warmup_s"] < des["run"]["duration_s"], "des.run", "warmup must be shorter than the run")
    measure = regimes["validation"]["measure_s"]
    _req(des["run"]["duration_s"] - des["run"]["warmup_s"] >= measure, "des.run", f"the measurement part must cover {measure} s")
    _req(
        des["grid"]["kappa"] == loop["reachable"]["kappa"], "loop.reachable.kappa", "must equal des.grid.kappa"
    )
    _req(
        des["grid"]["broadcast_period_s"] == loop["reachable"]["broadcast_period_s"] == des["c4_matched"]["broadcast_period_s"],
        "loop.reachable.broadcast_period_s",
        "must equal des.grid.broadcast_period_s and des.c4_matched.broadcast_period_s",
    )
    a, b, c, d = _first(_RHO_RE, des["grid"]["rho"], "des.grid.rho")
    rho_grid = tuple(float(a) * float(b) ** (j / int(c)) for j in range(int(d) + 1))
    n_models, n_rho = len(device_ids), len(rho_grid)
    n_kappa, n_delta = len(des["grid"]["kappa"]), len(des["grid"]["broadcast_period_s"])
    n_seeds = len(des["seeds"])
    formal = n_models * n_rho * n_kappa * n_delta * n_seeds
    c4 = des["c4_matched"]
    runs = c4["runs"]
    c4_conditions = len(c4["D_target_s"]) * len(c4["rho"]) * len(c4["broadcast_period_s"])
    _req(runs["llm_pairs"] == n_models * c4_conditions, "des.c4_matched.runs.llm_pairs", f"must be models x D_target x rho x broadcast = {n_models * c4_conditions}")
    _req(runs["non_llm_control"] == c4["non_llm_control"]["n"] * c4_conditions, "des.c4_matched.runs.non_llm_control", f"must be elasticities x D_target x rho x broadcast = {c4['non_llm_control']['n'] * c4_conditions}")
    _req(runs["replay"] == runs["llm_pairs"], "des.c4_matched.runs.replay", "must equal llm_pairs")
    _req(runs["seeds_per_condition"] == len(c4["seeds"]), "des.c4_matched.runs.seeds_per_condition", "must equal the number of C4 seeds")
    c4_total = (runs["llm_pairs"] + runs["non_llm_control"] + runs["replay"]) * runs["seeds_per_condition"]
    _req(runs["total"] == c4_total, "des.c4_matched.runs.total", f"must be (pairs + control + replay) x seeds = {c4_total}")
    _req(runs["scale_checks_max"] == runs["total"], "des.c4_matched.runs.scale_checks_max", "must equal total")
    _req(
        2 * formal + 2 * runs["total"] <= des["run_cap_total"],
        "des.run_cap_total",
        f"{des['run_cap_total']} cannot hold the formal grid, C4 and their scale checks ({2 * formal + 2 * runs['total']})",
    )

    # --- regimes -----------------------------------------------------------------------------
    sat_timeout, sat_latency = _first(_SAT_RULE_RE, regimes["sat_dimension"]["rule"], "regimes.sat_dimension.rule")
    rules = regimes["osc_dimension"]["rule"]
    (cv_no,) = _first(_NOOSC_RE, rules[0], "regimes.osc_dimension.rule[0]")
    cv_osc, acf1, acf3 = _first(_OSC_RE, rules[1], "regimes.osc_dimension.rule[1]")
    _req(rules[2] == "IND otherwise", "regimes.osc_dimension.rule[2]", "must be 'IND otherwise'")
    _req(_close(float(cv_no), float(cv_osc)), "regimes.osc_dimension.rule", "the NOOSC and OSC lines must use the same CV_ex threshold")
    low, high = regimes["sensitivity_cv"]
    _req(low < float(cv_osc) < high, "regimes.sensitivity_cv", "must bracket the CV_ex threshold")
    _req(regimes["combined"] == _EXPECTED_COMBINED, "regimes.combined", "must be the six-row table that combines the OSC and SAT dimensions")
    validation = regimes["validation"]
    _req(validation["seed_cell"] in seeds["cells_stage_9"], "regimes.validation.seed_cell", "must be a cell of seed_namespace.cells_stage_9")
    names = [case["name"] for case in validation["cases"]]
    _req(len(set(names)) == len(names), "regimes.validation.cases", "case names must be unique")
    for i, case in enumerate(validation["cases"]):
        where = f"regimes.validation.cases[{i}]"
        _req(("rate" in case) != ("base" in case), where, "give exactly one of rate and base")
        _req("base" not in case or case["base"] in names[:i], where, "base must name an earlier case")
    _req(validation["warmup_s"] == loop["warmup_s"] and validation["measure_s"] == loop["measure_s"], "regimes.validation", "warmup and measurement lengths must equal the loop's")

    # --- loop ----------------------------------------------------------------------------------
    _req(loop["arrival_seeds"] == seeds["loop_arrivals"], "loop.arrival_seeds", "must equal seed_namespace.loop_arrivals")
    _req(loop["run_order_seed"] == seeds["run_order"], "loop.run_order_seed", "must equal seed_namespace.run_order")
    check_cfg = loop["interference_check"]
    (deviation,) = _first(_DEVIATION_RE, check_cfg["bad_window_if"][0], "loop.interference_check.bad_window_if[0]")
    (missing,) = _first(_MISSING_RE, check_cfg["invalid_window_if"][0], "loop.interference_check.invalid_window_if[0]")
    (valid_min,) = _first(_VALID_RE, check_cfg["run_invalid_if"][0], "loop.interference_check.run_invalid_if[0]")
    (bad_max,) = _first(_BAD_RE, check_cfg["run_invalid_if"][1], "loop.interference_check.run_invalid_if[1]")

    # --- gates ---------------------------------------------------------------------------------
    gates = raw["gates"]
    _req(_close(gates["K0"]["digit_mass_gate"], readout["digit_mass_gate"]), "gates.K0.digit_mass_gate", "must equal readout.digit_mass_gate")
    _req(_close(gates["K0"]["min_gate_pass_rate"], readout["min_gate_pass_rate"]), "gates.K0.min_gate_pass_rate", "must equal readout.min_gate_pass_rate")
    _req(_close(gates["K1"]["ci"], raw["psychometrics"]["bootstrap"]["ci"]), "gates.K1.ci", "must equal psychometrics.bootstrap.ci")
    _req(gates["K3"]["min_correct"] <= gates["K3"]["of"], "gates.K3", "min_correct must not exceed of")

    # --- ladder --------------------------------------------------------------------------------
    matrix, ladder_runs = ladder["test_matrix"], ladder["runs"]
    (scales_text,) = _first(_LADDER_S_RE, matrix["demand"], "w1c_ladder.test_matrix.demand", search=True)
    try:
        scales = tuple(float(x) for x in scales_text.split(","))
    except ValueError:
        _fail("w1c_ladder.test_matrix.demand", f"cannot read the scale list from {scales_text!r}")
    base_points = len(scales) * n_rho * len(matrix["kappa"])
    expected_points = [base_points] * 4 + [base_points * len(matrix["broadcast_period_s"])] * 2
    _req(matrix["points_per_level"] == expected_points, "w1c_ladder.test_matrix.points_per_level", f"must be {expected_points} (scales x rho x kappa, times broadcast periods for levels 5-6)")
    per_level = [p * matrix["seeds_per_point"] for p in matrix["points_per_level"]]
    _req(ladder_runs["per_level"] == per_level, "w1c_ladder.runs.per_level", f"must be points x seeds = {per_level}")
    _req(ladder_runs["total"] == sum(per_level), "w1c_ladder.runs.total", f"must be the sum of per_level = {sum(per_level)}")
    _req(ladder_runs["scale_checks_max"] == ladder_runs["total"], "w1c_ladder.runs.scale_checks_max", "must equal total")
    _req(ladder["run_cap"] >= ladder_runs["total"] + ladder_runs["scale_checks_max"], "w1c_ladder.run_cap", "must cover the runs and their scale checks")
    _req(matrix["seed_cell"] in seeds["cells_stage_9"], "w1c_ladder.test_matrix.seed_cell", "must be a cell of seed_namespace.cells_stage_9")
    _req(ladder["level1_min_pointwise_agreement"] >= ladder["later_levels_min_pointwise_agreement"], "w1c_ladder", "level 1 must be at least as strict as the later levels")
    (set_rate,) = _first(_SET_RE, ladder["cancellation_timeout_rate"]["set"], "w1c_ladder.cancellation_timeout_rate.set", search=True)
    band_low, band_high = _first(_BAND_RE, ladder["cancellation_timeout_rate"]["still_above"], "w1c_ladder.cancellation_timeout_rate.still_above", search=True)
    _req(float(band_low) < float(band_high), "w1c_ladder.cancellation_timeout_rate.still_above", "band lower end must be below the upper end")

    # --- seed namespace ------------------------------------------------------------------------
    (salt_text,) = _first(_SEED_KEY_RE, seeds["key"], "seed_namespace.key")
    _req(int(salt_text, 16) == SALT, "seed_namespace.key", f"last word must be {SALT:#x}")
    _req(len(set(seeds["stages"].values())) == len(seeds["stages"]), "seed_namespace.stages", "stage numbers must be distinct")
    _req(len(set(seeds["cells_stage_9"].values())) == len(seeds["cells_stage_9"]), "seed_namespace.cells_stage_9", "cell numbers must be distinct")
    for key in ("des", "loop_arrivals", "steering_random", "development_and_validation"):
        _req(key in seeds["stages"], "seed_namespace.stages", f"missing stage {key!r}")
    for family in ("steering_random_directions", "development_pilot"):
        _req(seeds[family][0] <= seeds[family][1], f"seed_namespace.{family}", "[first, last] must have first <= last")

    return Derived(
        rho_grid=rho_grid,
        sat_timeout_rate=float(sat_timeout),
        sat_mean_latency_frac=float(sat_latency),
        cv_ex_threshold=float(cv_osc),
        acf_l1_min=float(acf1),
        acf_l3_min=float(acf3),
        interference_bad_deviation=float(deviation),
        interference_missing_poll_frac=float(missing) / 100.0,
        interference_min_valid_window_frac=float(valid_min) / 100.0,
        interference_max_bad_window_frac=float(bad_max) / 100.0,
        ladder_set_min_rate=float(set_rate),
        ladder_ind_band=(float(band_low) / 100.0, float(band_high) / 100.0),
        u_offset=offset,
        ladder_demand_scales=scales,
    )


# ---------------------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ValidationReport:
    visited: frozenset[str]
    derived: Derived


def validate_config(raw: Any) -> ValidationReport:
    """Validate a decoded document: schema first, then the cross-section relations."""
    ctx = _Ctx()
    SCHEMA(raw, "", ctx)
    return ValidationReport(visited=frozenset(ctx.visited), derived=_semantic(raw))


def _freeze(obj: Any) -> Any:
    if isinstance(obj, dict):
        return MappingProxyType({k: _freeze(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return tuple(_freeze(v) for v in obj)
    return obj


def amendment_key(path: str | Path) -> str:
    """The key under which the manifest of a run binds an amendment file: ``prereg_amendment_<tag>``, the tag being the file
    name without the ``KILLTEST_PREREG_`` prefix, lower case, every run of other characters one underscore (``KILLTEST_PREREG_v1_1.md`` -> ``prereg_amendment_v1_1``)."""
    stem = Path(path).stem
    tag = re.sub(r"^killtest_prereg_", "", stem, flags=re.IGNORECASE).lower()
    return "prereg_amendment_" + re.sub(r"[^a-z0-9]+", "_", tag).strip("_")


@dataclass(frozen=True)
class Config:
    """A validated, read-only configuration. Index it like a mapping: ``cfg["des"]["deadline_s"]``."""

    path: Path
    md5: str
    raw: Mapping[str, Any] = field(repr=False)
    derived: Derived = field(repr=False)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get_path(self, dotted: str) -> Any:
        """``cfg.get_path("des.cpu.nice")``; list items by index, ``models.devices.0.id``."""
        node: Any = self.raw
        for part in dotted.split("."):
            node = node[int(part)] if isinstance(node, tuple) else node[part]
        return node

    @property
    def protocol_version(self) -> str:
        return self.raw["version"]

    @property
    def revision(self) -> str:
        return self.raw["revision"]

    def assert_md5(self, expected: str) -> None:
        """Guard: the md5 of the file must equal ``expected``."""
        if self.md5 != expected:
            raise ConfigHashMismatch(
                f"{self.path.name} has md5 {self.md5}, the frozen value is {expected}; "
                "the configuration changed after freezing"
            )

    def device(self, device_id: str) -> Mapping[str, Any]:
        """The entry of an end-side model by id: ``models.devices`` first, then ``models.substitute_devices``. The substitutes are found by id only; no other
        method of this class and no ``models.devices`` loop sees them."""
        for device in self.raw["models"]["devices"]:
            if device["id"] == device_id:
                return device
        for device in self.raw["models"].get("substitute_devices", ()):
            if device["id"] == device_id:
                return device
        raise KeyError(device_id)

    def helper(self, helper_id: str) -> Mapping[str, Any]:
        for helper in self.raw["models"]["helpers"]:
            if helper["id"] == helper_id:
                return helper
        raise KeyError(helper_id)

    def prereg_amendment_files(self, *, repo_root: Path = REPO_ROOT) -> dict[str, Path]:
        """The amendment files (``paths.prereg_amendments``) by the manifest key that binds them, for example
        ``{"prereg_amendment_v1_1": <repo>/protocol/KILLTEST_PREREG_v1_1.md}``. Empty for a configuration without amendments."""
        return {amendment_key(f): repo_root / f for f in self.raw["paths"].get("prereg_amendments", ())}

    def missing_paths(self, *, repo_root: Path = REPO_ROOT, external: bool = True) -> list[str]:
        """Files and directories the config points to that do not exist (for ``--dry-run``).

        ``external=False`` checks only what lives in the repository (the prompt file and the amendment files).
        """
        problems: list[str] = []
        prompts = Path(self.raw["prompts"]["file"])
        if not (prompts if prompts.is_absolute() else repo_root / prompts).is_file():
            problems.append(f"prompts.file: {prompts}")
        for key, path in self.prereg_amendment_files(repo_root=repo_root).items():
            if not path.is_file():
                problems.append(f"paths.prereg_amendments ({key}): {path}")
        if not external:
            return problems
        sources = self.raw["data"]
        simpleqa = Path(sources["simpleqa_mc"]["source"])
        if not simpleqa.is_dir():
            problems.append(f"data.simpleqa_mc.source: {simpleqa}")
        else:
            for key in ("calibration_file", "evaluation_file", "full_file"):
                if not (simpleqa / sources["simpleqa_mc"][key]).is_file():
                    problems.append(f"data.simpleqa_mc.{key}: {simpleqa / sources['simpleqa_mc'][key]}")
        if not Path(sources["mmlu_redux"]["source"]).is_dir():
            problems.append(f"data.mmlu_redux.source: {sources['mmlu_redux']['source']}")
        for i, device in enumerate(self.raw["models"]["devices"]):
            if "path" in device and not Path(device["path"]).is_dir():
                problems.append(f"models.devices[{i}].path: {device['path']}")
        for i, device in enumerate(self.raw["models"].get("substitute_devices", ())):  # the substitute devices
            if "path" in device and not Path(device["path"]).is_dir():
                problems.append(f"models.substitute_devices[{i}].path: {device['path']}")
        for i, helper in enumerate(self.raw["models"]["helpers"]):
            if not Path(helper["path"]).is_dir():
                problems.append(f"models.helpers[{i}].path: {helper['path']}")
        return problems


def parse_config_bytes(data: bytes, path: Path = DEFAULT_CONFIG) -> tuple[dict[str, Any], ValidationReport]:
    try:
        raw = yaml.load(data, Loader=_UniqueKeyLoader)  # noqa: S506
    except yaml.YAMLError as error:
        raise ConfigError(f"{path.name}: not valid YAML: {error}") from error
    return raw, validate_config(raw)


def load_config(path: str | Path | None = None, *, expected_md5: str | None = None) -> Config:
    """Read, validate and freeze the configuration; optionally check the expected md5 first."""
    path = Path(path) if path is not None else DEFAULT_CONFIG
    data = path.read_bytes()
    digest = bytes_md5(data)
    if expected_md5 is not None and digest != expected_md5:
        raise ConfigHashMismatch(
            f"{path.name} has md5 {digest}, the frozen value is {expected_md5}; "
            "the configuration changed after freezing"
        )
    raw, report = parse_config_bytes(data, path)
    return Config(path=path, md5=digest, raw=_freeze(raw), derived=report.derived)
