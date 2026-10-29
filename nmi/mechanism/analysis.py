"""Offline mechanism analyses: the decision readout, paired bootstrap and sign-flip tests,
Holm adjustment, the intervention decomposition and the freezing of intervention sites.

All resampling and splitting operate on question IDs. Test outcomes never enter
preprocessing, hyperparameter selection, or intervention-site selection.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.special import logsumexp

# About eight float32 epsilons; only clamps numerical excursions at 0/1.
FP32_MASS_BOUNDARY_TOLERANCE = 1e-6
def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def readout(logits: Sequence[float], *, legal_mass: float = 1.0,
            correct_answer: int | None = None, valid: bool = True) -> dict:
    """Semantic order is four answers, then help; answer indexes are zero based.

    Argmax help uses strict >, preserving the project's answer-on-tie rule.
    legal_mass must be computed from full-vocabulary logits by the runtime.
    FP32 sum overshoot/undershoot within 1e-6 of 0/1 is clamped with raw_mass
    and mass_adjusted retained. This does not relax the exact .5 quality gate.
    """
    x = np.asarray(logits, dtype=float)
    if x.shape != (5,):
        raise ValueError("Expected semantic logits [answer0, answer1, answer2, answer3, help]")
    try:
        raw_mass = float(legal_mass)
    except (TypeError, ValueError):
        raw_mass = float('nan')
    mass = raw_mass
    if np.isfinite(mass):
        if -FP32_MASS_BOUNDARY_TOLERANCE <= mass < 0:
            mass = 0.
        elif 1 < mass <= 1 + FP32_MASS_BOUNDARY_TOLERANCE:
            mass = 1.
    mass_adjusted = bool(np.isfinite(raw_mass) and mass != raw_mass)
    mass_metadata = dict(raw_mass=raw_mass if np.isfinite(raw_mass) else None,
                         mass_adjusted=mass_adjusted,
                         mass_boundary_tolerance=FP32_MASS_BOUNDARY_TOLERANCE)
    finite = bool(np.isfinite(x).all() and np.isfinite(mass) and 0 <= mass <= 1)
    # The 0.5 mass criterion is exact; the boundary tolerance does not apply here.
    ok = bool(valid and finite and mass >= .5)
    if correct_answer is not None and correct_answer not in range(4):
        raise ValueError("correct_answer must be zero based in [0,3]")
    if not finite:
        return dict(valid=False, margin=None, action=None, p_help=None,
                    legal_mass=mass if np.isfinite(mass) else None,
                    best_answer=None, answer_correct=None, centered_logits=None, **mass_metadata)
    m = float(x[4] - np.max(x[:4]))
    p = np.exp(x - logsumexp(x))
    answer = int(np.argmax(x[:4]))
    return dict(valid=ok, margin=m, action=int(m > 0) if ok else None,
                legal_argmax_action=int(m > 0), p_help=float(p[4]),
                entropy=float(-np.sum(p * np.log(np.clip(p, 1e-300, 1)))),
                legal_mass=mass, best_answer=answer, **mass_metadata,
                answer_correct=None if correct_answer is None else int(answer == correct_answer),
                centered_logits=(x - x.mean()).tolist())


def normalize_row(row: Mapping) -> dict:
    r = dict(row)
    aliases = {"a_H": ("aH", "a_h", "expert_accuracy"), "c0": ("fixed_fee",),
               "w": ("cost", "waiting_cost"), "question_id": ("item_key", "qid", "canonical_key"),
               "fold": ("split",), "semantic": ("frame", "framework"),
               "label_mapping": ("mapping_id", "label_map_id"), "label": ("help_label",)}
    for name, others in aliases.items():
        if name not in r:
            for other in others:
                if other in r:
                    r[name] = r[other]
                    break
    if r.get("fold") == "validation":
        r["fold"] = "val"
    if isinstance(r.get('label_mapping'), Mapping):
        if 'label' not in r:
            labels = [str(k) for k,v in r['label_mapping'].items() if v == 'help']
            if len(labels) != 1:
                raise ValueError('Semantic label mapping must identify exactly one help label')
            r['label'] = labels[0]
        # Keep the full physical mapping for provenance and use the
        # mapping replicate ID for paired condition matching/probe targets.
        r['semantic_label_mapping'] = dict(r['label_mapping'])
        if 'mapping_id' not in r:
            raise ValueError('Full label map requires the frozen mapping_id')
        r['label_mapping'] = r['mapping_id']
    if "a_H" in r and "c0" in r and "w" in r:
        r["V_H"] = float(r["a_H"]) - float(r["c0"]) - float(r["w"])
        if r.get("z") is not None:
            r["M"] = r["V_H"] - float(r["z"])
    logits = r.get("semantic_logits", r.get("logits"))
    if logits is not None:
        mass_input = r.get("legal_mass", 1.)
        # Preserve provenance on a second normalization, but never let an old
        # archived raw mass override a newly merged matched/intervened mass.
        prior_raw = r.get("raw_mass")
        if r.get("mass_adjusted") and prior_raw is not None and mass_input is not None:
            if np.isfinite(prior_raw) and np.isfinite(mass_input):
                expected = 0. if -FP32_MASS_BOUNDARY_TOLERANCE <= prior_raw < 0 else (1. if 1 < prior_raw <= 1+FP32_MASS_BOUNDARY_TOLERANCE else prior_raw)
                if mass_input == expected:
                    mass_input = prior_raw
        r.update(readout(logits, legal_mass=mass_input,
                         correct_answer=r.get("correct_answer"), valid=r.get("valid", True)))
    return r


def paired_bootstrap(question_ids: Sequence, values: Sequence[float], *, n_boot: int = 2000,
                     seed: int = 0, confidence: float = .95) -> dict:
    """Rows within each question are averaged; resample whole question clusters.

    Supply already paired differences (including four-term DID) to preserve all
    correlations. NaNs are omitted within question and counted explicitly.
    """
    if n_boot < 1 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap parameters")
    ids, v = np.asarray(question_ids, dtype=str), np.asarray(values, dtype=float)
    if ids.shape != v.shape or v.ndim != 1:
        raise ValueError("question_ids and values must be matching vectors")
    means = []
    for q in np.unique(ids):
        a = v[ids == q]
        a = a[np.isfinite(a)]
        if len(a):
            means.append(float(a.mean()))
    if not means:
        return dict(estimate=None, ci=None, n_questions=0, n_rows=len(v), n_finite_rows=0,
                    n_boot=n_boot, seed=seed, confidence=confidence)
    x = np.asarray(means)
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot)
    # Bound temporary allocation independently of question count.
    for start in range(0, n_boot, 64):
        size = min(64, n_boot-start)
        samples[start:start+size] = x[rng.integers(len(x), size=(size, len(x)))].mean(axis=1)
    alpha = (1-confidence)/2
    return dict(estimate=float(x.mean()), ci=np.quantile(samples, [alpha, 1-alpha]).tolist(),
                n_questions=len(x), n_rows=len(v), n_finite_rows=int(np.isfinite(v).sum()),
                n_boot=n_boot, seed=seed, confidence=confidence)


def quality_bounds(actions: Sequence[int | None], *, question_ids: Sequence | None = None) -> dict:
    """Unknown actions may be either 0 or 1; thresholds flag quality, never delete rows."""
    a = np.array([np.nan if x is None else x for x in actions], dtype=float)
    if np.any(np.isfinite(a) & ~np.isin(a, [0, 1])):
        raise ValueError("Actions must be 0, 1, or unknown")
    q = np.arange(len(a)).astype(str) if question_ids is None else np.asarray(question_ids, dtype=str)
    if len(q) != len(a):
        raise ValueError("Question IDs length mismatch")
    if not len(a):
        return dict(n=0, valid_fraction=None, passes_95=False, passes_99=False, bounds=None)
    known = np.isfinite(a)
    low, high = np.where(known, a, 0.), np.where(known, a, 1.)
    lo = paired_bootstrap(q, low)
    hi = paired_bootstrap(q, high)
    fraction = float(known.mean())
    return dict(n=len(a), n_valid=int(known.sum()), valid_fraction=fraction,
                passes_95=fraction >= .95, passes_99=fraction >= .99,
                bounds=[lo["estimate"], hi["estimate"]],
                conservative_ci=[lo["ci"][0], hi["ci"][1]],
                common_valid_only_mean=float(a[known].mean()) if known.any() else None)


def paired_effect(question_ids: Sequence, before: Sequence[int | None],
                  after: Sequence[int | None]) -> dict:
    b = np.asarray([np.nan if x is None else x for x in before], dtype=float)
    a = np.asarray([np.nan if x is None else x for x in after], dtype=float)
    if a.shape != b.shape:
        raise ValueError("Paired action shapes differ")
    for x in (a, b):
        if np.any(np.isfinite(x) & ~np.isin(x, [0, 1])):
            raise ValueError("Actions must be binary or unknown")
    known = np.isfinite(a) & np.isfinite(b)
    lo = np.where(np.isfinite(a), a, 0)-np.where(np.isfinite(b), b, 1)
    hi = np.where(np.isfinite(a), a, 1)-np.where(np.isfinite(b), b, 0)
    return dict(common_valid=paired_bootstrap(question_ids, np.where(known, a-b, np.nan)),
                lower_bound=paired_bootstrap(question_ids, lo),
                upper_bound=paired_bootstrap(question_ids, hi),
                n_common_valid=int(known.sum()), n_pairs=len(a), units="fraction; multiply by 100 for pp")


def intervention_decomposition(before: Sequence[Mapping], after: Sequence[Mapping]) -> dict:
    if len(before) != len(after) or not before:
        raise ValueError("Nonempty paired baseline/intervention records required")
    b, a = [normalize_row(r) for r in before], [normalize_row(r) for r in after]
    for x, y in zip(b, a):
        for field in ("question_id", "condition_id", "model", "domain", "label_mapping"):
            if x.get(field) != y.get(field):
                raise ValueError(f"Intervention recipient mismatch in {field}")
    q = [r["question_id"] for r in b]
    valid = [x.get("valid", False) and y.get("valid", False) and
             x.get("centered_logits") is not None and y.get("centered_logits") is not None for x, y in zip(b, a)]
    delta_help, delta_answer, delta_margin, changed, accuracy = [], [], [], [], []
    for x, y, ok in zip(b, a, valid):
        if ok:
            xb, ya = np.asarray(x["centered_logits"]), np.asarray(y["centered_logits"])
            dh, da = float(ya[4]-xb[4]), float(ya[:4].max()-xb[:4].max())
            delta_help.append(dh); delta_answer.append(da); delta_margin.append(dh-da)
            changed.append(float(x["best_answer"] != y["best_answer"]))
        else:
            delta_help.append(np.nan); delta_answer.append(np.nan); delta_margin.append(np.nan); changed.append(np.nan)
        accuracy.append(float(y["answer_correct"])-float(x["answer_correct"])
                        if ok and x.get("answer_correct") is not None and y.get("answer_correct") is not None else np.nan)
    return dict(action=paired_effect(q, [x.get("action") for x in b], [x.get("action") for x in a]),
                delta_centered_help=paired_bootstrap(q, delta_help),
                delta_centered_max_answer=paired_bootstrap(q, delta_answer),
                delta_legal_mass=paired_bootstrap(q, [float(y['legal_mass'])-float(x['legal_mass'])
                    if x.get('legal_mass') is not None and y.get('legal_mass') is not None else np.nan
                    for x,y in zip(b,a)]),
                delta_margin=paired_bootstrap(q, delta_margin),
                answer_identity_changed=paired_bootstrap(q, changed),
                forced_answer_accuracy_delta=paired_bootstrap(q, accuracy),
                baseline_quality=quality_bounds([x.get("action") for x in b], question_ids=q),
                intervention_quality=quality_bounds([x.get("action") for x in a], question_ids=q),
                interpretation="Full-vector effects establish site-state influence; not a selective value pathway.")


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("Finite p values in [0,1] required")
    order = np.argsort(p)
    out = np.empty(len(p)); running = 0.
    for k, idx in enumerate(order):
        running = max(running, min(1., (len(p)-k)*p[idx])); out[idx] = running
    return out.tolist()


def paired_signflip(question_ids: Sequence, values: Sequence[float], *,
                    n_resamples: int = 2000, seed: int = 0) -> dict:
    """Two-sided question-level randomization control with exchangeability caveat.

    This complements, rather than substitutes for, the percentile bootstrap CI.
    All conditions must first be collapsed into the prespecified per-question
    contrast. +1 Monte Carlo correction prevents reporting an artificial p=0.
    """
    ids, values = np.asarray(question_ids, dtype=str), np.asarray(values, dtype=float)
    if ids.shape != values.shape or values.ndim != 1 or n_resamples < 1:
        raise ValueError('Matching vectors and positive resample count required')
    x = np.asarray([values[ids == q][np.isfinite(values[ids == q])].mean()
                    for q in np.unique(ids) if np.isfinite(values[ids == q]).any()])
    if not len(x):
        return dict(p_value=None, n_questions=0)
    observed = abs(float(x.mean())); rng = np.random.default_rng(seed); extremes = 0
    for start in range(0, n_resamples, 64):
        size = min(64, n_resamples-start)
        signs = rng.choice([-1.,1.],size=(size,len(x)))
        extremes += int(np.sum(np.abs((signs*x).mean(axis=1)) >= observed-1e-14))
    return dict(p_value=(extremes+1)/(n_resamples+1), n_questions=len(x),
                n_resamples=n_resamples,seed=seed,
                assumption='Question-level sign exchangeability under the null')


def freeze_sites(candidates: Sequence[Mapping], *, max_sites: int = 2,
                 provenance: Mapping | None = None) -> dict:
    """Rank pre-specified C3 dev/val scores only; no test records allowed.

    Each candidate has model/layer/position/contrast/score and fold dev or val.
    Select per model; scores must already include the prespecified control and
    quality criteria. Does not infer statistical significance from score alone.
    """
    if max_sites not in (1,2):
        raise ValueError("The bounded C3 protocol permits one or two sites per model")
    grouped = {}
    for raw in candidates:
        r = dict(raw)
        if r.get("fold") not in ("train", "dev", "val"):
            raise ValueError("Site selection must never access test/pilot results")
        if not np.isfinite(r["score"]):
            raise ValueError("Finite development selection score required")
        for key in ("model", "layer", "position", "contrast"):
            if key not in r:
                raise ValueError(f"Missing site field {key}")
        grouped.setdefault(str(r["model"]), []).append(r)
    selected = []
    for model, records in sorted(grouped.items()):
        ordered = sorted(records, key=lambda r: (-r['score'], int(r['layer']), str(r['position']), str(r['contrast'])))
        used = set()
        for r in ordered:
            key = (r['layer'], r['position'])
            if key in used:
                continue
            selected.append(r); used.add(key)
            if len(used) == max_sites:
                break
    manifest = dict(version=1, selected_sites=selected, max_sites_per_model=max_sites,
                    provenance=dict(provenance or {}), selection_data="dev/val only",
                    disclaimer="Freeze before opening independent test; selection is not causal identification.")
    manifest['sha256'] = digest(manifest)
    return manifest
