"""The run configuration of the scenario S1 (``configs/scenario_s1_v1.yaml``): what the three phases run on, and the settings that the run manifest records.

The file names the task and its split, the prompt file, the token limits of the phases, the size of a unit of work, the expert's accuracy ``a_pct`` that the decide prompts state (an integer percent, set
here after the expert phase; ``null`` until then, and the decide phase refuses a run without it), the back end, and the models: the eight end-side models and the expert, each with its directory, its
tool-call family (the native writing of the adapter's sensitivity analysis), its ``chat_template_kwargs`` (no thinking), an optional pinned tool injection and optional ``vllm.LLM`` keyword arguments.

Every key is validated (types, ranges, no unknown and no missing key, no repeated YAML key); the md5 of the file goes into the manifest of every run and a resumed run must have the same.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from nmi.paths import REPO_ROOT
from nmi.scenario.adapter import FAMILY_TAG
from nmi.scenario.prompts import A_PCT_RANGE
from nmi.scenario.protocol import PhaseSettings
from nmi.scenario.render import INJECTIONS

DEFAULT_SCENARIO_CONFIG = REPO_ROOT / "configs" / "scenario_s1_v1.yaml"
DEFAULT_CONFIRM_CONFIG = REPO_ROOT / "configs" / "scenario_s1_confirm_v1.yaml"  # the confirmatory configuration: exploratory false, the same values otherwise

_TOP_KEYS = frozenset({"meta", "task", "data", "prompts", "paths", "a_pct", "phases", "backend", "models", "exploratory", "development_reason"})
_META_KEYS = frozenset({"version", "revision", "design"})
_DATA_KEYS = frozenset({"split_dir", "source"})
_PHASE_KEYS = frozenset({"expert_max_tokens", "answer_max_tokens", "follow_up_max_tokens", "decide_max_tokens", "unit_items"})
_BACKEND_KEYS = frozenset({"kind", "dtype", "seed"})
_MODEL_KEYS = frozenset({"id", "path", "family", "chat_template_kwargs"})
_MODEL_OPTIONAL = frozenset({"tool_injection", "llm_kwargs"})
TASKS = ("triviaqa",)
BACKENDS = ("vllm_offline",)


class ScenarioConfigError(ValueError):
    """The run configuration is malformed or inconsistent."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that refuses a mapping with a repeated key (PyYAML keeps the last one)."""


def _construct_unique_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False) -> Any:
    loader.flatten_mapping(node)
    seen: set[Any] = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise ScenarioConfigError(f"duplicate key {key!r} (line {key_node.start_mark.line + 1})")
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


def _plain(value: Any) -> Any:
    """Mappings and lists as plain, JSON-ready values (the config goes into the manifest)."""
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


@dataclass(frozen=True)
class ModelEntry:
    id: str
    path: str
    family: str
    chat_template_kwargs: Mapping[str, Any]
    tool_injection: str | None = None
    llm_kwargs: Mapping[str, Any] = MappingProxyType({})
    role: str = "device"

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "path": self.path, "family": self.family, "role": self.role, "chat_template_kwargs": _plain(self.chat_template_kwargs),
            "tool_injection": self.tool_injection, "llm_kwargs": _plain(self.llm_kwargs),
        }


@dataclass(frozen=True)
class ScenarioConfig:
    path: Path
    md5: str
    version: str
    revision: str
    design: str
    task: str
    split_dir: Path
    source: Path
    prompts_path: Path
    results_root: str
    a_pct: int | None
    settings: PhaseSettings
    unit_items: int
    backend: Mapping[str, Any]
    expert: ModelEntry
    devices: tuple[ModelEntry, ...]
    exploratory: bool
    development_reason: str

    @property
    def protocol_version(self) -> str:
        return self.version

    def model(self, model_id: str, phase: str) -> ModelEntry:
        """The model of a run: the expert for the phase ``expert``, one of the eight end-side models for ``answer`` and ``decide``."""
        if phase == "expert":
            if model_id != self.expert.id:
                raise ScenarioConfigError(f"the expert phase runs the expert {self.expert.id!r}, not {model_id!r}")
            return self.expert
        for entry in self.devices:
            if entry.id == model_id:
                return entry
        raise ScenarioConfigError(f"unknown end-side model {model_id!r}; the configuration lists {[d.id for d in self.devices]}" + (f" (the expert {self.expert.id!r} is not an end-side model)" if model_id == self.expert.id else ""))

    def as_dict(self) -> dict[str, Any]:
        """The content recorded in the manifest (the paths are as written)."""
        return {
            "version": self.version, "revision": self.revision, "task": self.task, "a_pct": self.a_pct, "phases": {**self.settings.as_dict(), "unit_items": self.unit_items},
            "backend": _plain(self.backend), "expert": self.expert.as_dict(), "devices": [d.as_dict() for d in self.devices], "exploratory": self.exploratory,
            "development_reason": self.development_reason,
        }


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _anchored(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _check_text(label: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ScenarioConfigError(f"{label} must be a non-empty string")
    return value


def _model(name: str, label: str, raw: Any, *, role: str) -> ModelEntry:
    if not isinstance(raw, dict) or not (_MODEL_KEYS <= set(raw) <= _MODEL_KEYS | _MODEL_OPTIONAL):
        keys = sorted(raw) if isinstance(raw, dict) else type(raw).__name__
        raise ScenarioConfigError(f"{name}: {label} keys {keys} must be {sorted(_MODEL_KEYS)} and optionally {sorted(_MODEL_OPTIONAL)}")
    mid = _check_text(f"{name}: {label}.id", raw["id"])
    path = _check_text(f"{name}: {label}.path", raw["path"])
    if raw["family"] not in FAMILY_TAG:
        raise ScenarioConfigError(f"{name}: {label}.family {raw['family']!r} must be one of {sorted(FAMILY_TAG)}")
    kwargs = raw["chat_template_kwargs"]
    if not isinstance(kwargs, dict):
        raise ScenarioConfigError(f"{name}: {label}.chat_template_kwargs must be a mapping (the models run without thinking)")
    injection = raw.get("tool_injection")
    if injection is not None and injection not in INJECTIONS:
        raise ScenarioConfigError(f"{name}: {label}.tool_injection must be one of {list(INJECTIONS)} or absent, got {injection!r}")
    llm_kwargs = raw.get("llm_kwargs", {})
    if not isinstance(llm_kwargs, dict) or any(k in llm_kwargs for k in ("logprobs_mode", "model", "seed", "dtype")):
        raise ScenarioConfigError(f"{name}: {label}.llm_kwargs must be a mapping without logprobs_mode, model, seed and dtype (the back end sets those)")
    return ModelEntry(
        id=mid, path=path, family=raw["family"], chat_template_kwargs=MappingProxyType(dict(kwargs)), tool_injection=injection, llm_kwargs=MappingProxyType(dict(llm_kwargs)), role=role,
    )


def load_scenario_config(path: str | Path = DEFAULT_SCENARIO_CONFIG, *, expected_md5: str | None = None) -> ScenarioConfig:
    path = Path(path)
    data = path.read_bytes()
    md5 = hashlib.md5(data, usedforsecurity=False).hexdigest()
    name = path.name
    if expected_md5 is not None and md5 != expected_md5:
        raise ScenarioConfigError(f"{name}: md5 {md5} differs from the expected {expected_md5}")
    try:
        raw = yaml.load(data, Loader=_UniqueKeyLoader)  # noqa: S506
    except yaml.YAMLError as error:
        raise ScenarioConfigError(f"{name}: not valid YAML: {error}") from error
    if not isinstance(raw, dict) or set(raw) != _TOP_KEYS:
        keys = sorted(raw) if isinstance(raw, dict) else type(raw).__name__
        raise ScenarioConfigError(f"{name}: top-level keys {keys} differ from {sorted(_TOP_KEYS)}")
    meta = raw["meta"]
    if not isinstance(meta, dict) or set(meta) != _META_KEYS:
        raise ScenarioConfigError(f"{name}: meta keys must be {sorted(_META_KEYS)}")
    for key in _META_KEYS:
        _check_text(f"{name}: meta.{key}", meta[key])
    if raw["task"] not in TASKS:
        raise ScenarioConfigError(f"{name}: task must be one of {list(TASKS)}, got {raw['task']!r}")
    dataset = raw["data"]
    if not isinstance(dataset, dict) or set(dataset) != _DATA_KEYS:
        raise ScenarioConfigError(f"{name}: data keys must be {sorted(_DATA_KEYS)}")
    for key in _DATA_KEYS:
        _check_text(f"{name}: data.{key}", dataset[key])
    _check_text(f"{name}: prompts", raw["prompts"])
    paths_ = raw["paths"]
    if not isinstance(paths_, dict) or set(paths_) != {"results_root"}:
        raise ScenarioConfigError(f"{name}: paths must have exactly results_root")
    _check_text(f"{name}: paths.results_root", paths_["results_root"])
    a_pct = raw["a_pct"]
    if a_pct is not None and (not _is_int(a_pct) or not A_PCT_RANGE[0] <= a_pct <= A_PCT_RANGE[1]):
        raise ScenarioConfigError(f"{name}: a_pct must be null or an integer from {A_PCT_RANGE[0]} to {A_PCT_RANGE[1]}, got {a_pct!r}")
    phases = raw["phases"]
    if not isinstance(phases, dict) or set(phases) != _PHASE_KEYS or not all(_is_int(v) and v >= 1 for v in phases.values()):
        raise ScenarioConfigError(f"{name}: phases must have exactly {sorted(_PHASE_KEYS)}, positive integers")
    backend = raw["backend"]
    if not isinstance(backend, dict) or set(backend) != _BACKEND_KEYS or backend["kind"] not in BACKENDS or backend["dtype"] != "bfloat16" or not _is_int(backend["seed"]):
        raise ScenarioConfigError(f"{name}: backend must be {{kind: {list(BACKENDS)}, dtype: bfloat16, seed: <integer>}}")
    models = raw["models"]
    if not isinstance(models, dict) or set(models) != {"expert", "devices"} or not isinstance(models["devices"], list) or not models["devices"]:
        raise ScenarioConfigError(f"{name}: models must have exactly expert and a non-empty list devices")
    expert = _model(name, "models.expert", models["expert"], role="expert")
    devices = tuple(_model(name, f"models.devices[{i}]", d, role="device") for i, d in enumerate(models["devices"]))
    ids = [expert.id, *(d.id for d in devices)]
    if len(set(ids)) != len(ids):
        raise ScenarioConfigError(f"{name}: a model id is listed twice: {ids}")
    if not isinstance(raw["exploratory"], bool) or raw["exploratory"] is not True:
        raise ScenarioConfigError(f"{name}: exploratory must be true (every run of the scenario before the confirmation is development evidence)")
    reason = _check_text(f"{name}: development_reason", raw["development_reason"])
    return ScenarioConfig(
        path=path, md5=md5, version=meta["version"], revision=meta["revision"], design=meta["design"], task=raw["task"], split_dir=_anchored(dataset["split_dir"]), source=Path(dataset["source"]),
        prompts_path=_anchored(raw["prompts"]), results_root=paths_["results_root"], a_pct=a_pct,
        settings=PhaseSettings(phases["expert_max_tokens"], phases["answer_max_tokens"], phases["follow_up_max_tokens"], phases["decide_max_tokens"]), unit_items=phases["unit_items"],
        backend=MappingProxyType(dict(backend)), expert=expert, devices=devices, exploratory=True, development_reason=reason,
    )


# ---------------------------------------------------------------------------------------
# the confirmatory configuration
# ---------------------------------------------------------------------------------------
#
# The confirmatory batch of the scenario runs on ``configs/scenario_s1_confirm_v1.yaml``: the same task, prompts, phases, back end and models as the development configuration, ``exploratory: false`` and a
# fixed ``a_pct`` (an integer, never null). It is a type of its own: ``load_scenario_config`` still demands ``exploratory: true`` (every development run is development evidence) and refuses this file, and
# ``load_confirm_scenario_config`` demands ``exploratory: false`` and refuses the development file, so that neither entry can be pointed at the other's configuration.

_CONFIRM_TOP_KEYS = (_TOP_KEYS - {"development_reason"}) | {"confirmation_note"}


@dataclass(frozen=True)
class ConfirmScenarioConfig(ScenarioConfig):
    """The run configuration of the confirmatory batch: a ``ScenarioConfig`` with ``exploratory`` false, ``a_pct`` fixed (an integer) and a ``confirmation_note`` in place of the development reason."""

    confirmation_note: str = ""

    def as_dict(self) -> dict[str, Any]:
        """The content recorded in the manifest: that of the development configuration with ``exploratory`` false, the note, and no development reason."""
        base = super().as_dict()
        base.pop("development_reason", None)
        base["confirmation_note"] = self.confirmation_note
        return base


def load_confirm_scenario_config(path: str | Path = DEFAULT_CONFIRM_CONFIG, *, expected_md5: str | None = None) -> ConfirmScenarioConfig:
    """Load and validate the confirmatory configuration. Every key is checked as in ``load_scenario_config``; the differences are ``exploratory`` (must be false), ``a_pct`` (must be an integer: the expert's
    accuracy is fixed from the development pool and does not follow the expert's result on the sealed pool) and ``confirmation_note`` (replaces ``development_reason``)."""
    path = Path(path)
    data = path.read_bytes()
    md5 = hashlib.md5(data, usedforsecurity=False).hexdigest()
    name = path.name
    if expected_md5 is not None and md5 != expected_md5:
        raise ScenarioConfigError(f"{name}: md5 {md5} differs from the expected {expected_md5}")
    try:
        raw = yaml.load(data, Loader=_UniqueKeyLoader)  # noqa: S506
    except yaml.YAMLError as error:
        raise ScenarioConfigError(f"{name}: not valid YAML: {error}") from error
    if not isinstance(raw, dict) or set(raw) != _CONFIRM_TOP_KEYS:
        keys = sorted(raw) if isinstance(raw, dict) else type(raw).__name__
        raise ScenarioConfigError(f"{name}: top-level keys {keys} differ from {sorted(_CONFIRM_TOP_KEYS)} (the confirmatory configuration)")
    meta = raw["meta"]
    if not isinstance(meta, dict) or set(meta) != _META_KEYS:
        raise ScenarioConfigError(f"{name}: meta keys must be {sorted(_META_KEYS)}")
    for key in _META_KEYS:
        _check_text(f"{name}: meta.{key}", meta[key])
    if raw["task"] not in TASKS:
        raise ScenarioConfigError(f"{name}: task must be one of {list(TASKS)}, got {raw['task']!r}")
    dataset = raw["data"]
    if not isinstance(dataset, dict) or set(dataset) != _DATA_KEYS:
        raise ScenarioConfigError(f"{name}: data keys must be {sorted(_DATA_KEYS)}")
    for key in _DATA_KEYS:
        _check_text(f"{name}: data.{key}", dataset[key])
    _check_text(f"{name}: prompts", raw["prompts"])
    paths_ = raw["paths"]
    if not isinstance(paths_, dict) or set(paths_) != {"results_root"}:
        raise ScenarioConfigError(f"{name}: paths must have exactly results_root")
    _check_text(f"{name}: paths.results_root", paths_["results_root"])
    a_pct = raw["a_pct"]
    if not _is_int(a_pct) or not A_PCT_RANGE[0] <= a_pct <= A_PCT_RANGE[1]:
        raise ScenarioConfigError(f"{name}: a_pct must be the frozen integer from {A_PCT_RANGE[0]} to {A_PCT_RANGE[1]} (null is refused), got {a_pct!r}")
    phases = raw["phases"]
    if not isinstance(phases, dict) or set(phases) != _PHASE_KEYS or not all(_is_int(v) and v >= 1 for v in phases.values()):
        raise ScenarioConfigError(f"{name}: phases must have exactly {sorted(_PHASE_KEYS)}, positive integers")
    backend = raw["backend"]
    if not isinstance(backend, dict) or set(backend) != _BACKEND_KEYS or backend["kind"] not in BACKENDS or backend["dtype"] != "bfloat16" or not _is_int(backend["seed"]):
        raise ScenarioConfigError(f"{name}: backend must be {{kind: {list(BACKENDS)}, dtype: bfloat16, seed: <integer>}}")
    models = raw["models"]
    if not isinstance(models, dict) or set(models) != {"expert", "devices"} or not isinstance(models["devices"], list) or not models["devices"]:
        raise ScenarioConfigError(f"{name}: models must have exactly expert and a non-empty list devices")
    expert = _model(name, "models.expert", models["expert"], role="expert")
    devices = tuple(_model(name, f"models.devices[{i}]", d, role="device") for i, d in enumerate(models["devices"]))
    ids = [expert.id, *(d.id for d in devices)]
    if len(set(ids)) != len(ids):
        raise ScenarioConfigError(f"{name}: a model id is listed twice: {ids}")
    if not isinstance(raw["exploratory"], bool) or raw["exploratory"] is not False:
        raise ScenarioConfigError(f"{name}: exploratory must be false (the confirmatory configuration; the development configuration is the one with exploratory: true)")
    note = _check_text(f"{name}: confirmation_note", raw["confirmation_note"])
    return ConfirmScenarioConfig(
        path=path, md5=md5, version=meta["version"], revision=meta["revision"], design=meta["design"], task=raw["task"], split_dir=_anchored(dataset["split_dir"]), source=Path(dataset["source"]),
        prompts_path=_anchored(raw["prompts"]), results_root=paths_["results_root"], a_pct=a_pct,
        settings=PhaseSettings(phases["expert_max_tokens"], phases["answer_max_tokens"], phases["follow_up_max_tokens"], phases["decide_max_tokens"]), unit_items=phases["unit_items"],
        backend=MappingProxyType(dict(backend)), expert=expert, devices=devices, exploratory=False, development_reason="", confirmation_note=note,
    )


__all__ = ["DEFAULT_SCENARIO_CONFIG", "ModelEntry", "ScenarioConfig", "ScenarioConfigError", "load_scenario_config"]
__all__ += ["DEFAULT_CONFIRM_CONFIG", "ConfirmScenarioConfig", "load_confirm_scenario_config"]
