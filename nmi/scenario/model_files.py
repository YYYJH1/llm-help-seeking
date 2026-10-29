"""The model directory as the manifest of a scenario run binds it: the versions of the model weights, the tokenizer and the chat template.

``model_fingerprint`` reads the small files of a local model directory and takes the size of the large ones:

* ``config_files``: md5 of ``config.json`` and ``generation_config.json`` (the second sets the engine's stop tokens and sampling defaults);
* ``tokenizer_files``: md5 of every tokenizer file there is, the chat template files among them (``tokenizer_config.json`` holds the template of most models, ``chat_template.jinja`` or ``chat_template.json``
  of the others; the rendered text of the template is bound separately, ``scenario_run.chat_template_md5`` and ``rendering_md5``);
* ``index_files``: md5 of the shard indexes (``*.index.json``: which tensor is in which shard, and the total size);
* ``weight_files``: the NAME and SIZE of every weight file. The weights are not read and not hashed here, so a swapped shard of the same size is not seen by this fingerprint; the real run writes
  ``llm/model.json`` with the sha256 of every shard (``nmi.provenance.hash_model_files``, cached by size and modification time), and that file is the binding of the weight bytes;
* ``declared``: what ``config.json`` says of itself (architectures, model type, dtype, transformers version, quantisation method);
* ``md5``: the md5 of the four mappings above, one value that changes when any of the files changes.

The file names that count are those of ``nmi.provenance.hash_model_files``; the two lists must stay identical. Nothing here loads a model or prints a file.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from nmi.provenance import file_md5

WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf")
TOKENIZER_FILES = (
    "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "tokenizer.model", "special_tokens_map.json", "added_tokens.json", "chat_template.jinja", "chat_template.json",
)
CONFIG_FILES = ("config.json", "generation_config.json")
INDEX_SUFFIX = ".index.json"


def declared_versions(config_path: str | Path) -> dict[str, Any]:
    """What a ``config.json`` declares: architectures, model type, dtype (``torch_dtype`` or ``dtype``), the transformers version that wrote it, and the quantisation method. An unreadable or odd file gives
    nulls (the engine fails on such a file by itself; this record never stops a run)."""
    out: dict[str, Any] = {"architectures": None, "model_type": None, "dtype": None, "transformers_version": None, "quantization_method": None}
    try:
        data = json.loads(Path(config_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return out
    if not isinstance(data, dict):
        return out
    architectures = data.get("architectures")
    out["architectures"] = [str(a) for a in architectures] if isinstance(architectures, list) else None
    for key, source in (("model_type", "model_type"), ("transformers_version", "transformers_version")):
        value = data.get(source)
        out[key] = value if isinstance(value, str) else None
    dtype = data.get("torch_dtype", data.get("dtype"))
    out["dtype"] = dtype if isinstance(dtype, str) else None
    quantization = data.get("quantization_config")
    method = quantization.get("quant_method") if isinstance(quantization, dict) else None
    out["quantization_method"] = method if isinstance(method, str) else None
    return out


def model_fingerprint(model_dir: str | Path) -> dict[str, Any]:
    """The fingerprint of a local model directory (see the module docstring); ``FileNotFoundError`` if there is no such directory."""
    directory = Path(model_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"model directory {directory} does not exist")
    config: dict[str, str] = {}
    tokenizer: dict[str, str] = {}
    index: dict[str, str] = {}
    weights: dict[str, int] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        name = path.name
        if name.endswith(WEIGHT_SUFFIXES):
            weights[name] = path.stat().st_size
        elif name in TOKENIZER_FILES:
            tokenizer[name] = file_md5(path)
        elif name in CONFIG_FILES:
            config[name] = file_md5(path)
        elif name.endswith(INDEX_SUFFIX):
            index[name] = file_md5(path)
    core = {"config_files": config, "tokenizer_files": tokenizer, "index_files": index, "weight_files": weights}
    combined = hashlib.md5(json.dumps(core, sort_keys=True).encode("utf-8"), usedforsecurity=False).hexdigest()
    return {**core, "declared": declared_versions(directory / "config.json"), "md5": combined}


__all__ = ["CONFIG_FILES", "INDEX_SUFFIX", "TOKENIZER_FILES", "WEIGHT_SUFFIXES", "declared_versions", "model_fingerprint"]
