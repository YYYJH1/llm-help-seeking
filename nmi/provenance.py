"""Run provenance: hashes, git state without git, run directories, manifests, model.json.

The repository state is stamped at the START of a run (a commit that lands mid-run must not be
recorded as the code that produced the results), and SIGTERM is turned into aborted.json.
Git is never called: the commit id is read from .git/HEAD, the ref file and packed-refs, and
"uncommitted changes" is decided by comparing the working files with the blob ids in HEAD's
tree (loose objects only).

Run directory:

    results/<stage>/<YYYYMMDD_HHMMSS>_<tag>/manifest.json
                                           /llm/model.json      (write_model_json)
                                           /aborted.json        (on SIGTERM)

A tag that is already used in the same stage is an error: nothing is overwritten and
nothing is deleted. Time stamps in directory names are local time; every time field in
the manifest is ISO 8601 with the UTC offset.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import signal
import socket
import sys
import zlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from nmi.paths import (
    REPO_ROOT,
    PathPolicyError,
    assert_off_root_partition,
    results_root,
    run_dir_name,
    validate_name,
)

MANIFEST_SCHEMA = "nmi.manifest/1"

# ---------------------------------------------------------------------------------------
# hashing
# ---------------------------------------------------------------------------------------

_CHUNK = 1 << 20


def file_digest(path: str | Path, algorithm: str = "sha256") -> str:
    """Hex digest of the whole file (a cap would let the tail change unnoticed)."""
    h = hashlib.new(algorithm, usedforsecurity=False) if algorithm == "md5" else hashlib.new(algorithm)
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def file_sha256(path: str | Path) -> str:
    return file_digest(path, "sha256")


def file_md5(path: str | Path) -> str:
    return file_digest(path, "md5")


def bytes_md5(data: bytes) -> str:
    return hashlib.md5(data, usedforsecurity=False).hexdigest()


_NOTED: set[tuple[str, str]] = set()


def note_skipped(path: str | Path, what: str) -> None:
    """One line on stderr for a check whose input document is not part of this release; the check is skipped (printed once per path and check)."""
    key = (str(path), what)
    if key not in _NOTED:
        _NOTED.add(key)
        print(f"note: {path} not found; the {what} check is skipped", file=sys.stderr)


_SOURCE_EXCLUDE_DIRS = frozenset({"__pycache__", ".pytest_cache", ".ruff_cache", ".git"})
_SOURCE_EXCLUDE_SUFFIXES = (".pyc", ".pyo")


def _skip_dir(name: str) -> bool:
    return name in _SOURCE_EXCLUDE_DIRS or name.endswith(".egg-info")


def iter_source_files(root: str | Path) -> list[Path]:
    """Regular files under ``root``, sorted by relative POSIX path, caches excluded."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"source root {root} is not a directory")
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if not _skip_dir(d))
        for name in filenames:
            if name.endswith(_SOURCE_EXCLUDE_SUFFIXES):
                continue
            found.append(Path(dirpath) / name)
    return sorted(found, key=lambda p: p.relative_to(root).as_posix())


def source_hash(root: str | Path = REPO_ROOT / "nmi") -> str:
    """sha256 over (relative path, file sha256) pairs: the hash of the ``nmi/`` source tree."""
    root = Path(root)
    h = hashlib.sha256()
    for path in iter_source_files(root):
        rel = path.relative_to(root).as_posix()
        h.update(rel.encode("utf-8") + b"\0" + file_sha256(path).encode("ascii") + b"\n")
    return h.hexdigest()


class SourceChangedError(RuntimeError):
    """The hash of the source tree differs from the one recorded when the batch started."""


class SourceHashGuard:
    """Guard for batch runs: stop dispatching once ``nmi/`` changes.

    Create it when the batch starts and call ``check()`` before every dispatch."""

    def __init__(self, root: str | Path = REPO_ROOT / "nmi") -> None:
        self.root = Path(root)
        self.baseline = source_hash(self.root)

    def check(self) -> None:
        current = source_hash(self.root)
        if current != self.baseline:
            raise SourceChangedError(
                f"source hash of {self.root} changed during the run "
                f"(started {self.baseline[:12]}, now {current[:12]}); stop dispatching"
            )


# ---------------------------------------------------------------------------------------
# git state without calling git
# ---------------------------------------------------------------------------------------

_HEX_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")

# The files that determine a run's results. A scoped flag is useful; "anything anywhere in
# the repository" would be permanently true while other jobs write logs there.
DEFAULT_DIRTY_SCOPE = ("nmi", "scripts", "configs", "pyproject.toml")


class GitReadError(RuntimeError):
    """A git object is unreadable without calling git (packed or corrupt)."""


@dataclass(frozen=True)
class GitHead:
    commit: str | None
    ref: str | None
    detached: bool


@dataclass(frozen=True)
class GitState:
    head: GitHead
    dirty: bool | None
    dirty_scope: tuple[str, ...]
    dirty_paths: tuple[str, ...] = ()
    dirty_note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "head": self.head.commit,
            "ref": self.head.ref,
            "detached": self.head.detached,
            "dirty": self.dirty,
            "dirty_scope": list(self.dirty_scope),
            "dirty_paths": list(self.dirty_paths),
            "dirty_note": self.dirty_note,
        }


def resolve_git_dir(repo_root: str | Path) -> Path | None:
    """``<repo>/.git`` as a directory, or the target of a ``gitdir:`` file (worktrees)."""
    dot = Path(repo_root) / ".git"
    if dot.is_dir():
        return dot
    if dot.is_file():
        text = dot.read_text(encoding="utf-8").strip()
        if text.startswith("gitdir:"):
            target = Path(text[len("gitdir:"):].strip())
            return target if target.is_absolute() else (Path(repo_root) / target).resolve()
    return None


def _git_dirs(git_dir: Path) -> list[Path]:
    """The git dir plus the common dir of a linked worktree (refs and objects live there)."""
    dirs = [git_dir]
    marker = git_dir / "commondir"
    if marker.is_file():
        common = (git_dir / marker.read_text(encoding="utf-8").strip()).resolve()
        if common != git_dir:
            dirs.append(common)
    return dirs


def _packed_ref(directory: Path, ref: str) -> str | None:
    packed = directory / "packed-refs"
    if not packed.is_file():
        return None
    for line in packed.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "^")):
            continue
        sha, _, name = line.partition(" ")
        if name.strip() == ref and _HEX_RE.fullmatch(sha.lower()):
            return sha.lower()
    return None


def _resolve_ref(git_dirs: Sequence[Path], ref: str, depth: int = 0) -> str | None:
    if depth > 5:
        return None
    for directory in git_dirs:
        candidate = directory / ref
        if candidate.is_file():
            text = candidate.read_text(encoding="utf-8").strip()
            if _HEX_RE.fullmatch(text.lower()):
                return text.lower()
            if text.startswith("ref:"):
                return _resolve_ref(git_dirs, text[4:].strip(), depth + 1)
    for directory in git_dirs:
        found = _packed_ref(directory, ref)
        if found:
            return found
    return None


def read_git_head(repo_root: str | Path = REPO_ROOT) -> GitHead:
    """Commit id from .git/HEAD, the ref file and packed-refs; never runs git."""
    git_dir = resolve_git_dir(repo_root)
    if git_dir is None:
        return GitHead(None, None, False)
    head_file = git_dir / "HEAD"
    if not head_file.is_file():
        return GitHead(None, None, False)
    text = head_file.read_text(encoding="utf-8").strip()
    if text.startswith("ref:"):
        ref = text[4:].strip()
        return GitHead(_resolve_ref(_git_dirs(git_dir), ref), ref, False)
    if _HEX_RE.fullmatch(text.lower()):
        return GitHead(text.lower(), None, True)
    return GitHead(None, None, False)


def git_blob_id(data: bytes, algorithm: str = "sha1") -> str:
    """The object id git gives a file with this content."""
    return hashlib.new(algorithm, b"blob %d\0" % len(data) + data).hexdigest()


def _read_loose_object(git_dirs: Sequence[Path], sha: str) -> tuple[str, bytes]:
    for directory in git_dirs:
        path = directory / "objects" / sha[:2] / sha[2:]
        if path.is_file():
            try:
                raw = zlib.decompress(path.read_bytes())
            except zlib.error as error:
                raise GitReadError(f"object {sha} is corrupt: {error}") from error
            header, _, body = raw.partition(b"\0")
            kind, _, size = header.partition(b" ")
            if int(size) != len(body):
                raise GitReadError(f"object {sha} is corrupt")
            return kind.decode("ascii"), body
    raise GitReadError(f"object {sha} is not a loose object (packed objects are not read)")


def _parse_tree(body: bytes, hash_len: int) -> list[tuple[str, str, str]]:
    entries: list[tuple[str, str, str]] = []
    pos = 0
    while pos < len(body):
        space = body.index(b" ", pos)
        nul = body.index(b"\0", space)
        mode = body[pos:space].decode("ascii")
        name = body[space + 1:nul].decode("utf-8", errors="surrogateescape")
        sha = body[nul + 1:nul + 1 + hash_len].hex()
        entries.append((mode, name, sha))
        pos = nul + 1 + hash_len
    return entries


def _commit_tree(body: bytes) -> str:
    for line in body.split(b"\n"):
        if line.startswith(b"tree "):
            return line[5:].decode("ascii").strip()
        if not line:
            break
    raise GitReadError("commit object has no tree line")


def _tree_blobs(
    git_dirs: Sequence[Path], tree_sha: str, prefix: str, hash_len: int
) -> dict[str, str]:
    """Blob id for every file below a tree (submodule links are skipped)."""
    kind, body = _read_loose_object(git_dirs, tree_sha)
    if kind != "tree":
        raise GitReadError(f"object {tree_sha} is a {kind}, expected a tree")
    out: dict[str, str] = {}
    for mode, name, sha in _parse_tree(body, hash_len):
        rel = f"{prefix}{name}"
        if mode == "40000":
            out.update(_tree_blobs(git_dirs, sha, rel + "/", hash_len))
        elif mode != "160000":
            out[rel] = sha
    return out


def _expected_blobs(
    git_dirs: Sequence[Path], root_tree: str, scope_path: str, hash_len: int
) -> dict[str, str]:
    """Blob ids committed at HEAD for one scope entry (a file or a directory)."""
    parts = [p for p in scope_path.split("/") if p]
    tree_sha = root_tree
    prefix = ""
    for i, part in enumerate(parts):
        _, body = _read_loose_object(git_dirs, tree_sha)
        match = next((e for e in _parse_tree(body, hash_len) if e[1] == part), None)
        if match is None:
            return {}
        mode, _, sha = match
        last = i == len(parts) - 1
        if mode == "40000":
            tree_sha = sha
            prefix += part + "/"
            if last:
                return _tree_blobs(git_dirs, tree_sha, prefix, hash_len)
        elif last and mode != "160000":
            return {prefix + part: sha}
        else:
            return {}
    return _tree_blobs(git_dirs, root_tree, "", hash_len)


def _worktree_blobs(repo_root: Path, scope_path: str, algorithm: str) -> dict[str, str]:
    """Blob ids of the working files of one scope entry; caches and bytecode are skipped."""
    base = repo_root / scope_path
    out: dict[str, str] = {}

    def blob_of(path: Path) -> str:
        data = os.readlink(path).encode("utf-8") if path.is_symlink() else path.read_bytes()
        return git_blob_id(data, algorithm)

    if base.is_symlink() or base.is_file():
        out[scope_path] = blob_of(base)
    elif base.is_dir():
        for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
            dirnames[:] = sorted(d for d in dirnames if not _skip_dir(d))
            for name in filenames:
                if name.endswith(_SOURCE_EXCLUDE_SUFFIXES):
                    continue
                path = Path(dirpath) / name
                out[path.relative_to(repo_root).as_posix()] = blob_of(path)
    return out


def git_state(
    repo_root: str | Path = REPO_ROOT,
    scope: Sequence[str] = DEFAULT_DIRTY_SCOPE,
    *,
    max_paths: int = 20,
) -> GitState:
    """HEAD commit plus "uncommitted changes in ``scope``", all without running git.

    ``dirty`` is True when a file in ``scope`` was modified, added or deleted relative to
    HEAD's tree (staged or not). It is None when it cannot be decided: no commit yet, or
    HEAD's objects are packed (this reader handles loose objects only). The ignore rules
    are an approximation of this repository's .gitignore (``__pycache__``, ``*.pyc``,
    ``*.egg-info``, tool caches); anything else untracked inside ``scope`` counts as dirty.
    """
    repo_root = Path(repo_root)
    scope = tuple(scope)
    head = read_git_head(repo_root)
    if head.commit is None:
        return GitState(head, None, scope, dirty_note="no readable HEAD commit")
    git_dir = resolve_git_dir(repo_root)
    assert git_dir is not None
    dirs = _git_dirs(git_dir)
    algorithm, hash_len = ("sha1", 20) if len(head.commit) == 40 else ("sha256", 32)
    try:
        kind, body = _read_loose_object(dirs, head.commit)
        if kind != "commit":
            raise GitReadError(f"HEAD object is a {kind}")
        root_tree = _commit_tree(body)
        expected: dict[str, str] = {}
        for entry in scope:
            expected.update(_expected_blobs(dirs, root_tree, entry, hash_len))
    except GitReadError as error:
        return GitState(head, None, scope, dirty_note=str(error))
    actual: dict[str, str] = {}
    for entry in scope:
        actual.update(_worktree_blobs(repo_root, entry, algorithm))
    changed = sorted(
        {p for p in expected.keys() & actual.keys() if expected[p] != actual[p]}
        | (expected.keys() - actual.keys())
        | (actual.keys() - expected.keys())
    )
    return GitState(head, bool(changed), scope, tuple(changed[:max_paths]))


# ---------------------------------------------------------------------------------------
# environment, load, time
# ---------------------------------------------------------------------------------------

DEFAULT_PACKAGES = (
    "numpy", "scipy", "pandas", "pyarrow", "pyyaml", "torch", "transformers", "tokenizers",
    "safetensors", "vllm", "httpx", "aiohttp", "datasets", "statsmodels", "pytest",
)


def environment_versions(packages: Iterable[str] = DEFAULT_PACKAGES) -> dict[str, Any]:
    """Interpreter, platform and installed versions (metadata only, nothing is imported)."""
    versions: dict[str, str | None] = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": versions,
    }


def read_loadavg(path: str | Path = "/proc/loadavg") -> dict[str, Any] | None:
    try:
        raw = Path(path).read_text(encoding="ascii").strip()
        parts = raw.split()
        return {
            "raw": raw,
            "load1": float(parts[0]),
            "load5": float(parts[1]),
            "load15": float(parts[2]),
        }
    except (OSError, ValueError, IndexError):
        return None


def now_local() -> datetime:
    """Timezone-aware local time."""
    return datetime.now().astimezone()


def iso(moment: datetime) -> str:
    """ISO 8601 with the UTC offset (a naive value is taken as local time)."""
    if moment.tzinfo is None:
        moment = moment.astimezone()
    return moment.isoformat(timespec="seconds")


# ---------------------------------------------------------------------------------------
# run directory and manifest
# ---------------------------------------------------------------------------------------

class LabelExistsError(FileExistsError):
    """The tag is already used in this stage; pick another, nothing is overwritten."""


def _atomic_write_json(path: Path, obj: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def create_run_dir(
    stage: str,
    tag: str,
    *,
    root: str | Path | None = None,
    when: datetime | None = None,
) -> Path:
    """Create ``<root>/<stage>/<YYYYMMDD_HHMMSS>_<tag>``; an existing tag is an error."""
    validate_name("stage", stage)
    validate_name("tag", tag)
    stage_path = results_root(root) / stage
    assert_off_root_partition(stage_path)
    stage_path.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}"
    for existing in sorted(stage_path.iterdir()):
        match = re.fullmatch(r"\d{8}_\d{6}_(.+)", existing.name)
        if match and match.group(1) == tag:
            raise LabelExistsError(
                f"tag {tag!r} already exists in stage {stage!r} as {existing.name}; "
                "choose another tag (nothing is overwritten or deleted)"
            )
    when = when or now_local()
    run_path = stage_path / run_dir_name(tag, when)
    try:
        run_path.mkdir()
    except FileExistsError as error:
        raise LabelExistsError(f"run directory {run_path} already exists") from error
    assert run_path.name.endswith(suffix)
    return run_path


@dataclass
class RunRecord:
    """An open run: the directory plus the manifest that is rewritten at the end."""

    run_dir: Path
    manifest: dict[str, Any]
    _start: datetime = field(repr=False, default_factory=now_local)
    _previous_sigterm: Any = field(repr=False, default=None)

    @property
    def run_id(self) -> str:
        return self.manifest["run_id"]

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    def write_manifest(self) -> None:
        _atomic_write_json(self.manifest_path, self.manifest)

    def finish(
        self,
        status: str = "ok",
        *,
        exit_code: int | None = None,
        error: str | None = None,
        now: datetime | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        """Close the run: end time, final load, exit status; the manifest is rewritten."""
        if status not in {"ok", "error", "aborted"}:
            raise ValueError(f"exit status must be ok, error or aborted, got {status!r}")
        end = now or now_local()
        if end.tzinfo is None:
            end = end.astimezone()
        self.manifest["end_time"] = iso(end)
        self.manifest["duration_s"] = round((end - self._start).total_seconds(), 3)
        self.manifest["loadavg_end"] = read_loadavg()
        self.manifest["exit_status"] = status
        self.manifest["exit_code"] = exit_code
        self.manifest["error"] = error
        if extra:
            self.manifest.update(extra)
        self.write_manifest()

    def abort(
        self,
        reason: str,
        *,
        signum: int | None = None,
        now: datetime | None = None,
    ) -> None:
        """Write aborted.json and mark the manifest aborted (idempotent)."""
        if self.manifest.get("exit_status") == "aborted":
            return
        moment = now or now_local()
        _atomic_write_json(
            self.run_dir / "aborted.json",
            {
                "aborted": True,
                "reason": reason,
                "signal": signum,
                "time": iso(moment),
                "run_id": self.run_id,
                "stage": self.manifest["stage"],
                "tag": self.manifest["tag"],
                "git_head": self.manifest["git"]["head"],
            },
        )
        self.finish("aborted", exit_code=(128 + signum) if signum else None, error=reason, now=moment)

    def install_sigterm_handler(self) -> Callable[[], None]:
        """Turn SIGTERM into aborted.json plus SystemExit. Returns a function that restores
        the previous handler. Must be called from the main thread."""

        def handler(signum: int, frame: Any) -> None:
            self.abort(f"terminated by signal {signum}", signum=signum)
            raise SystemExit(128 + signum)

        self._previous_sigterm = signal.signal(signal.SIGTERM, handler)

        def restore() -> None:
            signal.signal(signal.SIGTERM, self._previous_sigterm)

        return restore

    def __enter__(self) -> "RunRecord":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self.manifest.get("exit_status") != "running":
            return False
        if exc_type is None or (issubclass(exc_type, SystemExit) and exc.code in (None, 0)):
            self.finish("ok", exit_code=0)
        elif issubclass(exc_type, KeyboardInterrupt):
            self.abort("interrupted by KeyboardInterrupt")
        else:
            code = exc.code if issubclass(exc_type, SystemExit) and isinstance(exc.code, int) else 1
            self.finish("error", exit_code=code, error=f"{exc_type.__name__}: {exc}")
        return False


def begin_run(
    *,
    stage: str,
    tag: str,
    protocol_version: str,
    config_revision: str | None = None,
    results: str | Path | None = None,
    argv: Sequence[str] | None = None,
    input_files: Mapping[str, str | Path] | None = None,
    input_md5: Mapping[str, str] | None = None,
    seed_keys: Iterable[Sequence[int]] = (),
    parents: Sequence[str] = (),
    repo_root: str | Path = REPO_ROOT,
    source_root: str | Path | None = None,
    dirty_scope: Sequence[str] = DEFAULT_DIRTY_SCOPE,
    now: datetime | None = None,
) -> RunRecord:
    """Create the run directory and write the first manifest (status ``running``).

    The repository state is read now, at the start: a commit that lands during a run must not be
    recorded as the code that produced it.

    ``input_files`` maps names to files whose md5 is recorded (configuration and protocol files);
    ``input_md5`` carries md5 values that are already known (the split)."""
    repo_root = Path(repo_root)
    start = now or now_local()
    if start.tzinfo is None:
        start = start.astimezone()
    run_path = create_run_dir(stage, tag, root=results, when=start)
    argv = list(sys.argv if argv is None else argv)
    md5s: dict[str, str] = {name: file_md5(path) for name, path in (input_files or {}).items()}
    md5s.update(input_md5 or {})
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "run_id": f"{validate_name('stage', stage)}/{run_path.name}",
        "stage": stage,
        "tag": tag,
        "parent_run_ids": list(parents),
        "start_time": iso(start),
        "end_time": None,
        "duration_s": None,
        "exit_status": "running",
        "exit_code": None,
        "error": None,
        "command_line": argv,
        "command": " ".join(argv),
        "hostname": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "loadavg_start": read_loadavg(),
        "loadavg_end": None,
        "git": git_state(repo_root, dirty_scope).to_dict(),
        "nmi_source_sha256": source_hash(Path(source_root) if source_root else repo_root / "nmi"),
        "protocol_version": protocol_version,
        "config_revision": config_revision,
        "input_md5": md5s,
        "seed_keys": [list(map(int, key)) for key in seed_keys],
        "environment": environment_versions(),
    }
    record = RunRecord(run_dir=run_path, manifest=manifest, _start=start)
    record.write_manifest()
    return record


# ---------------------------------------------------------------------------------------
# llm/model.json
# ---------------------------------------------------------------------------------------

_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf")
_TOKENIZER_NAMES = (
    "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "tokenizer.model",
    "special_tokens_map.json", "added_tokens.json", "chat_template.jinja", "chat_template.json",
)
_CONFIG_NAMES = ("config.json", "generation_config.json")


def _hash_with_cache(path: Path, cache: dict[str, Any] | None) -> str:
    if cache is None:
        return file_sha256(path)
    stat = path.stat()
    entry = cache.get(str(path))
    if entry and entry["size"] == stat.st_size and entry["mtime_ns"] == stat.st_mtime_ns:
        return entry["sha256"]
    digest = file_sha256(path)
    cache[str(path)] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest}
    return digest


def hash_model_files(
    model_dir: str | Path, *, cache: dict[str, Any] | None = None
) -> dict[str, dict[str, str]]:
    """sha256 of weight shards, tokenizer files and config files of a local model directory.

    ``cache`` (a dict the caller loads and saves) avoids rehashing multi-GB shards when
    path, size and mtime are unchanged.
    """
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model directory {model_dir} does not exist")
    weights: dict[str, str] = {}
    tokenizer: dict[str, str] = {}
    configs: dict[str, str] = {}
    for path in sorted(model_dir.iterdir()):
        if not path.is_file():
            continue
        if path.name.endswith(_WEIGHT_SUFFIXES):
            weights[path.name] = _hash_with_cache(path, cache)
        elif path.name in _TOKENIZER_NAMES:
            tokenizer[path.name] = _hash_with_cache(path, cache)
        elif path.name in _CONFIG_NAMES or path.name.endswith(".index.json"):
            configs[path.name] = _hash_with_cache(path, cache)
    return {"weight_shards": weights, "tokenizer_files": tokenizer, "config_files": configs}


def write_model_json(
    run_dir: str | Path,
    *,
    model_id: str,
    model_dir: str | Path,
    dtype: str,
    vllm_version: str | None = None,
    serve_params: Mapping[str, Any] | None = None,
    chat_template_kwargs: Mapping[str, Any] | None = None,
    thinking: bool | None = None,
    hash_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``<run_dir>/llm/model.json`` and return its content."""
    kwargs = dict(chat_template_kwargs or {})
    if thinking is None:
        thinking = kwargs.get("enable_thinking")
    if vllm_version is None:
        try:
            vllm_version = metadata.version("vllm")
        except metadata.PackageNotFoundError:
            vllm_version = None
    content: dict[str, Any] = {
        "model_id": model_id,
        "model_dir": str(model_dir),
        **hash_model_files(model_dir, cache=hash_cache),
        "dtype": dtype,
        "vllm_version": vllm_version,
        "serve_params": dict(serve_params or {}),
        "chat_template_kwargs": kwargs,
        "thinking": thinking,
    }
    out = Path(run_dir) / "llm"
    out.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(out / "model.json", content)
    return content


__all__ = [
    "DEFAULT_DIRTY_SCOPE",
    "GitHead",
    "GitReadError",
    "GitState",
    "LabelExistsError",
    "PathPolicyError",
    "RunRecord",
    "SourceChangedError",
    "SourceHashGuard",
    "begin_run",
    "bytes_md5",
    "create_run_dir",
    "environment_versions",
    "file_digest",
    "file_md5",
    "file_sha256",
    "git_blob_id",
    "git_state",
    "hash_model_files",
    "iso",
    "iter_source_files",
    "now_local",
    "read_git_head",
    "read_loadavg",
    "resolve_git_dir",
    "source_hash",
    "write_model_json",
]
