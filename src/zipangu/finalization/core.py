"""Shared safety, hashing, streaming, and runtime-state primitives."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import yaml


DEFAULT_MODEL_ID = "ZIPANGU-K-I-4B"
DEFAULT_BASE_MODEL_ID = "empero-ai/Qwen3.8-4B-Distill"
# Backward-compatible alias for legacy imports; model-aware stages resolve the canonical id.
MODEL_ID = DEFAULT_BASE_MODEL_ID
DEFAULT_PATHS = {
    "dataset_root": Path("E:/zipangu-datesets"),
    "eval_root": Path("E:/zipangu-eval"),
    "cache_root": Path("E:/zipangu-cache"),
    "model_root": Path("E:/zipangu-models"),
    "temp_root": Path("E:/zipangu-temp"),
}
FIRST_WAVE_DATASETS = (
    "math_japanese_8k",
    "ace_reason_math_japanese",
    "magpie_sft_v1",
    "extraction_wiki_ja",
    "nemotron_sft_multilingual_v2",
    "fable_5_premium",
)
QUALITY_THRESHOLDS = (95, 90, 85, 80, 75)
MIN_RETENTION = {
    "math_japanese_8k": 0.25,
    "ace_reason_math_japanese": 0.10,
    "magpie_sft_v1": 0.25,
    "extraction_wiki_ja": 0.25,
    "nemotron_sft_multilingual_v2": 0.20,
    "fable_5_premium": 0.10,
}
JP_HEAVY_BUCKETS = {
    "general_japanese": 0.30,
    "instruction_extraction": 0.15,
    "japanese_math_reasoning": 0.20,
    "japanese_stem_code": 0.10,
    "frontier_reasoning": 0.15,
    "agent_code_retention": 0.10,
}


class FinalizationBlocked(RuntimeError):
    """A stage cannot safely proceed with the currently available inputs."""


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, set):
        return sorted(value)
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=json_default)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def config_hash(config: Mapping[str, Any]) -> str:
    return sha256_text(canonical_json(config))


def atomic_write_text(path: str | Path, text: str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
    try:
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def atomic_write_json(path: str | Path, value: Any) -> Path:
    return atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, default=json_default) + "\n")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, Mapping):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return dict(value)


def free_bytes(path: str | Path) -> int:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return int(shutil.disk_usage(target).free)


def storage_preflight(
    path: str | Path,
    *,
    estimated_output_bytes: int = 0,
    reserve_bytes: int = 0,
) -> dict[str, Any]:
    available = free_bytes(path)
    required = max(0, int(estimated_output_bytes)) + max(0, int(reserve_bytes))
    return {
        "path": str(Path(path)),
        "free_bytes": available,
        "estimated_output_bytes": int(estimated_output_bytes),
        "reserve_bytes": int(reserve_bytes),
        "required_bytes": required,
        "allowed": available > required,
    }


def _path_fingerprint(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    try:
        stat = path.stat()
        relative = str(path)
        if root:
            try:
                relative = path.relative_to(root).as_posix()
            except ValueError:
                relative = str(path)
        return {"path": relative, "bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}
    except OSError as exc:
        return {"path": str(path), "error": type(exc).__name__}


def input_fingerprint(paths: Iterable[str | Path], *, root: str | Path | None = None) -> str:
    """Hash a deterministic file inventory without reading huge raw files."""

    root_path = Path(root).resolve() if root else None
    entries: list[dict[str, Any]] = []
    for raw in paths:
        path = Path(raw)
        if path.is_file():
            entries.append(_path_fingerprint(path, root=root_path))
        elif path.is_dir():
            for child in sorted((item for item in path.rglob("*") if item.is_file()), key=lambda item: str(item).casefold()):
                entries.append(_path_fingerprint(child, root=root_path))
        else:
            entries.append({"path": str(path), "missing": True})
    return sha256_text(canonical_json(entries))


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    import gzip

    file_path = Path(path)
    opener = gzip.open if file_path.name.casefold().endswith(".gz") else open
    with opener(file_path, "rt", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                yield {"_parse_error": f"line:{line_number}"}
                continue
            yield item if isinstance(item, Mapping) else {"_invalid_row": item}


class RuntimeStore:
    """Small JSON runtime store with stage-level checkpoints.

    Runtime state belongs outside the repository by default.  Writes are
    atomic, and a failed stage is recorded without preventing other stages
    from running.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.state: dict[str, Any] = self._load()
        self._recover_stale_runs()

    def _load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"schema_version": 1, "stages": {}, "artifacts": [], "training_allowed": False}
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return {"schema_version": 1, "stages": {}, "artifacts": [], "training_allowed": False}
        return dict(value) if isinstance(value, Mapping) else {"schema_version": 1, "stages": {}}

    def save(self) -> None:
        atomic_write_json(self.path, self.state)

    def _recover_stale_runs(self) -> None:
        changed = False
        stages = self.state.get("stages", {})
        if isinstance(stages, Mapping):
            for value in stages.values():
                if not isinstance(value, dict) or value.get("status") != "RUNNING":
                    continue
                value["status"] = "PARTIAL"
                reasons = value.setdefault("blocked_reasons", [])
                if "previous_process_interrupted_or_stale" not in reasons:
                    reasons.append("previous_process_interrupted_or_stale")
                changed = True
        if changed:
            self.save()

    def stage(self, name: str) -> dict[str, Any]:
        stages = self.state.setdefault("stages", {})
        value = stages.setdefault(name, {"status": "PENDING", "checkpoints": [], "artifacts": []})
        return value

    def can_resume(self, name: str, fingerprint: str, cfg_hash: str) -> bool:
        current = self.stage(name)
        return (
            (
                current.get("status") == "PASS"
                or (
                    current.get("status") == "BLOCKED"
                    and bool(current.get("artifacts"))
                    and not current.get("errors")
                )
            )
            and current.get("input_fingerprint") == fingerprint
            and current.get("config_hash") == cfg_hash
        )

    def start(self, name: str, *, fingerprint: str, cfg_hash: str, resume: bool = False) -> bool:
        previous = self.stage(name)
        if resume and self.can_resume(name, fingerprint, cfg_hash):
            return False
        resume_from = []
        if (
            resume
            and previous.get("input_fingerprint") == fingerprint
            and previous.get("config_hash") == cfg_hash
            and previous.get("status") in {"PARTIAL", "RUNNING", "ERROR"}
        ):
            resume_from = list(previous.get("resume_from", [])) + list(
                previous.get("checkpoints", [])
            )
        self.state.setdefault("stages", {})[name] = {
            "status": "RUNNING",
            "input_fingerprint": fingerprint,
            "config_hash": cfg_hash,
            "checkpoints": [],
            "resume_from": resume_from,
            "artifacts": [],
            "errors": [],
            "blocked_reasons": [],
        }
        self.save()
        return True

    def checkpoint(self, name: str, checkpoint: Mapping[str, Any]) -> None:
        self.stage(name).setdefault("checkpoints", []).append(dict(checkpoint))
        self.save()

    def finish(
        self,
        name: str,
        status: str,
        *,
        artifacts: Iterable[str | Path] = (),
        errors: Iterable[str] = (),
        blocked_reasons: Iterable[str] = (),
        details: Mapping[str, Any] | None = None,
    ) -> None:
        item = self.stage(name)
        item["status"] = status
        item["artifacts"] = [str(Path(value)) for value in artifacts]
        item["errors"] = list(errors)
        item["blocked_reasons"] = list(blocked_reasons)
        if details:
            detail_values = dict(details)
            item["details"] = detail_values
            reserved = {
                "status",
                "artifacts",
                "errors",
                "blocked_reasons",
                "checkpoints",
                "resume_from",
                "input_fingerprint",
                "config_hash",
                "training_allowed",
            }
            item.update(
                {
                    key: value
                    for key, value in detail_values.items()
                    if key not in reserved
                }
            )
        self.state["training_allowed"] = False
        self.save()

    def add_artifacts(self, paths: Iterable[str | Path]) -> None:
        values = self.state.setdefault("artifacts", [])
        for path in paths:
            value = str(Path(path))
            if value not in values:
                values.append(value)
        self.save()

    def statuses(self) -> dict[str, str]:
        return {name: str(value.get("status", "PENDING")) for name, value in self.state.get("stages", {}).items()}
