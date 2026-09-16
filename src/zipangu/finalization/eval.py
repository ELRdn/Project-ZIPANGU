"""Official evaluation-corpus acquisition and metadata-only fingerprinting."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
import json
from pathlib import Path
import re
import urllib.request
import zipfile
from typing import Any

from .contamination import normalize_text
from .core import atomic_write_json, sha256_text
from .schema_adapters import project_eval_record


class EvalUnavailable(RuntimeError):
    """An optional reader or an official evaluation source is unavailable."""


class EvalAccessDenied(EvalUnavailable):
    """The evaluation source requires access that was not granted."""


EVAL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "id": "japanese_mt_bench",
        "name": "Japanese MT-Bench (llm-jp-judge embedded)",
        "version": "v2.0.0",
        "split": "embedded",
        "source": "llm-jp/llm-jp-judge",
        "source_revision": "v2.0.0",
        "license_raw": "see upstream llm-jp-judge release",
        "kind": "github",
        "required": True,
        "relative_path": "sources/llm-jp-judge-2.0.0",
    },
    {
        "id": "llm_jp_instruction_eval",
        "name": "llm-jp-instructions",
        "version": "v1.0",
        "split": "test",
        "source": "llm-jp/llm-jp-instructions",
        "source_revision": "v1.0",
        "license_raw": "CC BY 4.0",
        "kind": "hf",
        "required": True,
        "relative_path": "sources/llm-jp-instructions-v1.0-test",
    },
    {
        "id": "answer_carefully",
        "name": "AnswerCarefully",
        "version": "v2.0",
        "split": "test",
        "source": "llm-jp/AnswerCarefully",
        "source_revision": "v2.0",
        "license_raw": "upstream metadata required",
        "kind": "hf_gated",
        "required": True,
        "relative_path": "sources/AnswerCarefully-v2.0-test",
    },
    {
        "id": "zipangu_private_holdout",
        "name": "ZIPANGU private holdout",
        "version": "local",
        "split": "holdout",
        "source": "local-only",
        "source_revision": "local-unresolved",
        "license_raw": "private-local",
        "kind": "local_optional",
        "required": False,
        "relative_path": "private/zipangu_private_holdout",
    },
    {
        "id": "english_general_regression",
        "name": "English general regression",
        "version": "local",
        "split": "holdout",
        "source": "local-only",
        "source_revision": "local-unresolved",
        "license_raw": "unknown",
        "kind": "local_optional",
        "required": False,
        "relative_path": "secondary/english_general_regression",
    },
    {
        "id": "japanese_math_holdout",
        "name": "ZIPANGU Japanese math holdout",
        "version": "local",
        "split": "holdout",
        "source": "local-only",
        "source_revision": "local-unresolved",
        "license_raw": "unknown",
        "kind": "local_optional",
        "required": False,
        "relative_path": "secondary/japanese_math_holdout",
    },
    {
        "id": "japanese_writing_holdout",
        "name": "ZIPANGU Japanese writing holdout",
        "version": "local",
        "split": "holdout",
        "source": "local-only",
        "source_revision": "local-unresolved",
        "license_raw": "unknown",
        "kind": "local_optional",
        "required": False,
        "relative_path": "secondary/japanese_writing_holdout",
    },
    {
        "id": "code_reasoning_sanity",
        "name": "ZIPANGU code reasoning sanity",
        "version": "local",
        "split": "holdout",
        "source": "local-only",
        "source_revision": "local-unresolved",
        "license_raw": "unknown",
        "kind": "local_optional",
        "required": False,
        "relative_path": "secondary/code_reasoning_sanity",
    },
)


def _safe_file(path: Path) -> bool:
    name = path.name.casefold()
    return path.is_file() and not any(
        name.endswith(suffix) for suffix in (".lock", ".metadata", ".pyc", ".bin", ".safetensors")
    ) and path.suffix.casefold() in {".json", ".jsonl", ".gz", ".parquet"}


def _iter_json_file(path: Path) -> Iterator[dict[str, Any]]:
    import gzip

    opener = gzip.open if path.name.casefold().endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        try:
            value = json.load(handle) if path.suffix.casefold() == ".json" else None
        except json.JSONDecodeError as exc:
            yield {"_parse_error": exc.msg}
            return
        if value is None:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    yield {"_parse_error": "invalid_jsonl"}
                    continue
                yield item if isinstance(item, Mapping) else {"_invalid_row": item}
            return
        values: Any = value
        if isinstance(value, Mapping):
            for key in ("data", "questions", "examples", "rows", "test", "items"):
                if isinstance(value.get(key), list):
                    values = value[key]
                    break
        if isinstance(values, list):
            for item in values:
                yield item if isinstance(item, Mapping) else {"_invalid_row": item}
        elif isinstance(values, Mapping):
            yield dict(values)


def _iter_parquet(path: Path) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise EvalUnavailable("Parquet eval source requires pyarrow") from exc
    parquet_file = parquet.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=512):
        for item in batch.to_pylist():
            yield item if isinstance(item, Mapping) else {"_invalid_row": item}


def iter_eval_rows(path: str | Path, *, split_hint: str | None = None) -> Iterator[tuple[str, int, dict[str, Any]]]:
    """Yield rows with source identity while keeping the row content ephemeral."""

    root = Path(path)
    files = [root] if root.is_file() else [item for item in root.rglob("*") if _safe_file(item)]
    for file in sorted(files, key=lambda item: str(item).casefold()):
        relative = file.name if root.is_file() else file.relative_to(root).as_posix()
        split_context = relative.casefold()
        root_is_test_only = root.is_dir() and root.name.casefold() in {"test", "testing"}
        if (
            split_hint
            and split_hint.casefold() == "test"
            and "test" not in split_context
            and not root_is_test_only
        ):
            # The contract is explicitly the test split; dev/train files are
            # never silently mixed into the evaluation fingerprint.
            continue
        rows = _iter_parquet(file) if file.suffix.casefold() == ".parquet" else _iter_json_file(file)
        for row_index, row in enumerate(rows, start=1):
            yield relative, row_index, row


def _metadata_license(path: Path) -> str:
    readme = path / "README.md" if path.is_dir() else None
    if readme is None or not readme.is_file():
        return "unknown"
    try:
        text = readme.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "unknown"
    match = re.search(r"(?im)^license\s*:\s*([^\n]+)", text)
    return match.group(1).strip() if match else "unknown"


def _row_identity(relative: str, row_index: int) -> str:
    return f"{relative}#{row_index}"


def fingerprint_eval_source(
    spec: Mapping[str, Any],
    source_path: str | Path,
    fingerprint_root: str | Path,
) -> dict[str, Any]:
    """Fingerprint an eval source; output contains hashes and metadata only."""

    path = Path(source_path)
    destination = Path(fingerprint_root) / f"{spec['id']}.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.unlink(missing_ok=True)
    row_count = 0
    content_digests: list[str] = []
    try:
        with temporary.open("w", encoding="utf-8") as output:
            for relative, row_index, row in iter_eval_rows(
                path, split_hint=str(spec.get("split", ""))
            ):
                identity = _row_identity(relative, row_index)
                projection = project_eval_record(str(spec["id"]), row)
                content = normalize_text(projection.content)
                prompt = normalize_text(projection.prompt)
                if not content:
                    raise EvalUnavailable(
                        f"empty projection is forbidden: {spec['id']}:{identity} "
                        f"adapter={projection.adapter_id}"
                    )
                content_digest = sha256_text(content)
                prompt_digest = sha256_text(prompt) if prompt else ""
                content_digests.append(content_digest)
                output.write(
                    json.dumps(
                        {
                            "source_row_id": identity,
                            "content_hash": content_digest,
                            "prompt_hash": prompt_digest,
                            "normalized_length": len(content),
                            "schema_adapter": projection.adapter_id,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                row_count += 1
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    aggregate = sha256_text("\n".join(content_digests)) if content_digests else None
    return {
        "id": spec["id"],
        "name": spec["name"],
        "version": spec["version"],
        "split": spec["split"],
        "row_count": row_count,
        "source": spec["source"],
        "source_revision": spec["source_revision"],
        "local_path": str(path),
        "license_raw": _metadata_license(path) if spec.get("license_raw", "").startswith("see") else spec.get("license_raw", "unknown"),
        "content_hash": aggregate,
        "availability": "available" if row_count else "unavailable_empty",
        "contamination_scan_ready": bool(row_count),
        "training_forbidden": True,
        "required": bool(spec.get("required", False)),
        "fingerprint_path": str(destination),
    }


def _download_github_release(destination: Path, *, revision: str) -> None:
    if any(_safe_file(item) for item in destination.rglob("*")):
        return
    destination.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/llm-jp/llm-jp-judge/archive/refs/tags/{revision}.zip"
    archive = destination.parent / f".{destination.name}.zip"
    try:
        urllib.request.urlretrieve(url, archive)
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(destination.parent)
        extracted = destination.parent / f"llm-jp-judge-{revision}"
        if not extracted.is_dir() and revision.startswith("v"):
            extracted = destination.parent / f"llm-jp-judge-{revision[1:]}"
        if extracted.is_dir() and extracted != destination:
            for child in extracted.iterdir():
                target = destination / child.name
                if target.exists():
                    continue
                child.replace(target)
            extracted.rmdir()
    finally:
        archive.unlink(missing_ok=True)


def _resolve_hf_revision(repo_id: str, *, gated: bool = False) -> str:
    try:
        from huggingface_hub import HfApi

        info = HfApi().dataset_info(repo_id, revision="main")
        revision = getattr(info, "sha", None)
    except Exception as exc:
        if gated or any(token in str(exc).casefold() for token in ("gated", "401", "403", "access")):
            raise EvalAccessDenied(f"access to {repo_id} was not granted") from exc
        raise EvalUnavailable(f"could not resolve a full revision for {repo_id}") from exc
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise EvalUnavailable(f"{repo_id} did not return a full 40-character revision")
    return revision

def _download_hf_snapshot(destination: Path, *, repo_id: str, revision: str, gated: bool = False) -> None:
    if any(_safe_file(item) for item in destination.rglob("*")):
        return
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise EvalUnavailable("huggingface_hub is required to acquire official eval sources") from exc
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=str(destination),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
    except Exception as exc:
        message = str(exc).casefold()
        if gated or any(token in message for token in ("gated", "401", "403", "access")):
            raise EvalAccessDenied(f"access to {repo_id} was not granted") from exc
        raise EvalUnavailable(f"could not acquire {repo_id}: {type(exc).__name__}") from exc


def acquire_eval_sources(eval_root: str | Path) -> list[dict[str, Any]]:
    """Acquire only official/free sources and build external fingerprints."""

    root = Path(eval_root)
    root.mkdir(parents=True, exist_ok=True)
    fingerprint_root = root / "_fingerprints"
    entries: list[dict[str, Any]] = []
    for spec in EVAL_SPECS:
        source_spec = dict(spec)
        destination = root / str(spec["relative_path"])
        try:
            kind = spec["kind"]
            if kind == "github":
                _download_github_release(destination, revision=str(spec["source_revision"]))
            elif kind == "hf":
                resolved = _resolve_hf_revision(str(spec["source"]))
                source_spec["source_revision"] = resolved
                _download_hf_snapshot(destination, repo_id=str(spec["source"]), revision=resolved)
            elif kind == "hf_gated":
                resolved = _resolve_hf_revision(str(spec["source"]), gated=True)
                source_spec["source_revision"] = resolved
                _download_hf_snapshot(destination, repo_id=str(spec["source"]), revision=resolved, gated=True)
            elif kind == "local_optional":
                if not destination.is_dir():
                    entries.append(
                        {
                            "id": spec["id"],
                            "name": spec["name"],
                            "version": spec["version"],
                            "split": spec["split"],
                            "row_count": 0,
                            "source": spec["source"],
                            "source_revision": spec["source_revision"],
                            "local_path": str(destination),
                            "license_raw": spec["license_raw"],
                            "content_hash": None,
                            "availability": "unavailable_not_provided",
                            "contamination_scan_ready": False,
                            "training_forbidden": True,
                            "required": False,
                            "fingerprint_path": None,
                        }
                    )
                    continue
            entry = fingerprint_eval_source(source_spec, destination, fingerprint_root)
        except EvalAccessDenied as exc:
            entry = {
                "id": spec["id"],
                "name": spec["name"],
                "version": spec["version"],
                "split": spec["split"],
                "row_count": 0,
                "source": spec["source"],
                "source_revision": spec["source_revision"],
                "local_path": str(destination),
                "license_raw": spec["license_raw"],
                "content_hash": None,
                "availability": "blocked_access_denied",
                "contamination_scan_ready": False,
                "training_forbidden": True,
                "required": bool(spec.get("required", False)),
                "fingerprint_path": None,
                "blocked_reason": str(exc),
            }
        except (EvalUnavailable, OSError, zipfile.BadZipFile) as exc:
            entry = {
                "id": spec["id"],
                "name": spec["name"],
                "version": spec["version"],
                "split": spec["split"],
                "row_count": 0,
                "source": spec["source"],
                "source_revision": spec["source_revision"],
                "local_path": str(destination),
                "license_raw": spec["license_raw"],
                "content_hash": None,
                "availability": "unavailable",
                "contamination_scan_ready": False,
                "training_forbidden": True,
                "required": bool(spec.get("required", False)),
                "fingerprint_path": None,
                "blocked_reason": f"{type(exc).__name__}",
            }
        entries.append(entry)
    return entries


def write_eval_registry(repo_root: str | Path, entries: Iterable[Mapping[str, Any]]) -> Path:
    path = Path(repo_root) / "data" / "manifests" / "eval_registry.json"
    metadata = []
    for item in entries:
        allowed = {
            "id",
            "name",
            "version",
            "split",
            "row_count",
            "source",
            "source_revision",
            "local_path",
            "license_raw",
            "content_hash",
            "availability",
            "contamination_scan_ready",
            "training_forbidden",
            "required",
            "fingerprint_path",
            "blocked_reason",
        }
        metadata.append({key: item.get(key) for key in sorted(allowed) if key in item})
    payload = {"schema_version": 1, "training_forbidden": True, "evaluations": metadata}
    return atomic_write_json(path, payload)


def load_eval_registry(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError("eval registry must be an object")
    return dict(value)
