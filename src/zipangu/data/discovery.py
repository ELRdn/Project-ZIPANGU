"""Dataset location discovery and lightweight inventory primitives.

This module deliberately knows nothing about model training.  It only finds
local sources and describes them without editing the source tree.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import yaml


DATASET_ROOT_ENV = "DATASET_ROOT"
DEFAULT_POLICY_PATH = Path("configs/datasets/source_policies.yaml")
SUPPORTED_DATA_SUFFIXES = frozenset({".jsonl", ".jsonl.gz", ".parquet"})
IGNORED_DIRECTORY_NAMES = frozenset({".cache", ".git", "__pycache__"})
IGNORED_FILE_SUFFIXES = frozenset({".lock", ".metadata", ".pyc"})
METADATA_JSON_NAMES = frozenset(
    {
        "manifest.json",
        "dataset_infos.json",
        "dataset_config.json",
        "curriculum_stages.json",
    }
)


@dataclass(frozen=True)
class SourcePolicy:
    dataset_id: str
    repo: str
    local_dir: str
    category: str
    role: str
    recommended_tier: str
    recommended_usage: str
    recommended_max_weight: float
    license_expected: str
    vendor_specific: bool
    audit_mode: str = "full"
    stats_sample_rows: int = 0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], index: int) -> "SourcePolicy":
        required = ("id", "repo", "local_dir", "category", "role")
        missing = [field for field in required if not isinstance(raw.get(field), str) or not raw[field].strip()]
        if missing:
            raise ValueError(f"source policy {index} is missing required fields: {', '.join(missing)}")
        max_weight = raw.get("recommended_max_weight", 0.0)
        if isinstance(max_weight, bool) or not isinstance(max_weight, (int, float)):
            raise ValueError(f"source policy {raw['id']} recommended_max_weight must be numeric")
        audit_mode = raw.get("audit_mode", "full")
        if audit_mode not in {"full", "stats_only"}:
            raise ValueError(f"source policy {raw['id']} has unsupported audit_mode: {audit_mode}")
        sample_rows = raw.get("stats_sample_rows", 0)
        if isinstance(sample_rows, bool) or not isinstance(sample_rows, int) or sample_rows < 0:
            raise ValueError(f"source policy {raw['id']} stats_sample_rows must be a non-negative integer")
        return cls(
            dataset_id=str(raw["id"]),
            repo=str(raw["repo"]),
            local_dir=str(raw["local_dir"]),
            category=str(raw["category"]),
            role=str(raw["role"]),
            recommended_tier=str(raw.get("recommended_tier", "QUARANTINE")),
            recommended_usage=str(raw.get("recommended_usage", "requires_review")),
            recommended_max_weight=float(max_weight),
            license_expected=str(raw.get("license_expected", "unknown")),
            vendor_specific=bool(raw.get("vendor_specific", False)),
            audit_mode=audit_mode,
            stats_sample_rows=sample_rows,
        )


@dataclass(frozen=True)
class DataFile:
    path: Path
    relative_path: str
    suffix: str
    size_bytes: int


@dataclass(frozen=True)
class DatasetLocation:
    policy: SourcePolicy
    root: Path | None
    path: Path | None
    revision: str
    card_metadata: Mapping[str, Any]

    @property
    def found(self) -> bool:
        return self.path is not None and self.path.is_dir()


def load_source_policies(path: str | Path = DEFAULT_POLICY_PATH) -> tuple[Path, list[SourcePolicy], dict[str, Any]]:
    policy_path = Path(path)
    with policy_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"source policy root must be a mapping: {policy_path}")
    if payload.get("schema_version") != 1:
        raise ValueError("source policy schema_version must be 1")
    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("source policy sources must be a non-empty list")
    policies = [SourcePolicy.from_mapping(item, index) for index, item in enumerate(raw_sources)]
    ids = [policy.dataset_id for policy in policies]
    if len(ids) != len(set(ids)):
        raise ValueError("source policy dataset IDs must be unique")
    repos = [policy.repo for policy in policies]
    if len(repos) != len(set(repos)):
        raise ValueError("source policy repository IDs must be unique")
    return policy_path, policies, dict(payload)


def resolve_dataset_root(
    repo_root: str | Path,
    *,
    explicit_root: str | Path | None = None,
    policy_payload: Mapping[str, Any] | None = None,
) -> Path | None:
    """Resolve a root without scanning broad filesystem areas.

    The order follows the project contract: explicit argument, environment,
    policy default, a few workspace-local conventions, then the known HDD
    location used by the current project.
    """

    root = Path(repo_root).resolve()
    candidates: list[Path] = []
    if explicit_root:
        candidates.append(Path(explicit_root).expanduser())
    env_value = os.environ.get(DATASET_ROOT_ENV)
    if env_value:
        candidates.append(Path(env_value).expanduser())
    if policy_payload:
        default_root = policy_payload.get("default_dataset_root")
        if isinstance(default_root, str) and default_root.strip():
            candidates.append(Path(default_root).expanduser())
    candidates.extend(
        [
            root / "data" / "raw",
            root / "data",
            root.parent / "zipangu-datesets",
            Path("E:/zipangu-datesets"),
        ]
    )
    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        key = os.path.normcase(str(resolved))
        if key in seen:
            continue
        seen.add(key)
        if resolved.is_dir():
            return resolved
    return None


def _local_dir_lookup(root: Path) -> dict[str, Path]:
    try:
        entries = root.iterdir()
    except OSError:
        return {}
    return {
        entry.name.casefold(): entry
        for entry in entries
        if entry.is_dir() and entry.name.casefold() not in {name.casefold() for name in IGNORED_DIRECTORY_NAMES}
    }


def discover_sources(
    policies: Iterable[SourcePolicy],
    repo_root: str | Path,
    *,
    explicit_root: str | Path | None = None,
    policy_payload: Mapping[str, Any] | None = None,
) -> tuple[Path | None, list[DatasetLocation]]:
    dataset_root = resolve_dataset_root(
        repo_root,
        explicit_root=explicit_root,
        policy_payload=policy_payload,
    )
    lookup = _local_dir_lookup(dataset_root) if dataset_root else {}
    locations: list[DatasetLocation] = []
    for policy in policies:
        path = lookup.get(policy.local_dir.casefold()) if dataset_root else None
        if path is None and dataset_root:
            # A small exact-name fallback helps with repositories downloaded
            # using a different case, while avoiding recursive HDD scans.
            fallback = policy.repo.replace("/", "__")
            path = lookup.get(fallback.casefold())
        locations.append(
            DatasetLocation(
                policy=policy,
                root=dataset_root,
                path=path,
                revision=read_local_revision(path) if path else "unknown",
                card_metadata=read_dataset_card(path) if path else {},
            )
        )
    return dataset_root, locations


def _is_ignored(path: Path) -> bool:
    if any(part.casefold() in {name.casefold() for name in IGNORED_DIRECTORY_NAMES} for part in path.parts):
        return True
    return path.suffix.casefold() in IGNORED_FILE_SUFFIXES


def is_data_file(path: Path) -> bool:
    if _is_ignored(path):
        return False
    name = path.name.casefold()
    if name in METADATA_JSON_NAMES:
        return False
    lower = name
    if lower.endswith(".jsonl.gz"):
        return True
    return path.suffix.casefold() in {".jsonl", ".parquet"}


def iter_data_files(dataset_path: str | Path) -> Iterable[DataFile]:
    root = Path(dataset_path)
    if not root.is_dir():
        return
    files: list[DataFile] = []
    for path in root.rglob("*"):
        if not path.is_file() or not is_data_file(path):
            continue
        relative = path.relative_to(root).as_posix()
        suffix = ".jsonl.gz" if relative.casefold().endswith(".jsonl.gz") else path.suffix.casefold()
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        files.append(DataFile(path=path, relative_path=relative, suffix=suffix, size_bytes=size))
    yield from sorted(files, key=lambda item: item.relative_path.casefold())


def infer_config(relative_path: str) -> str:
    parts = Path(relative_path).parts
    for part in parts[:-1]:
        if re.fullmatch(r"v\d+(?:\.\d+)*", part, flags=re.IGNORECASE):
            return part
    for part in parts[:-1]:
        if part.casefold() in {"agent_traces", "openai_chat"}:
            return part
    return "default"


def infer_split(relative_path: str) -> str:
    stem = Path(relative_path).name.casefold()
    if stem.endswith(".jsonl.gz"):
        stem = stem[:-9]
    else:
        stem = Path(stem).stem
    for token in ("eval", "validation", "test", "train"):
        if re.search(rf"(?:^|[-_]){token}(?:[-_.]|$)", stem):
            return token
    match = re.search(r"(?:ultra-v\d+_)?(code|math|stem)_(hi|ja|ko|pt)", stem)
    if match:
        return f"{match.group(1)}_{match.group(2)}"
    return stem or "unknown"


def read_dataset_card(dataset_path: Path | None) -> dict[str, Any]:
    if dataset_path is None:
        return {}
    readme = dataset_path / "README.md"
    if not readme.is_file():
        return {"readme": False}
    try:
        text = readme.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {"readme": False}
    metadata: dict[str, Any] = {"readme": True}
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            try:
                front_matter = yaml.safe_load(parts[1])
            except yaml.YAMLError:
                front_matter = None
            if isinstance(front_matter, Mapping):
                metadata.update({str(key): value for key, value in front_matter.items()})
    metadata["readme_bytes"] = len(text.encode("utf-8"))
    return metadata


def _metadata_candidates(dataset_path: Path) -> Iterable[Path]:
    cache = dataset_path / ".cache" / "huggingface" / "download"
    if not cache.is_dir():
        return
    yield from sorted(cache.rglob("*.metadata"), key=lambda path: str(path).casefold())


def read_local_revision(dataset_path: Path | None) -> str:
    if dataset_path is None:
        return "unknown"
    revisions: set[str] = set()
    for metadata_path in _metadata_candidates(dataset_path):
        try:
            text = metadata_path.read_text(encoding="utf-8", errors="replace")
            payload = json.loads(text)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, Mapping):
            continue
        for key in ("commit_hash", "revision", "commit"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                revisions.add(value.strip())
    if len(revisions) == 1:
        return next(iter(revisions))
    if len(revisions) > 1:
        return "mixed:" + ",".join(sorted(revisions))
    return "unknown"


def _safe_sample(value: Any, *, depth: int = 0) -> Any:
    if depth > 2:
        return "<nested>"
    if isinstance(value, str):
        compact = " ".join(value.split())
        if len(compact) > 160:
            compact = compact[:157] + "..."
        return compact
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    if isinstance(value, list):
        return [_safe_sample(item, depth=depth + 1) for item in value[:3]]
    if isinstance(value, Mapping):
        return {str(key): _safe_sample(item, depth=depth + 1) for key, item in list(value.items())[:12]}
    return str(value)


def safe_sample(value: Any) -> Any:
    """Return a bounded sample suitable for a tracked inventory report."""

    return _safe_sample(value)


def inventory_location(location: DatasetLocation, *, sample_rows: int = 5, batch_size: int = 256) -> dict[str, Any]:
    """Collect file and schema metadata through the streaming adapter."""

    from .adapters import AdapterUnavailable, inspect_location

    result: dict[str, Any] = {
        "dataset_id": location.policy.dataset_id,
        "repo": location.policy.repo,
        "local_path": str(location.path) if location.path else None,
        "found": location.found,
        "revision": location.revision,
        "card_metadata": dict(location.card_metadata),
        "files": [],
        "file_count": 0,
        "physical_file_count": 0,
        "logical_file_count": 0,
        "total_bytes": 0,
        "logical_total_bytes": 0,
        "row_count": 0,
        "row_count_known": False,
        "unknown_row_count_files": 0,
        "configs": [],
        "splits": [],
        "schema": {},
        "samples": [],
        "adapter_warnings": [],
        "language_metadata": location.card_metadata.get("language", []),
        "source_metadata": {
            "repo": location.policy.repo,
            "category": location.policy.category,
            "vendor_specific": location.policy.vendor_specific,
            "license_expected": location.policy.license_expected,
        },
        "teacher_source_fields": [],
        "length_profile": {},
    }
    if not location.found:
        result["adapter_warnings"].append("dataset_path_missing")
        return result
    try:
        inspected = inspect_location(location, sample_rows=sample_rows, batch_size=batch_size)
    except AdapterUnavailable as exc:
        result["adapter_warnings"].append(str(exc))
        inspected = None
    if inspected is None:
        return result
    result.update(inspected)
    return result


def location_to_manifest(location: DatasetLocation, inventory: Mapping[str, Any]) -> dict[str, Any]:
    """Create a raw-source manifest without copying raw text."""

    return {
        "schema_version": 1,
        "dataset_id": location.policy.dataset_id,
        "source_repo": location.policy.repo,
        "local_path": inventory.get("local_path"),
        "source_revision": location.revision,
        "role": location.policy.role,
        "category": location.policy.category,
        "found": location.found,
        "file_count": inventory.get("file_count", 0),
        "physical_file_count": inventory.get("physical_file_count", inventory.get("file_count", 0)),
        "logical_file_count": inventory.get("logical_file_count", inventory.get("file_count", 0)),
        "total_bytes": inventory.get("total_bytes", 0),
        "logical_total_bytes": inventory.get("logical_total_bytes", inventory.get("total_bytes", 0)),
        "row_count_known": inventory.get("row_count_known", False),
        "unknown_row_count_files": inventory.get("unknown_row_count_files", 0),
        "file_formats": inventory.get("file_formats", []),
        "configs": inventory.get("configs", []),
        "splits": inventory.get("splits", []),
        "row_count": inventory.get("row_count", 0),
        "schema": inventory.get("schema", {}),
        "license_metadata": inventory.get("card_metadata", {}).get("license"),
        "language_metadata": inventory.get("language_metadata", inventory.get("card_metadata", {}).get("language")),
        "source_metadata": inventory.get("source_metadata", {}),
        "teacher_source_fields": inventory.get("teacher_source_fields", []),
        "length_profile": inventory.get("length_profile", {}),
        "readme_present": bool(inventory.get("card_metadata", {}).get("readme")),
        "provenance_status": "known_revision" if location.revision != "unknown" else "unknown_revision",
        "raw_data_unchanged": True,
    }


def policies_as_dict(policies: Iterable[SourcePolicy]) -> list[dict[str, Any]]:
    return [asdict(policy) for policy in policies]
