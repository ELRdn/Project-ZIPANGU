"""Build approved-only pilot candidates, or fail closed with a report."""

from __future__ import annotations

import gzip
import heapq
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from ..registry import split_issues, validate_recipe_against_registry, validate_registry
from .adapters import iter_source_rows
from .audit import _runtime_audit_dir
from .canonical import canonicalize_source_row
from .dedup import iter_audit_records
from .discovery import DatasetLocation
from .filtering import apply_filter, hard_filter
from .quality import score_record
from .sampling import deterministic_key, select_by_token_mass


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def _approved_gate(repo_root: Path, recipe: Mapping[str, Any]) -> tuple[str, list[str]]:
    registry_path = repo_root / "configs" / "datasets" / "registry.yaml"
    registry = _load_yaml(registry_path)
    issues = validate_registry(registry)
    issues.extend(validate_recipe_against_registry(recipe, registry, repo_root=repo_root))
    errors, blocked = split_issues(issues)
    if errors:
        return "ERROR", [issue.render() for issue in errors]
    if blocked:
        return "BLOCKED", [issue.render() for issue in blocked]
    return "PASS", []


def _selected_audit_records(
    repo_root: Path,
    recipe: Mapping[str, Any],
    target_tokens: int,
) -> list[dict[str, Any]]:
    """Select from a bounded deterministic reservoir of audit metadata.

    Audit indexes can be much larger than memory.  Keep only the lowest
    deterministic priorities needed by each recipe bucket; raw text is never
    loaded during this planning pass.
    """

    max_records_per_bucket = 100_000
    bucket_candidates: dict[str, set[str]] = {}
    for bucket_name, bucket in (recipe.get("buckets") or {}).items():
        if isinstance(bucket, Mapping):
            bucket_candidates[str(bucket_name)] = {
                str(dataset_id) for dataset_id in (bucket.get("candidates") or [])
            }

    heaps: dict[str, list[tuple[int, int, dict[str, Any]]]] = {
        bucket_name: [] for bucket_name in bucket_candidates
    }
    sequence = 0
    for record in iter_audit_records(_runtime_audit_dir(repo_root)):
        if record.get("eligibility") is not True or record.get("contamination_status") != "clear":
            continue
        dataset_id = str(record.get("source_dataset"))
        priority = int(deterministic_key(record), 16)
        for bucket_name, candidates in bucket_candidates.items():
            if dataset_id not in candidates:
                continue
            sequence += 1
            entry = (-priority, sequence, dict(record))
            heap = heaps[bucket_name]
            if len(heap) < max_records_per_bucket:
                heapq.heappush(heap, entry)
            elif priority < -heap[0][0]:
                heapq.heapreplace(heap, entry)

    selected: list[dict[str, Any]] = []
    sampling_mass = recipe.get("sampling_mass") if isinstance(recipe.get("sampling_mass"), Mapping) else {}
    for bucket_name, heap in heaps.items():
        mass = float(sampling_mass.get(bucket_name, 0.0))
        bucket_records = [entry[2] for entry in heap]
        selected.extend(
            select_by_token_mass(
                bucket_records,
                target_tokens=max(1, int(target_tokens * mass)),
            )
        )
    unique: dict[str, dict[str, Any]] = {}
    for record in selected:
        unique[str(record.get("record_id"))] = record
    return list(unique.values())


def _reconstruct_selected(
    selected: list[Mapping[str, Any]],
    locations: Mapping[str, DatasetLocation],
    *,
    batch_size: int = 512,
) -> list[dict[str, Any]]:
    wanted = {(str(record.get("source_dataset")), str(record.get("source_row_id"))) for record in selected}
    output: list[dict[str, Any]] = []
    for dataset_id, location in locations.items():
        dataset_wanted = {row_id for source, row_id in wanted if source == dataset_id}
        if not dataset_wanted:
            continue
        for source_row in iter_source_rows(location, batch_size=batch_size):
            if source_row.source_row_id not in dataset_wanted:
                continue
            record = canonicalize_source_row(
                source_row,
                source_metadata={
                    "revision": location.revision,
                    "license": location.card_metadata.get("license", "unknown"),
                    "vendor_specific": location.policy.vendor_specific,
                },
            )
            record = apply_filter(record, hard_filter(record, source_role=location.policy.role))
            record.update(score_record(record))
            if record.get("eligibility") is True:
                output.append(record)
    return output


def _write_output(records: list[Mapping[str, Any]], output_path: Path, output_format: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "jsonl.gz":
        with gzip.open(output_path, "wt", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(dict(record), ensure_ascii=False, separators=(",", ":")) + "\n")
        return
    try:
        import pyarrow as pa
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - optional environment
        raise RuntimeError("Parquet pilot output requires pyarrow; use --format jsonl.gz for a local smoke test") from exc
    table = pa.Table.from_pylist([dict(record) for record in records])
    parquet.write_table(table, output_path, compression="zstd")


def build_pilot(
    repo_root: str | Path,
    recipe_path: str | Path,
    locations: Mapping[str, DatasetLocation],
    *,
    target_tokens: int,
    output_dir: str | Path | None = None,
    output_format: str = "parquet",
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    recipe = _load_yaml(Path(recipe_path))
    gate, issues = _approved_gate(root, recipe)
    status: dict[str, Any] = {
        "status": gate,
        "recipe": str(recipe_path),
        "target_tokens": target_tokens,
        "records_selected": 0,
        "issues": issues,
        "auto_approved": False,
    }
    if gate != "PASS":
        return status
    selected = _selected_audit_records(root, recipe, target_tokens)
    records = _reconstruct_selected(selected, locations)
    status["records_selected"] = len(records)
    if not records:
        status["status"] = "BLOCKED"
        status["issues"] = ["no approved, contamination-clear audit records available"]
        return status
    destination = Path(output_dir) if output_dir else root / "data" / "processed" / "c-i" / f"pilot-{target_tokens}"
    suffix = ".jsonl.gz" if output_format == "jsonl.gz" else ".parquet"
    output_path = destination / f"candidate{suffix}"
    _write_output(records, output_path, output_format)
    status["output_path"] = str(output_path)
    status["status"] = "PASS"
    return status
