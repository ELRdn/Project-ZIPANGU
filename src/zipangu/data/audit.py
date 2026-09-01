"""Streaming canonicalization, hard filtering and deterministic scoring."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping

from .adapters import AdapterUnavailable, iter_source_rows
from .canonical import canonicalize_source_row
from .discovery import DatasetLocation
from .filtering import apply_filter, hard_filter
from .provenance import audit_manifest_record
from .quality import quantile, score_record


def _runtime_audit_dir(repo_root: Path) -> Path:
    path = repo_root / "data" / "manifests" / "_runtime" / "audit"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _artifact_dir(repo_root: Path) -> Path:
    path = repo_root / "reports" / "dataset_audit" / "_artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _summary_template(location: DatasetLocation) -> dict[str, Any]:
    return {
        "dataset_id": location.policy.dataset_id,
        "repo": location.policy.repo,
        "local_path": str(location.path) if location.path else None,
        "audit_scope": location.policy.audit_mode,
        "raw_rows": 0,
        "rows_scanned": 0,
        "canonical_rows": 0,
        "valid_rows": 0,
        "eligible_rows": 0,
        "rejected_rows": 0,
        "eligible_token_count": 0,
        "excluded_eval_rows": 0,
        "reasoning_ready_rows": 0,
        "tool_ready_rows": 0,
        "language_counts": {},
        "category_counts": {},
        "teacher_counts": {},
        "split_counts": {},
        "rejection_reasons": {},
        "quality": {},
        "provenance": {
            "source_revision": location.revision,
            "license_metadata": location.card_metadata.get("license", "unknown"),
            "contamination_status": "not_checked_missing_eval_source",
        },
        "warnings": [],
    }


def _increment(mapping: dict[str, int], key: Any) -> None:
    name = str(key or "unknown")
    mapping[name] = mapping.get(name, 0) + 1


def _print_progress(dataset_id: str, rows: int, accepted: int, rejected: int, started: float, output_path: Path) -> None:
    elapsed = max(0.001, time.monotonic() - started)
    rate = rows / elapsed
    output_size = output_path.stat().st_size if output_path.exists() else 0
    print(
        f"PROGRESS dataset={dataset_id} rows={rows} accepted={accepted} rejected={rejected} "
        f"throughput={rate:.1f}/s elapsed={elapsed:.1f}s output_bytes={output_size}"
    )


def audit_location(
    location: DatasetLocation,
    *,
    repo_root: str | Path,
    inventory: Mapping[str, Any] | None = None,
    quality_config: Mapping[str, Any] | None = None,
    batch_size: int = 512,
    max_rows: int | None = None,
    progress_every: int = 5000,
) -> dict[str, Any]:
    """Audit one source and write a content-free row decision index."""

    root = Path(repo_root)
    summary = _summary_template(location)
    if not location.found:
        summary["warnings"].append("dataset_path_missing")
        return summary
    source_meta = {
        "revision": location.revision,
        "license": location.card_metadata.get("license", "unknown"),
        "vendor_specific": location.policy.vendor_specific,
    }
    thresholds = dict((quality_config or {}).get("hard_filter", {}))
    audit_dir = _runtime_audit_dir(root)
    output_path = audit_dir / f"{location.policy.dataset_id}.jsonl.gz"
    started = time.monotonic()
    quality_values: list[float] = []
    char_values: list[int] = []
    token_values: list[int] = []
    rows_to_scan = max_rows
    if location.policy.audit_mode == "stats_only":
        rows_to_scan = min(max_rows, location.policy.stats_sample_rows) if max_rows else location.policy.stats_sample_rows
        summary["audit_scope"] = "sampled"
    try:
        with gzip.open(output_path, "wt", encoding="utf-8") as output:
            for source_row in iter_source_rows(location, batch_size=batch_size, max_rows=rows_to_scan):
                summary["rows_scanned"] += 1
                summary["raw_rows"] += 1
                _increment(summary["split_counts"], source_row.source_split)
                record = canonicalize_source_row(source_row, source_metadata=source_meta)
                summary["canonical_rows"] += 1
                filter_result = hard_filter(
                    record,
                    thresholds=thresholds,
                    source_role=location.policy.role,
                )
                record = apply_filter(record, filter_result)
                score = score_record(record)
                record.update(score)
                if filter_result.eligible:
                    summary["valid_rows"] += 1
                    summary["eligible_rows"] += 1
                    summary["eligible_token_count"] += int(record.get("token_count") or 0)
                else:
                    summary["rejected_rows"] += 1
                    for reason in filter_result.reasons:
                        _increment(summary["rejection_reasons"], reason)
                if source_row.source_split in {"eval", "test", "validation"}:
                    summary["excluded_eval_rows"] += 1
                if record.get("reasoning_sft_ready"):
                    summary["reasoning_ready_rows"] += 1
                if record.get("tool_sft_ready"):
                    summary["tool_ready_rows"] += 1
                _increment(summary["language_counts"], record.get("language_overall"))
                _increment(summary["category_counts"], record.get("category"))
                _increment(summary["teacher_counts"], record.get("teacher_model"))
                quality_values.append(float(record.get("quality_score") or 0.0))
                char_values.append(int(record.get("char_count") or 0))
                token_values.append(int(record.get("token_count") or 0))
                if location.policy.audit_mode != "stats_only":
                    output.write(json.dumps(audit_manifest_record(record), ensure_ascii=False, separators=(",", ":")) + "\n")
                if summary["rows_scanned"] % progress_every == 0:
                    _print_progress(
                        location.policy.dataset_id,
                        summary["rows_scanned"],
                        summary["eligible_rows"],
                        summary["rejected_rows"],
                        started,
                        output_path,
                    )
    except AdapterUnavailable as exc:
        summary["warnings"].append(str(exc))
    if inventory:
        summary["raw_rows"] = int(inventory.get("row_count") or summary["raw_rows"])
        if location.policy.audit_mode == "stats_only":
            summary["warnings"].append("full_row_audit_skipped_by_future_cpt_policy")
    if quality_values:
        summary["quality"] = {
            "mean": round(sum(quality_values) / len(quality_values), 4),
            "median": quantile(quality_values, 0.5),
            "p10": quantile(quality_values, 0.1),
            "p90": quantile(quality_values, 0.9),
            "sampled_values": len(quality_values),
        }
    if char_values:
        summary["char_stats"] = {
            "median": quantile([float(value) for value in char_values], 0.5),
            "p95": quantile([float(value) for value in char_values], 0.95),
        }
    if token_values:
        summary["token_stats"] = {
            "median_estimate": quantile([float(value) for value in token_values], 0.5),
            "p95_estimate": quantile([float(value) for value in token_values], 0.95),
            "tokenizer_pending": True,
        }
    return summary


def write_audit_summary(repo_root: str | Path, summaries: Iterable[Mapping[str, Any]]) -> Path:
    path = _artifact_dir(Path(repo_root)) / "audit_summary.json"
    payload = {str(item.get("dataset_id")): dict(item) for item in summaries}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_audit_summary(repo_root: str | Path) -> dict[str, dict[str, Any]]:
    path = _artifact_dir(Path(repo_root)) / "audit_summary.json"
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}
