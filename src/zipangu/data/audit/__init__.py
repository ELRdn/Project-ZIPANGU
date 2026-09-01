"""Streaming audit facade with bounded statistics for very large sources."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import time
from typing import Any, Mapping

from ..adapters import AdapterUnavailable, iter_source_rows
from ..canonical import canonicalize_source_row
from ..filtering import apply_filter, hard_filter
from ..provenance import audit_manifest_record
from ..quality import quantile, score_record


def _runtime_audit_dir(repo_root: Path) -> Path:
    path = repo_root / "data" / "manifests" / "_runtime" / "audit"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _artifact_dir(repo_root: Path) -> Path:
    path = repo_root / "reports" / "dataset_audit" / "_artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _summary_template(location: Any) -> dict[str, Any]:
    return {
        "dataset_id": location.policy.dataset_id,
        "repo": location.policy.repo,
        "local_path": str(location.path) if location.path else None,
        "audit_scope": location.policy.audit_mode,
        "max_rows": None,
        "raw_rows": 0,
        "raw_rows_known": None,
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


class _Reservoir:
    def __init__(self, limit: int = 4096) -> None:
        if limit <= 0:
            raise ValueError("reservoir limit must be positive")
        self.limit = limit
        self.count = 0
        self._state = 0x9E3779B9
        self.values: list[float] = []

    def add(self, value: float) -> None:
        self.count += 1
        if len(self.values) < self.limit:
            self.values.append(value)
            return
        self._state = (1664525 * self._state + 1013904223) & 0xFFFFFFFF
        slot = self._state % self.count
        if slot < self.limit:
            self.values[slot] = value

    def summary(self) -> dict[str, Any]:
        return {
            "mean": round(sum(self.values) / len(self.values), 4) if self.values else None,
            "median": quantile(self.values, 0.5),
            "p10": quantile(self.values, 0.1),
            "p90": quantile(self.values, 0.9),
            "sampled_values": len(self.values),
            "observed_values": self.count,
        }


def _increment(mapping: dict[str, int], key: Any) -> None:
    name = str(key or "unknown")
    mapping[name] = mapping.get(name, 0) + 1


def _progress(dataset_id: str, rows: int, accepted: int, rejected: int, start: float, path: Path) -> None:
    elapsed = max(0.001, time.monotonic() - start)
    output_size = path.stat().st_size if path.exists() else 0
    print(
        f"PROGRESS dataset={dataset_id} rows={rows} accepted={accepted} rejected={rejected} "
        f"throughput={rows / elapsed:.1f}/s elapsed={elapsed:.1f}s output_bytes={output_size}"
    )


def audit_location(
    location: Any,
    *,
    repo_root: str | Path,
    inventory: Mapping[str, Any] | None = None,
    quality_config: Mapping[str, Any] | None = None,
    batch_size: int = 512,
    max_rows: int | None = None,
    progress_every: int = 5000,
) -> dict[str, Any]:
    summary = _summary_template(location)
    output_path = _runtime_audit_dir(Path(repo_root)) / f"{location.policy.dataset_id}.jsonl.gz"
    if not location.found:
        with gzip.open(output_path, "wt", encoding="utf-8"):
            pass
        summary["warnings"].append("dataset_path_missing")
        summary["raw_rows_known"] = False
        return summary
    summary["max_rows"] = max_rows
    if location.policy.audit_mode == "stats_only":
        rows_to_scan = location.policy.stats_sample_rows
        if max_rows is not None:
            rows_to_scan = min(rows_to_scan, max_rows)
        summary["audit_scope"] = "stats_only_sampled"
    else:
        rows_to_scan = max_rows
        if max_rows is not None:
            summary["audit_scope"] = "bounded"
    source_meta = {
        "revision": location.revision,
        "license": location.card_metadata.get("license", "unknown"),
        "vendor_specific": location.policy.vendor_specific,
    }
    thresholds = dict((quality_config or {}).get("hard_filter", {}))
    quality_values = _Reservoir()
    char_values = _Reservoir()
    token_values = _Reservoir()
    started = time.monotonic()
    try:
        with gzip.open(output_path, "wt", encoding="utf-8") as output:
            for source_row in iter_source_rows(location, batch_size=batch_size, max_rows=rows_to_scan):
                summary["rows_scanned"] += 1
                summary["raw_rows"] += 1
                _increment(summary["split_counts"], source_row.source_split)
                record = canonicalize_source_row(source_row, source_metadata=source_meta)
                summary["canonical_rows"] += 1
                result = hard_filter(record, thresholds=thresholds, source_role=location.policy.role)
                record = apply_filter(record, result)
                record.update(score_record(record))
                if result.eligible:
                    summary["valid_rows"] += 1
                    summary["eligible_rows"] += 1
                    summary["eligible_token_count"] += int(record.get("token_count") or 0)
                else:
                    summary["rejected_rows"] += 1
                    for reason in result.reasons:
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
                quality_values.add(float(record.get("quality_score") or 0.0))
                char_values.add(float(record.get("char_count") or 0))
                token_values.add(float(record.get("token_count") or 0))
                if location.policy.audit_mode != "stats_only":
                    output.write(json.dumps(audit_manifest_record(record), ensure_ascii=False, separators=(",", ":")) + "\n")
                if progress_every > 0 and summary["rows_scanned"] % progress_every == 0:
                    _progress(location.policy.dataset_id, summary["rows_scanned"], summary["eligible_rows"], summary["rejected_rows"], started, output_path)
    except AdapterUnavailable as exc:
        summary["warnings"].append(str(exc))
    if inventory:
        summary["raw_rows"] = int(inventory.get("row_count") or summary["raw_rows"])
        summary["raw_rows_known"] = bool(inventory.get("row_count_known", False))
    if location.policy.audit_mode == "stats_only":
        summary["warnings"].append("full_row_audit_skipped_by_future_cpt_policy")
    summary["quality"] = quality_values.summary()
    summary["char_stats"] = {"median": quantile(char_values.values, 0.5), "p95": quantile(char_values.values, 0.95), "observed_values": char_values.count}
    summary["token_stats"] = {"median_estimate": quantile(token_values.values, 0.5), "p95_estimate": quantile(token_values.values, 0.95), "tokenizer_pending": True, "observed_values": token_values.count}
    return summary


def write_audit_summary(
    repo_root: str | Path,
    summaries: Any,
    *,
    merge: bool = False,
) -> Path:
    path = _artifact_dir(Path(repo_root)) / "audit_summary.json"
    payload = load_audit_summary(repo_root) if merge else {}
    payload.update({str(item.get("dataset_id")): dict(item) for item in summaries})
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_audit_summary(repo_root: str | Path) -> dict[str, dict[str, Any]]:
    path = _artifact_dir(Path(repo_root)) / "audit_summary.json"
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}
