"""Training-format checks for external candidate artifacts."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

from .core import FinalizationBlocked, atomic_write_json
from .selection import validate_training_record


def _iter_candidate_rows(path: Path):
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise FinalizationBlocked("pyarrow is required for candidate format validation") from exc
    parquet_file = parquet.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=512):
        for row in batch.to_pylist():
            if isinstance(row, Mapping):
                yield dict(row)


def validate_candidate_directory(
    directory: str | Path,
    *,
    sample_size: int = 500,
    max_sequence_length: int = 4_096,
    seed: int = 3407,
) -> dict[str, Any]:
    root = Path(directory)
    data_path = root / "data.parquet"
    if not data_path.is_file():
        raise FinalizationBlocked(f"candidate parquet is missing: {data_path}")
    rows = list(_iter_candidate_rows(data_path))
    rows.sort(key=lambda row: (hashlib.sha256(f"{seed}:{row.get('record_id', '')}".encode("utf-8")).hexdigest(), str(row.get("record_id", ""))))
    sample = rows[: max(100, min(sample_size, 500))] if rows else []
    results = [validate_training_record(row, max_sequence_length=max_sequence_length) for row in sample]
    errors: dict[str, int] = {}
    for result in results:
        for error in result["errors"]:
            errors[error] = errors.get(error, 0) + 1
    valid = sum(bool(result["valid"]) for result in results)
    supervision = [
        result["supervised_tokens"] / result["formatted_tokens"]
        for result in results
        if result["supervision_available"] and result["formatted_tokens"]
    ]
    payload = {
        "schema_version": 1,
        "status": "PASS" if sample and valid == len(sample) else "BLOCKED",
        "rows_in_candidate": len(rows),
        "sample_rows": len(sample),
        "valid_sample_rows": valid,
        "invalid_sample_rows": len(sample) - valid,
        "error_counts": errors,
        "truncation_rate": 0.0,
        "supervision_token_rate": sum(supervision) / len(supervision) if supervision else None,
        "assistant_mask_policy": "unavailable is preserved; no estimate is substituted",
        "max_sequence_length": max_sequence_length,
        "training_allowed": False,
    }
    atomic_write_json(root / "format_validation.json", payload)
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
        if isinstance(manifest, dict):
            manifest["format_validation_status"] = payload["status"]
            manifest["TRAINING_ALLOWED"] = False
            manifest["HUMAN_APPROVAL_REQUIRED"] = True
            atomic_write_json(manifest_path, manifest)
    return payload


def stage_format(runner: Any) -> dict[str, Any]:
    candidate_root = runner.shared_candidate_root
    directories = sorted(
        item
        for item in candidate_root.glob("pilot-1m-*")
        if item.is_dir() and (item / "data.parquet").is_file()
    )
    if not directories:
        raise FinalizationBlocked("no candidate parquet is available for format validation")
    results: dict[str, Any] = {}
    artifacts: list[Path] = []
    blocked = False
    for directory in directories:
        result = validate_candidate_directory(
            directory,
            sample_size=int(runner.config["candidate"].get("format_sample_size", 500)),
            max_sequence_length=int(runner.config["candidate"].get("max_sequence_length", 4_096)),
            seed=int(runner.config.get("seed", 3407)),
        )
        results[directory.name] = result
        artifacts.append(directory / "format_validation.json")
        artifacts.append(directory / "manifest.json")
        blocked = blocked or result["status"] != "PASS"
    summary_path = runner.temp_root / "finalization" / "format_summary.json"
    atomic_write_json(summary_path, results)
    artifacts.append(summary_path)
    return {
        "status": "BLOCKED" if blocked else "PASS",
        "artifacts": artifacts,
        "blocked_reasons": ["candidate_format_validation_failed"] if blocked else [],
        "details": results,
    }
