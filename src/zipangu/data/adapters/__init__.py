"""Compatibility facade for the streaming adapters.

The repository was initialized without a baseline commit, so the original
single-file adapter is kept as a read-only implementation module while this
package supplies the corrected, file-at-a-time inventory path.  Both import
styles resolve to the same public API once the package is installed.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterator


_LEGACY_PATH = Path(__file__).resolve().parents[1] / "adapters.py"
_SPEC = importlib.util.spec_from_file_location("zipangu.data._legacy_adapters", _LEGACY_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - packaging failure
    raise ImportError(f"could not load adapter implementation: {_LEGACY_PATH}")
_legacy = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _legacy
_SPEC.loader.exec_module(_legacy)

AdapterUnavailable = _legacy.AdapterUnavailable
DataFile = _legacy.DataFile
DatasetLocation = _legacy.DatasetLocation
SourceRow = _legacy.SourceRow
iter_data_files = _legacy.iter_data_files
infer_config = _legacy.infer_config
infer_split = _legacy.infer_split
safe_sample = _legacy.safe_sample
parquet_metadata = _legacy.parquet_metadata


def _logical_files(location: DatasetLocation, files: list[Any] | None = None) -> list[Any]:
    physical_files = list(files if files is not None else iter_data_files(location.path))
    if location.policy.dataset_id == "fable_5_premium":
        preferred = [
            file
            for file in physical_files
            if file.relative_path.casefold().startswith("agent_traces/")
            and file.suffix in {".jsonl", ".jsonl.gz"}
        ]
        if preferred:
            return preferred
    return physical_files


def _iter_file_rows(file: Any, location: Any, *, batch_size: int) -> Iterator[SourceRow]:
    if file.suffix in {".jsonl", ".jsonl.gz"}:
        yield from _legacy._read_jsonl(file, location)
    elif file.suffix == ".parquet":
        yield from _legacy._read_parquet(file, location, batch_size=batch_size)
    else:
        yield from _legacy._read_json(file, location)


def iter_raw_rows(
    location: DatasetLocation,
    *,
    batch_size: int = 512,
    max_rows: int | None = None,
) -> Iterator[SourceRow]:
    yielded = 0
    if not location.found:
        return
    for file in _logical_files(location):
        for row in _iter_file_rows(file, location, batch_size=batch_size):
            yield row
            yielded += 1
            if max_rows is not None and yielded >= max_rows:
                return


def iter_source_rows(
    location: DatasetLocation,
    *,
    batch_size: int = 512,
    max_rows: int | None = None,
    group_event_streams: bool = True,
) -> Iterator[SourceRow]:
    if group_event_streams and location.policy.dataset_id == "claude_fable_code":
        rows = _legacy._event_stream_rows(location, batch_size=batch_size)
    else:
        rows = iter_raw_rows(location, batch_size=batch_size, max_rows=max_rows)
    yielded = 0
    for row in rows:
        yield row
        yielded += 1
        if max_rows is not None and yielded >= max_rows:
            return


_TEXT_FIELD_NAMES = frozenset(
    {
        "answer",
        "assistant",
        "assistant_final",
        "completion",
        "completion_text",
        "content",
        "conversations",
        "input",
        "instruction",
        "messages",
        "output",
        "problem",
        "prompt",
        "question",
        "query",
        "reasoning",
        "reasoning_content",
        "response",
        "system",
        "text",
        "user",
    }
)


_TEXT_FIELD_SUFFIXES = ("_text", "_content", "_messages", "_prompt", "_completion", "_reasoning", "_json")


def _is_text_field(field_name: str | None) -> bool:
    return field_name is None or field_name in _TEXT_FIELD_NAMES or any(
        field_name.endswith(suffix) for suffix in _TEXT_FIELD_SUFFIXES
    )


def _raw_char_count(value: Any, *, field_name: str | None = None) -> int:
    if isinstance(value, str):
        return len(value) if _is_text_field(field_name) else 0
    if isinstance(value, Mapping):
        return sum(
            _raw_char_count(item, field_name=str(key).casefold())
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return sum(_raw_char_count(item, field_name=field_name) for item in value)
    return 0


def _quantile(values: list[int], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction, 2)


def _length_profile(
    location: DatasetLocation,
    *,
    sample_rows: int = 10_000,
    batch_size: int = 256,
) -> dict[str, Any]:
    lengths: list[int] = []
    if sample_rows > 0:
        for source_row in iter_raw_rows(location, batch_size=batch_size, max_rows=sample_rows):
            length = _raw_char_count(source_row.raw)
            if length > 0:
                lengths.append(length)
    return {
        "unit": "raw_text_field_characters",
        "selection": "deterministic_logical_file_order_prefix",
        "sample_rows_requested": sample_rows,
        "observed_rows": len(lengths),
        "median": _quantile(lengths, 0.5),
        "p95": _quantile(lengths, 0.95),
    }


def _teacher_source_fields(schema: Mapping[str, Any]) -> list[str]:
    markers = ("teacher", "model", "source", "generator", "creator", "provider", "author")
    return sorted(
        str(name)
        for name in schema
        if any(marker in str(name).casefold() for marker in markers)
    )


def inspect_location(
    location: DatasetLocation,
    *,
    sample_rows: int = 5,
    batch_size: int = 256,
    length_sample_rows: int = 10_000,
) -> dict[str, Any]:
    """Inspect each physical file once; JSONL is never reread per sibling file."""

    files = list(iter_data_files(location.path)) if location.found else []
    logical_files = _logical_files(location, files)
    result: dict[str, Any] = {
        "files": [],
        "file_count": len(files),
        "physical_file_count": len(files),
        "logical_file_count": len(logical_files),
        "total_bytes": sum(file.size_bytes for file in files),
        "logical_total_bytes": sum(file.size_bytes for file in logical_files),
        "file_formats": sorted({file.suffix for file in files}),
        "row_count": 0,
        "row_count_known": True,
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
    schemas: dict[str, set[str]] = {}
    configs: set[str] = set()
    splits: set[str] = set()
    for file in logical_files:
        config = infer_config(file.relative_path)
        split = infer_split(file.relative_path)
        configs.add(config)
        splits.add(split)
        file_result: dict[str, Any] = {
            "path": file.relative_path,
            "bytes": file.size_bytes,
            "format": file.suffix,
            "config": config,
            "split": split,
        }
        if file.suffix == ".parquet":
            try:
                row_count, schema, samples = parquet_metadata(file.path)
            except AdapterUnavailable as exc:
                row_count = 0
                schema = {}
                samples = []
                result["row_count_known"] = False
                result["unknown_row_count_files"] += 1
                warning = str(exc)
                if warning not in result["adapter_warnings"]:
                    result["adapter_warnings"].append(warning)
            file_result.update({"rows": row_count, "schema": schema, "samples": samples[:sample_rows]})
        else:
            row_count = 0
            schema_values: dict[str, set[str]] = {}
            samples = []
            for row in _iter_file_rows(file, location, batch_size=batch_size):
                row_count += 1
                for key, value in row.raw.items():
                    schema_values.setdefault(str(key), set()).add(type(value).__name__)
                if len(samples) < sample_rows:
                    samples.append(safe_sample(row.raw))
            schema = {key: sorted(values) for key, values in sorted(schema_values.items())}
            file_result.update({"rows": row_count, "schema": schema, "samples": samples})
        result["row_count"] += row_count
        result["files"].append(file_result)
        for name, type_names in schema.items():
            schemas.setdefault(name, set()).update(type_names if isinstance(type_names, list) else [str(type_names)])
        if len(result["samples"]) < sample_rows:
            result["samples"].extend(samples[: sample_rows - len(result["samples"])])
    if result["unknown_row_count_files"]:
        result["adapter_warnings"].append(
            f"row_count_unknown_for_files:{result['unknown_row_count_files']}"
        )
    result["configs"] = sorted(configs)
    result["splits"] = sorted(splits)
    result["schema"] = {key: sorted(values) for key, values in sorted(schemas.items())}
    result["samples"] = result["samples"][:sample_rows]
    result["teacher_source_fields"] = _teacher_source_fields(result["schema"])
    result["length_profile"] = _length_profile(
        location,
        sample_rows=length_sample_rows,
        batch_size=batch_size,
    )
    return result
