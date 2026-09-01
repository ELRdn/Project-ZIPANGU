"""Memory-bounded readers for local JSONL and Parquet dataset sources."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import json
from pathlib import Path
from typing import Any, Iterator, Mapping

from .discovery import (
    DataFile,
    DatasetLocation,
    infer_config,
    infer_split,
    iter_data_files,
    safe_sample,
)


class AdapterUnavailable(RuntimeError):
    """Raised when an installed optional reader is required but unavailable."""


@dataclass(frozen=True)
class SourceRow:
    dataset_id: str
    repo: str
    source_path: str
    source_row_id: str
    source_config: str
    source_split: str
    row_index: int
    raw: Mapping[str, Any]


def _open_text(path: Path):
    if path.name.casefold().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def _read_jsonl(file: DataFile, location: DatasetLocation) -> Iterator[SourceRow]:
    config = infer_config(file.relative_path)
    split = infer_split(file.relative_path)
    with _open_text(file.path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                value = {"_parse_error": f"JSONDecodeError: {exc.msg}"}
            if not isinstance(value, Mapping):
                value = {"_invalid_row": value}
            yield SourceRow(
                dataset_id=location.policy.dataset_id,
                repo=location.policy.repo,
                source_path=file.relative_path,
                source_row_id=f"{file.relative_path}#{line_number}",
                source_config=config,
                source_split=split,
                row_index=line_number - 1,
                raw=value,
            )


def _read_json(file: DataFile, location: DatasetLocation) -> Iterator[SourceRow]:
    config = infer_config(file.relative_path)
    split = infer_split(file.relative_path)
    try:
        with file.path.open("r", encoding="utf-8", errors="replace") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        value = {"_parse_error": f"JSONDecodeError: {exc}"}
    values = value if isinstance(value, list) else [value]
    for index, item in enumerate(values):
        raw = item if isinstance(item, Mapping) else {"_invalid_row": item}
        yield SourceRow(
            dataset_id=location.policy.dataset_id,
            repo=location.policy.repo,
            source_path=file.relative_path,
            source_row_id=f"{file.relative_path}#{index + 1}",
            source_config=config,
            source_split=split,
            row_index=index,
            raw=raw,
        )


def _pyarrow_parquet():
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - depends on optional env
        raise AdapterUnavailable(
            "Parquet input requires pyarrow; install the audit extra without downloading dataset files"
        ) from exc
    return parquet


def parquet_metadata(path: Path) -> tuple[int, dict[str, str], list[dict[str, Any]]]:
    parquet = _pyarrow_parquet()
    parquet_file = parquet.ParquetFile(path)
    row_count = int(parquet_file.metadata.num_rows) if parquet_file.metadata else 0
    schema: dict[str, str] = {}
    for field in parquet_file.schema_arrow:
        schema[str(field.name)] = str(field.type)
    samples: list[dict[str, Any]] = []
    for batch in parquet_file.iter_batches(batch_size=5):
        for item in batch.to_pylist():
            samples.append(safe_sample(item))
            if len(samples) >= 5:
                break
        if len(samples) >= 5:
            break
    return row_count, schema, samples


def _read_parquet(file: DataFile, location: DatasetLocation, *, batch_size: int) -> Iterator[SourceRow]:
    parquet = _pyarrow_parquet()
    parquet_file = parquet.ParquetFile(file.path)
    config = infer_config(file.relative_path)
    split = infer_split(file.relative_path)
    row_index = 0
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        for value in batch.to_pylist():
            raw = value if isinstance(value, Mapping) else {"_invalid_row": value}
            yield SourceRow(
                dataset_id=location.policy.dataset_id,
                repo=location.policy.repo,
                source_path=file.relative_path,
                source_row_id=f"{file.relative_path}#{row_index + 1}",
                source_config=config,
                source_split=split,
                row_index=row_index,
                raw=raw,
            )
            row_index += 1


def iter_raw_rows(
    location: DatasetLocation,
    *,
    batch_size: int = 512,
    max_rows: int | None = None,
) -> Iterator[SourceRow]:
    """Yield raw rows without loading a dataset into memory."""

    yielded = 0
    if not location.found:
        return
    for file in iter_data_files(location.path):
        if file.suffix in {".jsonl", ".jsonl.gz"}:
            rows = _read_jsonl(file, location)
        elif file.suffix == ".parquet":
            rows = _read_parquet(file, location, batch_size=batch_size)
        else:
            rows = _read_json(file, location)
        for row in rows:
            yield row
            yielded += 1
            if max_rows is not None and yielded >= max_rows:
                return


def _event_stream_rows(location: DatasetLocation, *, batch_size: int) -> Iterator[SourceRow]:
    """Group small Claude Code event files into one conversation unit.

    The source contains metadata and message events rather than one canonical
    row per line.  The files are intentionally small, so grouping one file at
    a time is safe and prevents the metadata events from becoming fake SFT
    examples.
    """

    if not location.found:
        return
    for file in iter_data_files(location.path):
        if file.suffix not in {".jsonl", ".jsonl.gz"}:
            continue
        events = [row.raw for row in _read_jsonl(file, location)]
        if not events:
            continue
        yield SourceRow(
            dataset_id=location.policy.dataset_id,
            repo=location.policy.repo,
            source_path=file.relative_path,
            source_row_id=file.relative_path,
            source_config=infer_config(file.relative_path),
            source_split=infer_split(file.relative_path),
            row_index=0,
            raw={"_event_stream": events},
        )


def iter_source_rows(
    location: DatasetLocation,
    *,
    batch_size: int = 512,
    max_rows: int | None = None,
    group_event_streams: bool = True,
) -> Iterator[SourceRow]:
    if group_event_streams and location.policy.dataset_id == "claude_fable_code":
        rows = _event_stream_rows(location, batch_size=batch_size)
    else:
        rows = iter_raw_rows(location, batch_size=batch_size, max_rows=max_rows)
    yielded = 0
    for row in rows:
        yield row
        yielded += 1
        if max_rows is not None and yielded >= max_rows:
            return


def inspect_location(
    location: DatasetLocation,
    *,
    sample_rows: int = 5,
    batch_size: int = 256,
) -> dict[str, Any]:
    """Inspect files using Parquet metadata and streaming JSONL parsing."""

    files = list(iter_data_files(location.path)) if location.found else []
    result: dict[str, Any] = {
        "files": [],
        "file_count": len(files),
        "total_bytes": sum(file.size_bytes for file in files),
        "file_formats": sorted({file.suffix for file in files}),
        "row_count": 0,
        "configs": [],
        "splits": [],
        "schema": {},
        "samples": [],
        "adapter_warnings": [],
    }
    schemas: dict[str, set[str]] = {}
    configs: set[str] = set()
    splits: set[str] = set()
    for file in files:
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
            row_count, schema, samples = parquet_metadata(file.path)
            file_result["rows"] = row_count
            file_result["schema"] = schema
            file_result["samples"] = samples[:sample_rows]
            result["row_count"] += row_count
            for name, type_name in schema.items():
                schemas.setdefault(name, set()).add(type_name)
            if len(result["samples"]) < sample_rows:
                result["samples"].extend(samples[: sample_rows - len(result["samples"])])
        else:
            file_count = 0
            file_schema: dict[str, set[str]] = {}
            file_samples: list[Any] = []
            for row in iter_raw_rows(location, batch_size=batch_size, max_rows=None):
                if row.source_path != file.relative_path:
                    continue
                file_count += 1
                for key, value in row.raw.items():
                    file_schema.setdefault(str(key), set()).add(type(value).__name__)
                if len(file_samples) < sample_rows:
                    file_samples.append(safe_sample(row.raw))
            file_result["rows"] = file_count
            file_result["schema"] = {key: sorted(values) for key, values in sorted(file_schema.items())}
            file_result["samples"] = file_samples
            result["row_count"] += file_count
            for name, type_names in file_schema.items():
                schemas.setdefault(name, set()).update(type_names)
            if len(result["samples"]) < sample_rows:
                result["samples"].extend(file_samples[: sample_rows - len(result["samples"])])
        result["files"].append(file_result)
    result["configs"] = sorted(configs)
    result["splits"] = sorted(splits)
    result["schema"] = {key: sorted(values) for key, values in sorted(schemas.items())}
    result["samples"] = result["samples"][:sample_rows]
    return result
