"""Overnight finalization runner.

This module intentionally performs only local preparation.  It never changes
registry status, creates an approval, starts training, provisions RunPod, or
calls an external judge.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Iterator, Mapping
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any


from ..data.adapters import SourceRow, iter_source_rows
from ..data.canonical import canonicalize_source_row
from ..data.discovery import DatasetLocation, discover_sources, load_source_policies
from ..data.quality import score_record
from .contamination import (
    ContaminationIndex,
    contamination_status,
)
from .core import (
    FinalizationBlocked,
    atomic_write_json,
    atomic_write_text,
    load_yaml,
    storage_preflight,
)
from .eval import (
    acquire_eval_sources,
    iter_eval_rows,
    load_eval_registry,
    write_eval_registry,
)
from .schema_adapters import adapt_candidate_source_row, project_eval_record
from .selection import (
    classify_nemotron_split,
    fable_length_bucket,
    is_known_nemotron_split,
    nemotron_split_conflict,
    selection_bucket,
)
from .tokenization import (
    TokenCounter,
    TokenStats,
    TokenizationUnavailable,
    load_qwen_tokenizer,
    template_family_hash,
)


STAGE_ORDER = (
    "preflight",
    "eval",
    "contamination",
    "tokenize",
    "nemotron",
    "fable",
    "quality",
    "license",
    "candidate",
    "format",
    "baseline",
    "reports",
)
TRAINING_CANDIDATE_IDS = (
    "extraction_wiki_ja",
    "magpie_sft_v1",
    "nemotron_sft_multilingual_v2",
    "math_japanese_8k",
    "ace_reason_math_japanese",
    "gpt_5_6_traces",
    "fable_5_5_distillation",
    "claude_fable_code",
    "frontier_multi_teacher",
    "fable_5_premium",
)
BALANCED_BUCKETS = {
    "general_japanese": 0.25,
    "instruction_extraction": 0.15,
    "japanese_math_reasoning": 0.10,
    "japanese_stem_code": 0.10,
    "frontier_reasoning": 0.25,
    "agent_code_retention": 0.15,
}
KNOWN_LICENSE_MARKERS = ("apache-2", "apache 2", "mit", "cc-by-4", "cc by 4")


_CONTAMINATION_WORKER_INDEX: ContaminationIndex | None = None
_CONTAMINATION_WORKER_MISSING = False
_TOKENIZATION_WORKER_COUNTER: TokenCounter | None = None
CONTAMINATION_BATCH_CHARACTER_BUDGET = 1_000_000
# A 4.9M-character real row measured 1,310,756 tokens with the pinned
# tokenizer, already 5x beyond its declared 262,144-token model limit.
MAX_TOKENIZATION_CHARACTERS = 1_000_000
TOKEN_METADATA_FIELDS = (
    "record_id",
    "source_dataset",
    "source_repo",
    "source_row_id",
    "source_path",
    "source_config",
    "source_split",
    "source_revision",
    "category",
    "selection_bucket",
    "nemotron_bucket",
    "fable_bucket",
    "language",
    "language_user",
    "language_assistant",
    "language_reasoning",
    "teacher_model",
    "task_type",
    "prompt_length_bin",
    "answer_length_bin",
    "template_family_hash",
    "reasoning_sft_ready",
    "tool_sft_ready",
    "content_hash",
    "quality_score",
    "raw_character_count",
)


def _canonicalize_token_payload(payload: Any) -> dict[str, Any]:
    if (
        isinstance(payload, tuple)
        and len(payload) == 2
        and isinstance(payload[0], DatasetLocation)
        and isinstance(payload[1], SourceRow)
    ):
        return _canonical(payload[0], payload[1])
    return payload


def _compact_token_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Drop raw conversation text before crossing the worker IPC boundary."""

    return {key: record.get(key) for key in TOKEN_METADATA_FIELDS}


def _nested_text_char_count(value: Any) -> int:
    if isinstance(value, str):
        return len(value)
    if isinstance(value, Mapping):
        return sum(_nested_text_char_count(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_nested_text_char_count(item) for item in value)
    return 0


def _oversized_token_result(
    payload: Any,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    if not (
        isinstance(payload, tuple)
        and len(payload) == 2
        and isinstance(payload[0], DatasetLocation)
        and isinstance(payload[1], SourceRow)
    ):
        return None
    location, source_row = payload
    raw_characters = _nested_text_char_count(source_row.raw)
    if raw_characters <= MAX_TOKENIZATION_CHARACTERS:
        return None
    record = {
        "record_id": hashlib.sha256(
            f"{source_row.dataset_id}:{source_row.source_row_id}".encode("utf-8")
        ).hexdigest(),
        "source_dataset": location.policy.dataset_id,
        "source_repo": location.policy.repo,
        "source_row_id": source_row.source_row_id,
        "source_path": source_row.source_path,
        "source_config": source_row.source_config,
        "source_split": source_row.source_split,
        "source_revision": location.revision,
        "category": location.policy.category,
        "selection_bucket": selection_bucket(
            {
                "source_dataset": location.policy.dataset_id,
                "category": location.policy.category,
            }
        ),
        "task_type": location.policy.category,
        "prompt_length_bin": ">16384",
        "answer_length_bin": ">16384",
        "quality_score": 0.0,
        "reasoning_sft_ready": False,
        "tool_sft_ready": False,
        "raw_character_count": raw_characters,
    }
    counts = {
        "raw_content_tokens": None,
        "training_formatted_tokens": None,
        "supervised_assistant_tokens": None,
        "assistant_mask_status": "unavailable_oversized_record",
    }
    return _compact_token_record(record), counts


def _init_contamination_worker(index: ContaminationIndex, missing_required_eval: bool) -> None:
    global _CONTAMINATION_WORKER_INDEX, _CONTAMINATION_WORKER_MISSING
    _CONTAMINATION_WORKER_INDEX = index
    _CONTAMINATION_WORKER_MISSING = missing_required_eval


def _scan_contamination_batch(
    batch: list[Any],
) -> list[tuple[dict[str, Any], str, list[str], str]]:
    """Canonicalize and fingerprint one batch while preserving source order."""

    if _CONTAMINATION_WORKER_INDEX is None:
        raise RuntimeError("contamination worker was not initialized")
    rows: list[tuple[dict[str, Any], str, list[str], str]] = []
    for payload in batch:
        if len(payload) == 2 and isinstance(payload[0], DatasetLocation):
            location, source_row = payload
            record = _contamination_record(location, source_row)
            source_dataset = location.policy.dataset_id
            source_path = source_row.source_path
            source_row_id = source_row.source_row_id
        else:
            record, source_dataset, source_path, source_row_id = payload
        schema_adapter = str(
            record.get("contamination_schema_adapter") or "unregistered_candidate_schema"
        )
        scan_disposition = str(record.get("contamination_scan_disposition") or "scan")
        if scan_disposition != "scan":
            result = {
                "status": scan_disposition,
                "similarity": 0.0,
                "matches": [],
                "reasons": [
                    str(record.get("contamination_scan_reason") or scan_disposition),
                    f"schema_adapter:{schema_adapter}",
                ],
                "content_hash": "",
                "prompt_hash": "",
            }
        else:
            result = contamination_status(
                record,
                _CONTAMINATION_WORKER_INDEX,
                missing_required_eval=_CONTAMINATION_WORKER_MISSING,
            )
            result.setdefault("reasons", []).append(f"schema_adapter:{schema_adapter}")
        status = str(result.get("status", "not_checked_missing_eval_source"))
        reason_list = [str(reason) for reason in result.get("reasons", [])]
        effective_disposition = (
            status if status.startswith("excluded_") else scan_disposition
        )
        rows.append(
            (
                {
                    "record_id": record.get("record_id"),
                    "source_dataset": source_dataset,
                    "source_row_id": source_row_id,
                    "content_hash": result.get("content_hash"),
                    "prompt_hash": result.get("prompt_hash"),
                    "status": status,
                    "similarity": float(result.get("similarity") or 0.0),
                    "reasons": ";".join(reason_list),
                    "schema_adapter": schema_adapter,
                    "scan_disposition": effective_disposition,
                    "matched_eval_ids": ";".join(
                        str(item.get("id"))
                        for item in result.get("matches", [])
                        if isinstance(item, Mapping)
                    ),
                },
                status,
                reason_list,
                source_path,
            )
        )
    return rows


def _batched_contamination_payloads(
    payloads: Iterable[Any],
    batch_size: int,
    *,
    max_characters: int = CONTAMINATION_BATCH_CHARACTER_BUDGET,
) -> Iterator[list[Any]]:
    batch: list[Any] = []
    batch_characters = 0
    for payload in payloads:
        payload_characters = 0
        if (
            isinstance(payload, tuple)
            and len(payload) == 2
            and isinstance(payload[1], SourceRow)
        ):
            payload_characters = _nested_text_char_count(payload[1].raw)
        elif isinstance(payload, Mapping):
            payload_characters = _nested_text_char_count(payload)
        if batch and (
            len(batch) >= batch_size
            or batch_characters + payload_characters > max_characters
        ):
            yield batch
            batch = []
            batch_characters = 0
        batch.append(payload)
        batch_characters += payload_characters
        if len(batch) >= batch_size or batch_characters >= max_characters:
            yield batch
            batch = []
            batch_characters = 0
    if batch:
        yield batch


def _ordered_contamination_rows(
    payloads: Iterable[Any],
    index: ContaminationIndex,
    missing_required_eval: bool,
    *,
    workers: int,
    batch_size: int = 1,
    max_batch_characters: int = CONTAMINATION_BATCH_CHARACTER_BUDGET,
) -> Iterator[tuple[dict[str, Any], str, list[str], str]]:
    """Bounded, deterministic process map for CPU-heavy fingerprints."""

    worker_count = max(1, int(workers))
    batches = _batched_contamination_payloads(
        payloads,
        max(1, int(batch_size)),
        max_characters=max(1, int(max_batch_characters)),
    )
    if worker_count == 1:
        _init_contamination_worker(index, missing_required_eval)
        for batch in batches:
            yield from _scan_contamination_batch(batch)
        return

    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=_init_contamination_worker,
        initargs=(index, missing_required_eval),
    ) as executor:
        pending: deque[Any] = deque()
        max_pending = worker_count
        for batch in batches:
            pending.append(executor.submit(_scan_contamination_batch, batch))
            if len(pending) >= max_pending:
                yield from pending.popleft().result()
        while pending:
            yield from pending.popleft().result()


def _contamination_worker_count() -> int:
    available = max(1, os.cpu_count() or 1)
    default = min(12, max(1, available // 2))
    configured = os.environ.get("ZIPANGU_CONTAMINATION_WORKERS", str(default))
    try:
        requested = int(configured)
    except ValueError as exc:
        raise FinalizationBlocked("ZIPANGU_CONTAMINATION_WORKERS must be an integer") from exc
    return max(1, min(requested, available))


def _init_tokenization_worker(
    cache_root: str,
    model_root: str,
    model_id: str,
    revision: str,
) -> None:
    global _TOKENIZATION_WORKER_COUNTER
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer, loaded_revision = load_qwen_tokenizer(
        cache_root=cache_root,
        model_root=model_root,
        model_id=model_id,
        configured_revision=revision,
    )
    if loaded_revision != revision:
        raise TokenizationUnavailable("tokenizer worker revision mismatch")
    _TOKENIZATION_WORKER_COUNTER = TokenCounter(tokenizer)


def _count_token_batch(
    batch: list[Any],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if _TOKENIZATION_WORKER_COUNTER is None:
        raise RuntimeError("tokenization worker was not initialized")
    records = [_canonicalize_token_payload(payload) for payload in batch]
    counts = _TOKENIZATION_WORKER_COUNTER.count_records(records)
    return [
        (_compact_token_record(record), count)
        for record, count in zip(records, counts, strict=True)
    ]


def _ordered_tokenized_records(
    records: Iterable[Any],
    *,
    cache_root: Path,
    model_root: Path,
    model_id: str,
    revision: str,
    workers: int,
    batch_size: int = 4,
    sequential_counter: TokenCounter | None = None,
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Bounded process map for tokenizer calls with stable input order."""

    worker_count = max(1, int(workers))
    safe_batch_size = max(1, int(batch_size))
    if worker_count == 1:
        counter = sequential_counter
        if counter is None:
            tokenizer, loaded_revision = load_qwen_tokenizer(
                cache_root=cache_root,
                model_root=model_root,
                model_id=model_id,
                configured_revision=revision,
            )
            if loaded_revision != revision:
                raise TokenizationUnavailable("tokenizer revision mismatch")
            counter = TokenCounter(tokenizer)
        batch: list[Any] = []

        def count_batch(values: list[Any]) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
            canonical_batch = [_canonicalize_token_payload(payload) for payload in values]
            counts = counter.count_records(canonical_batch)
            yield from (
                (_compact_token_record(record), count)
                for record, count in zip(canonical_batch, counts, strict=True)
            )

        for payload in records:
            oversized = _oversized_token_result(payload)
            if oversized is not None:
                if batch:
                    yield from count_batch(batch)
                    batch = []
                yield oversized
                continue
            batch.append(payload)
            if len(batch) >= safe_batch_size:
                yield from count_batch(batch)
                batch = []
        if batch:
            yield from count_batch(batch)
        return

    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=_init_tokenization_worker,
        initargs=(str(cache_root), str(model_root), model_id, revision),
    ) as executor:
        pending: deque[Any] = deque()
        max_pending = worker_count
        batch: list[Any] = []

        def submit_batch() -> None:
            nonlocal batch
            if batch:
                pending.append(executor.submit(_count_token_batch, batch))
                batch = []

        for payload in records:
            oversized = _oversized_token_result(payload)
            if oversized is not None:
                submit_batch()
                while pending:
                    yield from pending.popleft().result()
                yield oversized
                continue
            batch.append(payload)
            if len(batch) >= safe_batch_size:
                submit_batch()
            if len(pending) >= max_pending:
                yield from pending.popleft().result()
        submit_batch()
        while pending:
            yield from pending.popleft().result()


def _tokenization_worker_count() -> int:
    available = max(1, os.cpu_count() or 1)
    # Canonicalization and quality scoring are Python-heavy, so distribute the
    # complete source-row pipeline instead of feeding a tokenizer from one core.
    default = min(8, max(1, available // 2))
    configured = os.environ.get("ZIPANGU_TOKENIZATION_WORKERS", str(default))
    try:
        requested = int(configured)
    except ValueError as exc:
        raise FinalizationBlocked("ZIPANGU_TOKENIZATION_WORKERS must be an integer") from exc
    return max(1, min(requested, available))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return default


def _registry_items(repo_root: Path) -> dict[str, dict[str, Any]]:
    payload = load_yaml(repo_root / "configs" / "datasets" / "registry.yaml")
    items = payload.get("datasets")
    return {
        str(item.get("id")): dict(item)
        for item in items or []
        if isinstance(item, Mapping) and item.get("id")
    }


def _locations(repo_root: Path, dataset_root: Path) -> tuple[Path | None, list[DatasetLocation]]:
    policy_path, policies, policy_payload = load_source_policies(
        repo_root / "configs" / "datasets" / "source_policies.yaml"
    )
    del policy_path
    return discover_sources(
        policies,
        repo_root,
        explicit_root=dataset_root,
        policy_payload=policy_payload,
    )


def _iter_locations(
    repo_root: Path,
    dataset_root: Path,
    *,
    dataset_ids: Iterable[str] = TRAINING_CANDIDATE_IDS,
    after_dataset_id: str | None = None,
    after_source_row_id: str | None = None,
    batch_size: int = 512,
) -> Iterator[tuple[DatasetLocation, SourceRow]]:
    wanted = set(dataset_ids)
    _, locations = _locations(repo_root, dataset_root)
    cursor_found = after_dataset_id is None
    for location in locations:
        if location.policy.dataset_id not in wanted or not location.found:
            continue
        if not cursor_found:
            if location.policy.dataset_id != after_dataset_id:
                continue
            cursor_found = True
            location_cursor = after_source_row_id
        else:
            location_cursor = None
        for row in iter_source_rows(
            location,
            batch_size=batch_size,
            group_event_streams=True,
            after_source_row_id=location_cursor,
        ):
            yield location, row
    if after_dataset_id is not None and not cursor_found:
        raise FinalizationBlocked(f"dataset resume cursor was not found: {after_dataset_id}")


def _canonical(location: DatasetLocation, row: SourceRow) -> dict[str, Any]:
    record = canonicalize_source_row(
        row,
        source_metadata={
            "license": location.card_metadata.get("license", location.policy.license_expected),
            "source_revision": location.revision,
            "category": location.policy.category,
        },
    )
    record["source_revision"] = location.revision
    record["category"] = location.policy.category
    record["vendor_specific"] = location.policy.vendor_specific
    record["source_dataset"] = location.policy.dataset_id
    record["source_repo"] = location.policy.repo
    record["selection_bucket"] = selection_bucket(record)
    record["template_family_hash"] = template_family_hash(record)
    record["task_type"] = location.policy.category
    record["language"] = record.get("language_overall", "unknown")
    record["prompt_length_bin"] = _length_bin(len(str(record.get("user", ""))))
    record["answer_length_bin"] = _length_bin(len(str(record.get("assistant_final", ""))))
    record.update(score_record(record))
    return record


def _contamination_record(location: DatasetLocation, row: SourceRow) -> dict[str, Any]:
    """Canonical fields needed by fingerprinting, without quality side work."""

    adaptation = adapt_candidate_source_row(row)
    if adaptation.disposition != "scan":
        return {
            "record_id": hashlib.sha256(
                f"{row.dataset_id}:{row.source_row_id}".encode("utf-8")
            ).hexdigest(),
            "source_dataset": location.policy.dataset_id,
            "source_repo": location.policy.repo,
            "source_row_id": row.source_row_id,
            "source_path": row.source_path,
            "messages": [],
            "user": "",
            "assistant_final": "",
            "contamination_schema_adapter": adaptation.adapter_id,
            "contamination_scan_disposition": adaptation.disposition,
            "contamination_scan_reason": adaptation.reason,
        }
    record = canonicalize_source_row(
        adaptation.source_row,
        source_metadata={
            "license": location.card_metadata.get(
                "license", location.policy.license_expected
            ),
            "source_revision": location.revision,
            "category": location.policy.category,
        },
    )
    record["source_dataset"] = location.policy.dataset_id
    record["source_repo"] = location.policy.repo
    record["contamination_schema_adapter"] = adaptation.adapter_id
    record["contamination_scan_disposition"] = adaptation.disposition
    record["contamination_scan_reason"] = adaptation.reason
    return record


def _length_bin(length: int) -> str:
    if length <= 256:
        return "<=256"
    if length <= 1_024:
        return "257-1024"
    if length <= 4_096:
        return "1025-4096"
    if length <= 16_384:
        return "4097-16384"
    return ">16384"


def _parquet_writer(path: Path, rows: Iterable[Mapping[str, Any]], *, batch_size: int = 2_000) -> int:
    try:
        import pyarrow as pa
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise FinalizationBlocked("pyarrow is required for metadata parquet artifacts") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    count = 0
    batch: list[dict[str, Any]] = []
    try:
        for row in rows:
            batch.append(dict(row))
            if len(batch) < batch_size:
                continue
            table = pa.Table.from_pylist(batch)
            if writer is None:
                writer = parquet.ParquetWriter(path, table.schema, compression="zstd")
            writer.write_table(table)
            count += len(batch)
            batch = []
        if batch:
            table = pa.Table.from_pylist(batch)
            if writer is None:
                writer = parquet.ParquetWriter(path, table.schema, compression="zstd")
            writer.write_table(table)
            count += len(batch)
    finally:
        if writer is not None:
            writer.close()
    return count


def _iter_parquet(path: Path, *, start_row: int = 0) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise FinalizationBlocked("pyarrow is required to read candidate metadata") from exc
    parquet_file = parquet.ParquetFile(path)
    row_offset = 0
    for group_index in range(parquet_file.num_row_groups):
        group_rows = int(parquet_file.metadata.row_group(group_index).num_rows)
        if row_offset + group_rows <= start_row:
            row_offset += group_rows
            continue
        for batch in parquet_file.iter_batches(
            batch_size=2_000,
            row_groups=[group_index],
        ):
            for item in batch.to_pylist():
                if row_offset < start_row:
                    row_offset += 1
                    continue
                row_offset += 1
                if isinstance(item, Mapping):
                    yield dict(item)


def _contamination_resume_info(
    stage: Mapping[str, Any],
    partial_path: Path,
) -> tuple[int, str, str]:
    """Validate a closed partial parquet against the latest checkpoint."""

    checkpoints = stage.get("resume_from")
    if not isinstance(checkpoints, list) or not checkpoints or not partial_path.is_file():
        return 0, "", ""
    latest = checkpoints[-1]
    if not isinstance(latest, Mapping):
        return 0, "", ""
    checkpoint_rows = int(latest.get("rows_scanned") or 0)
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise FinalizationBlocked("pyarrow is required to resume contamination") from exc
    try:
        parquet_file = parquet.ParquetFile(partial_path)
        actual_rows = int(parquet_file.metadata.num_rows)
        if actual_rows <= 0 or parquet_file.num_row_groups <= 0:
            raise FinalizationBlocked("contamination resume partial is empty")
        last_group = parquet_file.read_row_group(
            parquet_file.num_row_groups - 1,
            columns=["source_dataset", "source_row_id"],
        )
        last_dataset = str(last_group.column("source_dataset")[-1].as_py() or "")
        last_row_id = str(last_group.column("source_row_id")[-1].as_py() or "")
    except FinalizationBlocked:
        raise
    except Exception as exc:
        raise FinalizationBlocked("contamination resume partial is unreadable") from exc
    if actual_rows < checkpoint_rows or actual_rows - checkpoint_rows >= 10_000:
        raise FinalizationBlocked(
            "contamination resume partial/checkpoint mismatch: "
            f"partial={actual_rows}, checkpoint={checkpoint_rows}"
        )
    return actual_rows, last_dataset, last_row_id


def _contamination_parts_dir(result_path: Path) -> Path:
    return result_path.with_name(f".{result_path.name}.parts")


def _prepare_contamination_parts(parts_dir: Path, *, resume: bool) -> None:
    """Create a fresh parts directory or fail before an implicit resume."""

    if parts_dir.is_dir() and not resume:
        try:
            next(parts_dir.iterdir())
        except StopIteration:
            pass
        else:
            raise FinalizationBlocked(
                "fresh contamination scan refuses existing chunks; "
                "archive them or rerun explicitly with --resume"
            )
    parts_dir.mkdir(parents=True, exist_ok=True)


def _contamination_parts_resume_info(
    stage: Mapping[str, Any],
    parts_dir: Path,
) -> tuple[int, str, str]:
    """Validate durable closed parquet chunks and return their exact tail."""

    if not parts_dir.is_dir():
        return 0, "", ""
    for temporary in parts_dir.glob("*.tmp"):
        temporary.unlink(missing_ok=True)
    files = sorted(parts_dir.glob("part-*.parquet"), key=lambda path: path.name)
    if not files:
        return 0, "", ""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise FinalizationBlocked("pyarrow is required to resume contamination chunks") from exc
    total = 0
    last_dataset = ""
    last_row_id = ""
    for file_index, path in enumerate(files):
        pieces = path.stem.split("-")
        if len(pieces) != 3 or pieces[0] != "part":
            raise FinalizationBlocked(f"invalid contamination chunk name: {path.name}")
        try:
            start, end = int(pieces[1]), int(pieces[2])
        except ValueError as exc:
            raise FinalizationBlocked(f"invalid contamination chunk range: {path.name}") from exc
        if start != total or end <= start or end - start > 10_000:
            raise FinalizationBlocked(f"non-contiguous contamination chunk: {path.name}")
        try:
            parquet_file = pq.ParquetFile(path)
            rows = int(parquet_file.metadata.num_rows)
            if rows != end - start or parquet_file.num_row_groups <= 0:
                raise FinalizationBlocked(f"contamination chunk row mismatch: {path.name}")
            last_group = parquet_file.read_row_group(
                parquet_file.num_row_groups - 1,
                columns=["source_dataset", "source_row_id"],
            )
            last_dataset = str(last_group.column("source_dataset")[-1].as_py() or "")
            last_row_id = str(last_group.column("source_row_id")[-1].as_py() or "")
        except FinalizationBlocked:
            raise
        except Exception as exc:
            if file_index == len(files) - 1:
                path.unlink(missing_ok=True)
                break
            raise FinalizationBlocked(f"unreadable contamination chunk: {path.name}") from exc
        total = end
    checkpoints = stage.get("resume_from") or stage.get("checkpoints") or []
    checkpoint_rows = max(
        (int(item.get("rows_scanned") or 0) for item in checkpoints if isinstance(item, Mapping)),
        default=0,
    )
    if checkpoint_rows and total + 10_000 < checkpoint_rows:
        raise FinalizationBlocked(
            "contamination chunk/checkpoint mismatch: "
            f"chunks={total}, checkpoint={checkpoint_rows}"
        )
    return total, last_dataset, last_row_id


def _iter_contamination_parts(parts_dir: Path) -> Iterator[dict[str, Any]]:
    for path in sorted(parts_dir.glob("part-*.parquet"), key=lambda item: item.name):
        yield from _iter_parquet(path)


def _token_rows_path(temp_root: Path) -> Path:
    return temp_root / "finalization" / "token_rows.jsonl"


def _load_token_rows(temp_root: Path) -> Iterator[dict[str, Any]]:
    path = _token_rows_path(temp_root)
    if not path.is_file():
        raise FinalizationBlocked("tokenize stage metadata is missing")
    from .core import iter_jsonl

    yield from iter_jsonl(path)


def _contamination_map(repo_root: Path) -> dict[str, dict[str, Any]]:
    path = repo_root / "data" / "manifests" / "contamination_results.parquet"
    if not path.is_file():
        return {}
    result: dict[str, dict[str, Any]] = {}
    try:
        for row in _iter_parquet(path):
            identifier = str(row.get("record_id", ""))
            if identifier:
                result[identifier] = row
    except Exception:
        # An interrupted parquet writer must never be treated as a clean scan.
        return {}
    return result


def _iter_contamination_results(
    repo_root: Path,
    *,
    start_row: int = 0,
) -> Iterator[dict[str, Any]]:
    path = repo_root / "data" / "manifests" / "contamination_results.parquet"
    if not path.is_file():
        return
    try:
        yield from _iter_parquet(path, start_row=start_row)
    except Exception:
        # A missing/corrupt result stream is fail-closed for tokenization.
        return


def _available_eval_entries(repo_root: Path) -> list[dict[str, Any]]:
    path = repo_root / "data" / "manifests" / "eval_registry.json"
    payload = load_eval_registry(path)
    values = payload.get("evaluations")
    return [dict(value) for value in values or [] if isinstance(value, Mapping)]


def _required_eval_missing(entries: Iterable[Mapping[str, Any]]) -> bool:
    return any(
        bool(entry.get("required", False))
        and entry.get("availability") != "available"
        for entry in entries
    )


def _report_dir(repo_root: Path) -> Path:
    path = repo_root / "reports" / "pretrain_finalization"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_stage_report(repo_root: Path, name: str, lines: Iterable[str]) -> Path:
    path = _report_dir(repo_root) / name
    atomic_write_text(path, "\n".join(lines).rstrip() + "\n")
    return path


def _stage_preflight(runner: Any) -> dict[str, Any]:
    runner.temp_root.mkdir(parents=True, exist_ok=True)
    runner.cache_root.mkdir(parents=True, exist_ok=True)
    runner.model_root.mkdir(parents=True, exist_ok=True)
    runner.eval_root.mkdir(parents=True, exist_ok=True)
    if not runner.dataset_root.is_dir():
        raise FinalizationBlocked(f"dataset root is missing: {runner.dataset_root}")
    disk = {
        "dataset": storage_preflight(runner.dataset_root),
        "eval": storage_preflight(runner.eval_root, reserve_bytes=runner.config["disk_safety"]["reserve_bytes"]),
        "cache": storage_preflight(runner.cache_root, reserve_bytes=runner.config["disk_safety"]["reserve_bytes"]),
        "model": storage_preflight(runner.model_root, reserve_bytes=runner.config["disk_safety"]["reserve_bytes"]),
        "temp": storage_preflight(runner.temp_root, reserve_bytes=runner.config["disk_safety"]["reserve_bytes"]),
    }
    atomic_write_json(runner.temp_root / "finalization" / "preflight.json", disk)
    if any(not item["allowed"] for item in disk.values()):
        raise FinalizationBlocked("disk reserve preflight failed")
    return {"status": "PASS", "artifacts": [runner.temp_root / "finalization" / "preflight.json"], "details": {"model_id": runner.model_id, "base_model_id": runner.base_model_id}}


def _stage_eval(runner: Any) -> dict[str, Any]:
    entries = acquire_eval_sources(runner.eval_root)
    registry_path = write_eval_registry(runner.repo_root, entries)
    missing = [
        str(item["id"])
        for item in entries
        if item.get("required") and item.get("availability") != "available"
    ]
    lines = [
        "# Evaluation Corpus Status",
        "",
        "Raw evaluation data is stored only under the external eval root. This report contains metadata and availability; it does not contain evaluation prompts.",
        "",
        "| ID | Version | Split | Rows | Availability | Contamination ready | Required |",
        "| --- | --- | --- | ---: | --- | --- | --- |",
    ]
    for item in entries:
        lines.append(
            "| `{id}` | `{version}` | `{split}` | {rows:,} | `{availability}` | `{ready}` | `{required}` |".format(
                id=item.get("id"),
                version=item.get("version"),
                split=item.get("split"),
                rows=int(item.get("row_count") or 0),
                availability=item.get("availability"),
                ready=bool(item.get("contamination_scan_ready")),
                required=bool(item.get("required")),
            )
        )
    lines.extend(
        [
            "",
            f"- required sources missing or blocked: `{', '.join(missing) or 'none'}`",
            "- AnswerCarefully access is never bypassed; a gated or denied source remains blocked.",
            "- training_forbidden is true for every entry.",
        ]
    )
    report = _write_stage_report(runner.repo_root, "EVAL_CORPUS_STATUS.md", lines)
    status = "BLOCKED" if missing else "PASS"
    return {
        "status": status,
        "artifacts": [registry_path, report],
        "blocked_reasons": [f"required_eval_missing:{item}" for item in missing],
        "details": {"required_missing": missing, "entries": entries},
    }


def _build_eval_index(
    entries: Iterable[Mapping[str, Any]],
) -> tuple[ContaminationIndex, int, dict[str, int]]:
    index = ContaminationIndex()
    count = 0
    adapter_counts: Counter[str] = Counter()
    for entry in entries:
        if entry.get("availability") != "available":
            continue
        local_path = entry.get("local_path")
        if not local_path:
            continue
        source_path = Path(str(local_path))
        for relative, row_index, row in iter_eval_rows(source_path, split_hint=str(entry.get("split", ""))):
            identifier = f"{entry['id']}:{relative}#{row_index}"
            projection = project_eval_record(str(entry["id"]), row)
            index.add(identifier, projection.content, prompt=projection.prompt)
            adapter_counts[projection.adapter_id] += 1
            count += 1
    return index, count, dict(sorted(adapter_counts.items()))


def _stage_contamination(runner: Any) -> dict[str, Any]:
    policy = runner.config.get("contamination", {})
    policy = dict(policy) if isinstance(policy, Mapping) else {}
    entries = _available_eval_entries(runner.repo_root)
    index, eval_rows, eval_adapter_counts = _build_eval_index(entries)
    index_path = runner.eval_root / "_indexes" / "contamination_index.json"
    index.write(index_path)
    missing = _required_eval_missing(entries)
    result_path = runner.repo_root / "data" / "manifests" / "contamination_results.parquet"
    parts_dir = _contamination_parts_dir(result_path)
    _prepare_contamination_parts(parts_dir, resume=bool(runner.resume))
    resume_rows, resume_last_dataset, resume_last_row_id = _contamination_parts_resume_info(
        runner.runtime.stage("contamination"),
        parts_dir,
    )
    counts: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    last_dataset = resume_last_dataset
    last_file = None
    scanned = resume_rows
    workers = _contamination_worker_count()

    if resume_rows:
        copied = 0
        for row in _iter_contamination_parts(parts_dir):
            status = str(row.get("status", "not_checked_missing_eval_source"))
            counts[status] += 1
            for reason in str(row.get("reasons") or "").split(";"):
                if reason:
                    reasons[reason] += 1
            copied += 1
        if copied != resume_rows:
            raise FinalizationBlocked(
                f"contamination chunk copy mismatch: expected={resume_rows}, copied={copied}"
            )

    def payloads() -> Iterator[tuple[DatasetLocation, SourceRow]]:
        skipped = 0
        skipped_last_dataset = ""
        skipped_last_row_id = ""
        for location, source_row in _iter_locations(runner.repo_root, runner.dataset_root):
            if skipped < resume_rows:
                skipped += 1
                skipped_last_dataset = location.policy.dataset_id
                skipped_last_row_id = source_row.source_row_id
                continue
            if resume_rows and (
                skipped_last_dataset != resume_last_dataset
                or skipped_last_row_id != resume_last_row_id
            ):
                raise FinalizationBlocked(
                    "contamination resume source order no longer matches partial output"
                )
            yield location, source_row

    pending_rows: list[dict[str, Any]] = []
    pending_meta: list[tuple[str, list[str], str]] = []

    def commit_chunk() -> None:
        nonlocal last_dataset, last_file, scanned
        if not pending_rows:
            return
        start = scanned
        end = start + len(pending_rows)
        chunk_path = parts_dir / f"part-{start:012d}-{end:012d}.parquet"
        temporary = parts_dir / f".{chunk_path.name}.tmp"
        temporary.unlink(missing_ok=True)
        written = _parquet_writer(temporary, iter(pending_rows))
        if written != len(pending_rows):
            raise FinalizationBlocked("contamination chunk write count mismatch")
        os.replace(temporary, chunk_path)
        for row, (status, reason_list, source_path) in zip(
            pending_rows, pending_meta, strict=True
        ):
            counts[status] += 1
            for reason in reason_list:
                reasons[reason] += 1
            last_dataset = str(row.get("source_dataset") or "")
            last_file = source_path
        scanned = end
        runner.runtime.checkpoint(
            "contamination",
            {
                "rows_scanned": scanned,
                "source_dataset": last_dataset,
                "source_path": last_file,
                "source_row_id": pending_rows[-1].get("source_row_id"),
            },
        )
        pending_rows.clear()
        pending_meta.clear()

    for row, status, reason_list, source_path in _ordered_contamination_rows(
        payloads(),
        index,
        missing,
        workers=workers,
        batch_size=64,
    ):
        pending_rows.append(row)
        pending_meta.append((status, reason_list, source_path))
        if len(pending_rows) >= 10_000:
            commit_chunk()
    commit_chunk()

    candidate_rows = scanned
    finalizing_path = result_path.with_name(f".{result_path.name}.finalizing")
    finalizing_path.unlink(missing_ok=True)
    finalized_rows = _parquet_writer(
        finalizing_path,
        _iter_contamination_parts(parts_dir),
    )
    if finalized_rows != candidate_rows:
        raise FinalizationBlocked(
            f"contamination finalization count mismatch: expected={candidate_rows}, actual={finalized_rows}"
        )
    os.replace(finalizing_path, result_path)
    for part_path in parts_dir.glob("part-*.parquet"):
        part_path.unlink(missing_ok=True)
    try:
        parts_dir.rmdir()
    except OSError:
        pass
    excluded_counts = {
        key: value for key, value in counts.items() if key.startswith("excluded_")
    }
    excluded_rows = sum(excluded_counts.values())
    fingerprinted_rows = candidate_rows - excluded_rows
    review_required_rows = sum(
        value
        for key, value in counts.items()
        if key
        in {
            "quarantine_exact",
            "quarantine_near",
            "requires_manual_review",
            "excluded_empty_projection",
            "excluded_unregistered_schema",
        }
    )
    summary = {
        "status": (
            "not_checked_missing_eval_source"
            if missing
            else "requires_manual_review" if review_required_rows else "clear"
        ),
        "eval_rows_indexed": eval_rows,
        "candidate_rows_scanned": candidate_rows,
        "candidate_rows_fingerprinted": fingerprinted_rows,
        "candidate_rows_excluded": excluded_rows,
        "excluded_counts": dict(sorted(excluded_counts.items())),
        "review_required_rows": review_required_rows,
        "counts": dict(counts),
        "reasons": dict(reasons),
        "missing_required_eval": missing,
        "algorithm": "exact+prompt+long-substring+calibrated-13gram-minhash-lsh-simhash-v2",
        "empty_fingerprints_allowed": False,
        "policy": policy,
        "eval_schema_adapters": eval_adapter_counts,
        "workers": workers,
        "resumed_rows": resume_rows,
    }
    summary_path = runner.temp_root / "finalization" / "contamination_summary.json"
    atomic_write_json(summary_path, summary)
    report = _write_stage_report(
        runner.repo_root,
        "CONTAMINATION_REPORT.md",
        [
            "# Contamination Report",
            "",
            "The report is metadata-only. Candidate text is never written to the repository.",
            "",
            f"- global status: `{summary['status']}`",
            f"- evaluation rows indexed: `{eval_rows:,}`",
            f"- candidate rows scanned: `{candidate_rows:,}`",
            f"- candidate rows fingerprinted: `{fingerprinted_rows:,}`",
            f"- metadata/derivative/empty rows excluded: `{excluded_rows:,}`",
            f"- required eval missing: `{missing}`",
            f"- classifications: `{json.dumps(dict(counts), ensure_ascii=False, sort_keys=True)}`",
            "- comparison strategy: inverted indexes and LSH; O(N²) pairwise comparison is not used.",
        ],
    )
    return {
        "status": "BLOCKED" if missing else "PASS",
        "artifacts": [index_path, result_path, summary_path, report],
        "blocked_reasons": ["required_eval_source_missing"] if missing else [],
        "details": summary,
    }


def _stage_tokenize(runner: Any) -> dict[str, Any]:
    token_config = runner.config.get("tokenizer", {})
    try:
        tokenizer, revision = load_qwen_tokenizer(
            cache_root=runner.cache_root,
            model_root=runner.model_root,
            model_id=runner.base_model_id,
            configured_revision=token_config.get("revision") or runner.tokenizer_revision,
        )
    except TokenizationUnavailable as exc:
        raise FinalizationBlocked(str(exc)) from exc
    counter = TokenCounter(tokenizer)
    workers = _tokenization_worker_count()
    stats: dict[tuple[str, str], TokenStats] = defaultdict(TokenStats)
    output_path = _token_rows_path(runner.temp_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    token_stage = runner.runtime.stage("tokenize")
    resume_checkpoints = token_stage.get("resume_from") or []
    resume_checkpoint_rows = max(
        (
            int(item.get("rows_scanned") or 0)
            for item in resume_checkpoints
            if isinstance(item, Mapping)
        ),
        default=0,
    )
    rows_scanned = 0
    resume_dataset_id = ""
    resume_source_row_id = ""
    mask_unavailable = 0
    template_unavailable = 0
    oversized_unavailable = 0
    max_oversized_characters = 0
    unknown_nemotron = 0
    if resume_checkpoint_rows and output_path.is_file():
        if output_path.stat().st_size:
            with output_path.open("rb") as tail_check:
                tail_check.seek(-1, os.SEEK_END)
                if tail_check.read(1) != b"\n":
                    raise FinalizationBlocked("token metadata resume file has an incomplete final line")
        with output_path.open("r", encoding="utf-8") as existing:
            for line_number, line in enumerate(existing, start=1):
                if not line.strip():
                    raise FinalizationBlocked(
                        f"token metadata resume file has a blank line at {line_number}"
                    )
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise FinalizationBlocked(
                        f"token metadata resume file is invalid at line {line_number}"
                    ) from exc
                if not isinstance(record, Mapping):
                    raise FinalizationBlocked(
                        f"token metadata resume row is not an object at line {line_number}"
                    )
                rows_scanned += 1
                resume_dataset_id = str(record.get("source_dataset", ""))
                resume_source_row_id = str(record.get("source_row_id", ""))
                if record.get("supervised_assistant_tokens") is None:
                    mask_unavailable += 1
                if record.get("assistant_mask_status") == "unavailable_chat_template":
                    template_unavailable += 1
                if record.get("assistant_mask_status") == "unavailable_oversized_record":
                    oversized_unavailable += 1
                    max_oversized_characters = max(
                        max_oversized_characters,
                        int(record.get("raw_character_count") or 0),
                    )
                source_dataset = str(record.get("source_dataset", ""))
                if source_dataset == "nemotron_sft_multilingual_v2":
                    source_path = str(record.get("source_path", ""))
                    source_split = str(record.get("source_split", ""))
                    if not is_known_nemotron_split(source_path, source_split) or nemotron_split_conflict(
                        source_path, source_split
                    ):
                        unknown_nemotron += 1
                stats[(source_dataset, str(record.get("category", "unknown")))].add(record)
        if rows_scanned < resume_checkpoint_rows or rows_scanned - resume_checkpoint_rows >= 10_000:
            raise FinalizationBlocked(
                "token metadata resume/checkpoint mismatch: "
                f"rows={rows_scanned}, checkpoint={resume_checkpoint_rows}"
            )
    contamination_path = runner.repo_root / "data" / "manifests" / "contamination_results.parquet"
    contamination_stage = runner.runtime.stage("contamination")
    contamination_ready = (
        contamination_path.is_file()
        and contamination_stage.get("status") in {"PASS", "BLOCKED"}
    )
    contamination_iter = (
        iter(_iter_contamination_results(runner.repo_root, start_row=rows_scanned))
        if contamination_ready
        else None
    )
    contamination_pending = None
    if contamination_iter is not None:
        try:
            contamination_pending = next(contamination_iter)
        except StopIteration:
            contamination_pending = None
    eval_entries = _available_eval_entries(runner.repo_root) if (runner.repo_root / "data" / "manifests" / "eval_registry.json").is_file() else []
    missing_eval = _required_eval_missing(eval_entries) if eval_entries else True
    def source_records() -> Iterator[tuple[DatasetLocation, SourceRow]]:
        source_rows = _iter_locations(
            runner.repo_root,
            runner.dataset_root,
            after_dataset_id=resume_dataset_id or None,
            after_source_row_id=resume_source_row_id or None,
            batch_size=16,
        )
        for location, source_row in source_rows:
            yield location, source_row

    with output_path.open("a" if rows_scanned else "w", encoding="utf-8") as output:
        tokenized_records = _ordered_tokenized_records(
            source_records(),
            cache_root=runner.cache_root,
            model_root=runner.model_root,
            model_id=runner.base_model_id,
            revision=revision,
            workers=workers,
            sequential_counter=counter,
        )
        for record, counts in tokenized_records:
            if counts.get("supervised_assistant_tokens") is None:
                mask_unavailable += 1
            if counts.get("assistant_mask_status") == "unavailable_chat_template":
                template_unavailable += 1
            if counts.get("assistant_mask_status") == "unavailable_oversized_record":
                oversized_unavailable += 1
                max_oversized_characters = max(
                    max_oversized_characters,
                    int(record.get("raw_character_count") or 0),
                )
            record.update(counts)
            source_dataset = str(record.get("source_dataset", ""))
            source_path = str(record.get("source_path", ""))
            source_split = str(record.get("source_split", ""))
            if source_dataset == "nemotron_sft_multilingual_v2":
                record["nemotron_bucket"] = classify_nemotron_split(
                    source_path,
                    source_split,
                )
                if not is_known_nemotron_split(source_path, source_split):
                    unknown_nemotron += 1
                elif nemotron_split_conflict(source_path, source_split):
                    unknown_nemotron += 1
                record["selection_bucket"] = selection_bucket(record)
            if source_dataset == "fable_5_premium":
                record["fable_bucket"] = fable_length_bucket(int(counts.get("training_formatted_tokens") or 0))
            contamination_item = None
            if contamination_pending is not None:
                if str(contamination_pending.get("record_id", "")) == str(record.get("record_id")):
                    contamination_item = contamination_pending
                    if not record.get("content_hash"):
                        record["content_hash"] = contamination_item.get("content_hash")
                    try:
                        contamination_pending = next(contamination_iter) if contamination_iter is not None else None
                    except StopIteration:
                        contamination_pending = None
            record["contamination_status"] = (
                str(contamination_item.get("status"))
                if contamination_item
                else "not_checked_missing_eval_source" if missing_eval else "not_scanned"
            )
            stats[(source_dataset, str(record.get("category", "unknown")))].add(counts)
            metadata = {
                key: record.get(key)
                for key in TOKEN_METADATA_FIELDS
                + (
                    "contamination_status",
                    "raw_content_tokens",
                    "training_formatted_tokens",
                    "supervised_assistant_tokens",
                    "assistant_mask_status",
                )
            }
            output.write(json.dumps(metadata, ensure_ascii=False, sort_keys=True) + "\n")
            rows_scanned += 1
            if rows_scanned % 10_000 == 0:
                runner.runtime.checkpoint(
                    "tokenize",
                    {
                        "rows_scanned": rows_scanned,
                        "source_dataset": source_dataset,
                        "source_path": source_path,
                        "source_row_id": str(record.get("source_row_id", "")),
                    },
                )
    csv_path, md_path = _write_token_reports(
        runner.repo_root,
        stats,
        revision,
        model_id=runner.model_id,
        base_model_id=runner.base_model_id,
    )
    details = {
        "rows_scanned": rows_scanned,
        "mask_unavailable_rows": mask_unavailable,
        "template_unavailable_rows": template_unavailable,
        "oversized_unavailable_rows": oversized_unavailable,
        "max_oversized_characters": max_oversized_characters,
        "oversized_character_limit": MAX_TOKENIZATION_CHARACTERS,
        "unknown_nemotron_rows": unknown_nemotron,
        "model_id": runner.model_id,
        "base_model_id": runner.base_model_id,
        "tokenizer_revision": revision,
        "workers": workers,
        "token_rows_path": str(output_path),
    }
    if unknown_nemotron:
        return {
            "status": "BLOCKED",
            "artifacts": [output_path, csv_path, md_path],
            "blocked_reasons": ["unknown_nemotron_split"],
            "details": details,
        }
    return {"status": "PASS", "artifacts": [output_path, csv_path, md_path], "details": details}


def _write_token_reports(
    repo_root: Path,
    stats: Mapping[tuple[str, str], TokenStats],
    revision: str,
    *,
    model_id: str,
    base_model_id: str,
) -> tuple[Path, Path]:
    report_dir = _report_dir(repo_root)
    csv_path = report_dir / "token_statistics.csv"
    fields = [
        "dataset",
        "category",
        "rows",
        "raw_content_tokens",
        "training_formatted_tokens",
        "supervised_assistant_tokens",
        "supervised_rows",
        "quantile_sample_size",
        "quantile_method",
        "total_tokens",
        "median",
        "mean",
        "p75",
        "p90",
        "p95",
        "p99",
        "max",
        "<=1k",
        "1k-4k",
        "4k-8k",
        "8k-32k",
        ">32k",
    ]
    rows: list[dict[str, Any]] = []
    for (dataset, category), accumulator in sorted(stats.items()):
        row = {"dataset": dataset, "category": category}
        row.update(accumulator.as_dict())
        rows.append(row)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Token Statistics",
        "",
        "Statistics use the actual Qwen tokenizer at the resolved full revision. Raw rows and generated content are not included in this report.",
        "Quantiles are exact for groups up to 100,000 rows and use a deterministic bounded reservoir for larger groups.",
        "",
        f"- model: `{model_id}`",
        f"- base model/tokenizer: `{base_model_id}`",
        f"- resolved tokenizer revision: `{revision}`",
        "- assistant masks unavailable are recorded as unavailable; they are not estimated.",
        "",
        "| Dataset | Category | Rows | Formatted tokens | Median | Mean | P95 | Max | Supervised rows |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['dataset']}` | `{row['category']}` | {row['rows']:,} | {row['total_tokens']:,} | {row['median'] or 'n/a'} | {row['mean'] or 'n/a'} | {row['p95'] or 'n/a'} | {row['max'] or 'n/a'} | {row['supervised_rows']:,} |"
        )
    md_path = report_dir / "TOKEN_STATISTICS.md"
    atomic_write_text(md_path, "\n".join(lines) + "\n")
    return csv_path, md_path
