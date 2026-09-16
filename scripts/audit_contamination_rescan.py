from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_SEED = 3407
DEFAULT_SAMPLE_SIZE = 100
DEFAULT_EXPECTED_ROWS = 4643875
DEFAULT_BATCH_SIZE = 65536
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()

# Raw text must never be projected or exported by this audit.
BLOCKED_TEXT_FIELDS = frozenset(
    {
        "text",
        "candidate_text",
        "eval_text",
        "prompt_text",
        "completion_text",
        "messages",
        "messages_json",
        "prompt_messages_json",
        "content",
        "prompt",
        "completion",
    }
)


class RescanAuditError(RuntimeError):
    """Fail-closed audit error; the caller maps this to a non-zero exit."""


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _atomic_write_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="strict").strip()
        except UnicodeDecodeError:
            return ""
    return str(value).strip()


def _split_tokens(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        tokens: list[str] = []
        for item in value:
            cleaned = _clean(item)
            if cleaned:
                tokens.append(cleaned)
        return tokens
    text = _clean(value)
    if not text:
        return []
    for separator in (";", ",", "|"):
        text = text.replace(separator, ";")
    return [item.strip() for item in text.split(";") if item.strip()]


def _normalize_eval_ids(value: Any) -> list[str]:
    return sorted(set(_split_tokens(value)))


def _eval_corpus(eval_id: str) -> str:
    return eval_id.split(":", 1)[0] if eval_id else "unmatched_eval"


def _exact_reason(reasons: Any) -> str:
    tokens = _split_tokens(reasons)
    exact_reasons = [
        reason
        for reason in (
            "full_content_sha256",
            "prompt_sha256",
            "long_substring_ge_80",
        )
        if reason in tokens
    ]
    return "+".join(exact_reasons)


def _is_genuine_exact(status: Any, content_hash: Any, reasons: Any) -> bool:
    if _clean(status) != "quarantine_exact":
        return False
    content = _clean(content_hash)
    if not content or content == EMPTY_SHA256:
        return False
    return bool(_exact_reason(reasons))


def _resolve_column(available: set[str], candidates: list[str]) -> str | None:
    for name in candidates:
        if name in available:
            return name
    return None


def allocate_stratified_sample(
    counts: Mapping[str, int],
    sample_size: int,
) -> dict[str, int]:
    positive = {key: int(value) for key, value in counts.items() if int(value) > 0}
    if sample_size <= 0 or not positive:
        return {key: 0 for key in sorted(positive)}
    if sample_size < len(positive):
        selected = sorted(positive, key=lambda key: (-positive[key], key))[:sample_size]
        return {key: int(key in selected) for key in sorted(positive)}
    allocation = {key: 1 for key in positive}
    remaining = sample_size - len(positive)
    total = sum(positive.values())
    quotas = {key: remaining * positive[key] / total for key in positive}
    for key, quota in quotas.items():
        allocation[key] += math.floor(quota)
    leftovers = sample_size - sum(allocation.values())
    order = sorted(
        positive,
        key=lambda key: (-(quotas[key] - math.floor(quotas[key])), -positive[key], key),
    )
    for key in order[:leftovers]:
        allocation[key] += 1
    return dict(sorted(allocation.items()))


def _sample_priority(seed: int, record_id: str, content_hash: str, stratum: str) -> int:
    payload = f"{seed}|{record_id}|{content_hash}|{stratum}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _push_reservoir(
    reservoir: list[tuple[int, dict[str, Any]]],
    row: dict[str, Any],
    *,
    priority: int,
    capacity: int,
) -> None:
    entry = (-priority, row)
    if len(reservoir) < capacity:
        heapq.heappush(reservoir, entry)
    elif capacity > 0 and priority < -reservoir[0][0]:
        heapq.heapreplace(reservoir, entry)


def _audit(
    old_path: Path,
    new_path: Path,
    out_dir: Path,
    *,
    sample_size: int,
    seed: int,
    expected_rows: int,
    batch_size: int,
) -> dict[str, Any]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RescanAuditError(
            "pyarrow is required; install the project audit extra"
        ) from exc
    if not old_path.is_file():
        raise RescanAuditError(f"old input is missing: {old_path}")
    if not new_path.is_file():
        raise RescanAuditError(f"new input is missing: {new_path}")
    if sample_size < 0:
        raise RescanAuditError("--sample-size must be >= 0")
    if batch_size <= 0:
        raise RescanAuditError("--batch-size must be > 0")

    old_file = parquet.ParquetFile(old_path)
    new_file = parquet.ParquetFile(new_path)
    old_total = int(old_file.metadata.num_rows)
    new_total = int(new_file.metadata.num_rows)
    if old_total != new_total:
        raise RescanAuditError(
            f"row count mismatch: old={old_total:,} new={new_total:,}"
        )
    if expected_rows > 0 and old_total != expected_rows:
        raise RescanAuditError(
            f"row count does not match expected: actual={old_total:,} "
            f"expected={expected_rows:,}"
        )

    # NB: schema_arrow keeps logical top-level names; the physical schema flattens list columns to leaf element names.
    old_names = set(old_file.schema_arrow.names)
    new_names = set(new_file.schema_arrow.names)
    old_record_col = _resolve_column(old_names, ["record_id"])
    new_record_col = _resolve_column(new_names, ["record_id"])
    old_status_col = _resolve_column(old_names, ["status"])
    new_status_col = _resolve_column(new_names, ["status"])
    if old_record_col is None or new_record_col is None:
        raise RescanAuditError("record_id column is missing from old or new input")
    if old_status_col is None or new_status_col is None:
        raise RescanAuditError("status column is missing from old or new input")
    old_content_col = _resolve_column(old_names, ["content_sha256", "content_hash"])
    new_content_col = _resolve_column(new_names, ["content_sha256", "content_hash"])
    old_prompt_col = _resolve_column(old_names, ["prompt_sha256", "prompt_hash"])
    new_prompt_col = _resolve_column(new_names, ["prompt_sha256", "prompt_hash"])
    old_reasons_col = _resolve_column(old_names, ["reasons", "reason"])
    new_reasons_col = _resolve_column(new_names, ["reasons", "reason"])
    old_source_col = _resolve_column(old_names, ["source_dataset"])
    new_source_col = _resolve_column(new_names, ["source_dataset"])
    old_row_col = _resolve_column(old_names, ["source_row_id"])
    new_row_col = _resolve_column(new_names, ["source_row_id"])
    old_matched_col = _resolve_column(
        old_names, ["matched_eval_ids", "matched_eval_id", "matches"]
    )
    new_matched_col = _resolve_column(
        new_names, ["matched_eval_ids", "matched_eval_id", "matches"]
    )
    column_mapping = {
        "old": {
            "record_id": old_record_col,
            "status": old_status_col,
            "content_sha256": old_content_col,
            "prompt_sha256": old_prompt_col,
            "reasons": old_reasons_col,
            "source_dataset": old_source_col,
            "source_row_id": old_row_col,
            "matched_eval_ids": old_matched_col,
        },
        "new": {
            "record_id": new_record_col,
            "status": new_status_col,
            "content_sha256": new_content_col,
            "prompt_sha256": new_prompt_col,
            "reasons": new_reasons_col,
            "source_dataset": new_source_col,
            "source_row_id": new_row_col,
            "matched_eval_ids": new_matched_col,
        },
    }
    warnings: list[str] = []
    for side in ("old", "new"):
        mapping = column_mapping[side]
        for logical in ("content_sha256", "reasons", "source_dataset"):
            if mapping[logical] is None:
                warnings.append(f"{side} input is missing {logical}; using default")
        if mapping["matched_eval_ids"] is None:
            warnings.append(f"{side} input is missing matched_eval_ids; unmatched")
    for mapping in (column_mapping["old"], column_mapping["new"]):
        for physical in mapping.values():
            if physical in BLOCKED_TEXT_FIELDS:
                raise RescanAuditError(f"refusing to project raw text: {physical}")

    old_columns = sorted({v for v in column_mapping["old"].values() if v})
    new_columns = sorted({v for v in column_mapping["new"].values() if v})
    waterfall: Counter[tuple[str, str]] = Counter()
    old_status_counts: Counter[str] = Counter()
    new_status_counts: Counter[str] = Counter()
    old_genuine = 0
    new_genuine = 0
    empty_sha_exact_new = 0
    stratum_counts: Counter[str] = Counter()
    reservoirs: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    scanned = 0
    old_batches = old_file.iter_batches(columns=old_columns, batch_size=batch_size)
    new_batches = new_file.iter_batches(columns=new_columns, batch_size=batch_size)
    old_buffer: list[dict[str, Any]] = []
    new_buffer: list[dict[str, Any]] = []
    old_exhausted = False
    new_exhausted = False
    while True:
        if not old_buffer and not old_exhausted:
            try:
                old_buffer = next(old_batches).to_pylist()
            except StopIteration:
                old_exhausted = True
        if not new_buffer and not new_exhausted:
            try:
                new_buffer = next(new_batches).to_pylist()
            except StopIteration:
                new_exhausted = True
        if not old_buffer and not new_buffer:
            break
        if not old_buffer or not new_buffer:
            raise RescanAuditError("row stream ended early on one input")
        take = min(len(old_buffer), len(new_buffer))
        old_chunk = old_buffer[:take]
        new_chunk = new_buffer[:take]
        old_buffer = old_buffer[take:]
        new_buffer = new_buffer[take:]
        for offset in range(take):
            old_row = old_chunk[offset]
            new_row = new_chunk[offset]
            old_id = _clean(old_row.get(old_record_col))
            new_id = _clean(new_row.get(new_record_col))
            if not old_id or not new_id:
                raise RescanAuditError(f"empty record_id at row order {scanned}")
            if old_id != new_id:
                raise RescanAuditError(
                    f"record_id order mismatch at index {scanned}: "
                    f"old={old_id} new={new_id}"
                )
            old_status = _clean(old_row.get(old_status_col)) or "unknown_status"
            new_status = _clean(new_row.get(new_status_col)) or "unknown_status"
            waterfall[(old_status, new_status)] += 1
            old_status_counts[old_status] += 1
            new_status_counts[new_status] += 1
            old_content = _clean(old_row.get(old_content_col)) if old_content_col else ""
            new_content = _clean(new_row.get(new_content_col)) if new_content_col else ""
            old_reasons = old_row.get(old_reasons_col) if old_reasons_col else ""
            new_reasons = new_row.get(new_reasons_col) if new_reasons_col else ""
            if _is_genuine_exact(old_status, old_content, old_reasons):
                old_genuine += 1
            new_is_genuine = _is_genuine_exact(new_status, new_content, new_reasons)
            if new_is_genuine:
                new_genuine += 1
            elif new_status == "quarantine_exact" and new_content == EMPTY_SHA256:
                empty_sha_exact_new += 1
            if new_is_genuine:
                source = (
                    _clean(new_row.get(new_source_col)) if new_source_col else ""
                ) or "unknown_source"
                source_row_id = _clean(new_row.get(new_row_col)) if new_row_col else ""
                matched_raw = new_row.get(new_matched_col) if new_matched_col else ""
                matched_ids = _normalize_eval_ids(matched_raw)
                primary_eval = matched_ids[0] if matched_ids else "unmatched_eval"
                eval_corpus = _eval_corpus(primary_eval)
                reason = _exact_reason(new_reasons)
                stratum = f"{source}|{eval_corpus}|{reason}|content:{new_content}"
                stratum_counts[stratum] += 1
                priority = _sample_priority(seed, new_id, new_content, stratum)
                prompt_value = _clean(new_row.get(new_prompt_col)) if new_prompt_col else ""
                candidate = {
                    "record_id": new_id,
                    "source_dataset": source,
                    "source_row_id": source_row_id,
                    "old_status": old_status,
                    "new_status": new_status,
                    "content_sha256": new_content,
                    "prompt_sha256": prompt_value,
                    "exact_reason": reason,
                    "eval_corpus": eval_corpus,
                    "matched_eval_id": primary_eval,
                    "matched_eval_ids": ";".join(matched_ids),
                    "sample_stratum": stratum,
                    "sample_priority": f"{priority:016x}",
                }
                _push_reservoir(
                    reservoirs[stratum],
                    candidate,
                    priority=priority,
                    capacity=sample_size,
                )
            scanned += 1
    if scanned != old_total or scanned != new_total:
        raise RescanAuditError(
            f"streamed count does not reconcile: scanned={scanned:,} "
            f"old={old_total:,} new={new_total:,}"
        )

    allocation = allocate_stratified_sample(stratum_counts, sample_size)
    sample_rows: list[dict[str, Any]] = []
    for stratum in sorted(allocation):
        rows = sorted(
            (row for _, row in reservoirs[stratum]),
            key=lambda row: (row["sample_priority"], row["record_id"]),
        )
        for row in rows[: allocation[stratum]]:
            sample_rows.append(dict(row))
    sample_rows.sort(
        key=lambda row: (
            row["source_dataset"],
            row["matched_eval_id"],
            row["exact_reason"],
            row["sample_priority"],
            row["record_id"],
        )
    )
    for sample_id, row in enumerate(sample_rows, start=1):
        row["sample_id"] = sample_id
    ordered_sample_rows = [
        {
            "sample_id": row["sample_id"],
            "record_id": row["record_id"],
            "source_dataset": row["source_dataset"],
            "source_row_id": row["source_row_id"],
            "old_status": row["old_status"],
            "new_status": row["new_status"],
            "content_sha256": row["content_sha256"],
            "prompt_sha256": row["prompt_sha256"],
            "exact_reason": row["exact_reason"],
            "eval_corpus": row["eval_corpus"],
            "matched_eval_id": row["matched_eval_id"],
            "matched_eval_ids": row["matched_eval_ids"],
            "sample_stratum": row["sample_stratum"],
            "sample_priority": row["sample_priority"],
        }
        for row in sample_rows
    ]
    expected_sample = min(sample_size, new_genuine)
    if len(ordered_sample_rows) != expected_sample:
        raise RescanAuditError("stratified sample size does not reconcile")

    waterfall_rows = [
        {"old_status": old, "new_status": new, "count": count}
        for (old, new), count in sorted(
            waterfall.items(), key=lambda item: (-item[1], item[0][0], item[0][1])
        )
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    waterfall_path = out_dir / "contamination_rescan_waterfall.csv"
    sample_path = out_dir / "contamination_rescan_genuine_exact_sample.csv"
    summary_path = out_dir / "contamination_rescan_audit.json"
    _atomic_write_csv(
        waterfall_path,
        waterfall_rows,
        ["old_status", "new_status", "count"],
    )
    if ordered_sample_rows:
        _atomic_write_csv(sample_path, ordered_sample_rows, list(ordered_sample_rows[0]))
    else:
        _atomic_write_csv(
            sample_path,
            [],
            [
                "sample_id",
                "record_id",
                "source_dataset",
                "source_row_id",
                "old_status",
                "new_status",
                "content_sha256",
                "prompt_sha256",
                "exact_reason",
                "eval_corpus",
                "matched_eval_id",
                "matched_eval_ids",
                "sample_stratum",
                "sample_priority",
            ],
        )
    result: dict[str, Any] = {
        "schema_version": 1,
        "scope": "post_rescan_old_vs_new",
        "inputs": {
            "old": {
                "path": str(old_path.resolve()),
                "sha256": _sha256_file(old_path),
                "parquet_rows": old_total,
            },
            "new": {
                "path": str(new_path.resolve()),
                "sha256": _sha256_file(new_path),
                "parquet_rows": new_total,
            },
        },
        "expected_rows": expected_rows,
        "row_identity": {
            "method": "record_id equality in source order",
            "rows_compared": scanned,
            "mismatches": 0,
        },
        "column_mapping": column_mapping,
        "warnings": warnings,
        "status_counts": {
            "old": dict(sorted(old_status_counts.items())),
            "new": dict(sorted(new_status_counts.items())),
        },
        "waterfall": waterfall_rows,
        "genuine_exact": {
            "definition": (
                "status=quarantine_exact with non-empty content_sha256 "
                "excluding empty-sha256 plus exact full-content, prompt, or "
                "80-character substring evidence"
            ),
            "old_rows": old_genuine,
            "new_rows": new_genuine,
            "empty_sha256_exact_new_excluded": empty_sha_exact_new,
        },
        "sample": {
            "method": (
                "deterministic hash-priority stratified sample of new genuine "
                "exact rows across source_dataset, eval_corpus, reason, hash cluster"
            ),
            "seed": seed,
            "population": new_genuine,
            "target_size": sample_size,
            "actual_size": len(ordered_sample_rows),
            "stratum_counts": dict(sorted(stratum_counts.items())),
            "allocation": allocation,
            "contains_raw_text": False,
        },
        "artifacts": {
            "waterfall_csv": str(waterfall_path.resolve()),
            "sample_csv": str(sample_path.resolve()),
            "summary_json": str(summary_path.resolve()),
        },
        "safety": {
            "training_allowed": False,
            "contains_raw_text": False,
        },
    }
    _atomic_write_json(summary_path, result)
    result["artifacts"]["waterfall_sha256"] = _sha256_file(waterfall_path)
    result["artifacts"]["sample_sha256"] = _sha256_file(sample_path)
    _atomic_write_json(summary_path, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare archived and rescanned contamination Parquet files, emit an "
            "old-vs-new transition waterfall, and sample genuine exact rows."
        )
    )
    parser.add_argument("--old-input", type=Path, required=True)
    parser.add_argument("--new-input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--expected-rows", type=int, default=DEFAULT_EXPECTED_ROWS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        result = _audit(
            args.old_input,
            args.new_input,
            args.out_dir,
            sample_size=args.sample_size,
            seed=args.seed,
            expected_rows=args.expected_rows,
            batch_size=args.batch_size,
        )
    except RescanAuditError as exc:
        print(f"contamination rescan audit failed: {exc}", flush=True)
        return 2
    print(json.dumps(result["genuine_exact"], ensure_ascii=False, indent=2))
    print(json.dumps({"waterfall_rows": len(result["waterfall"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
