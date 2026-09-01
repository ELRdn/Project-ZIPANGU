"""Exact and scalable near-duplicate helpers."""

from __future__ import annotations

from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Iterator, Mapping


def normalized_record_text(record: Mapping[str, Any]) -> str:
    messages = record.get("messages")
    if isinstance(messages, list):
        parts = []
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            role = " ".join(str(message.get("role", "")).split()).casefold()
            content = " ".join(str(message.get("content", "")).split())
            parts.append(f"<{role}>:{content}")
        return "\n".join(parts)
    return "\n".join(
        f"<{role}>:{' '.join(str(record.get(field, '') or '').split())}"
        for role, field in (("user", "user"), ("assistant", "assistant_final"), ("reasoning", "reasoning"))
        if record.get(field)
    )


def exact_content_hash(record: Mapping[str, Any]) -> str:
    return hashlib.sha256(normalized_record_text(record).encode("utf-8")).hexdigest()


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _winner_key(record: Mapping[str, Any]) -> tuple[float, int, int, int, str, str]:
    provenance = record.get("source_revision") not in (None, "", "unknown")
    license_known = record.get("license_raw") not in (None, "", "unknown", ["unknown"])
    schema_complete = bool(record.get("messages")) and bool(record.get("user")) and bool(record.get("assistant_final"))
    return (
        int(provenance),
        int(license_known),
        int(schema_complete),
        float(record.get("quality_score") or 0.0),
        str(record.get("source_dataset", "")),
        str(record.get("source_row_id", "")),
    )


def exact_deduplicate(records: Iterable[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    winners: dict[str, dict[str, Any]] = {}
    duplicates: list[dict[str, Any]] = []
    for raw_record in records:
        record = dict(raw_record)
        if record.get("eligibility") is not True:
            continue
        content_hash = str(record.get("content_hash") or exact_content_hash(record))
        record["content_hash"] = content_hash
        record["duplicate_cluster"] = f"exact:{content_hash}"
        previous = winners.get(content_hash)
        if previous is None:
            winners[content_hash] = record
            continue
        if _winner_key(record) > _winner_key(previous):
            duplicates.append({**previous, "duplicate_of": record.get("record_id"), "duplicate_kind": "exact"})
            winners[content_hash] = record
        else:
            duplicates.append({**record, "duplicate_of": previous.get("record_id"), "duplicate_kind": "exact"})
    return list(winners.values()), duplicates


def _iter_jsonl_gz(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value


def iter_audit_records(
    audit_dir: str | Path,
    dataset_ids: Iterable[str] | None = None,
) -> Iterator[dict[str, Any]]:
    root = Path(audit_dir)
    selected = {str(dataset_id) for dataset_id in dataset_ids} if dataset_ids else None
    for path in sorted(root.glob("*.jsonl.gz"), key=lambda item: item.name.casefold()):
        dataset_id = path.name[: -len(".jsonl.gz")]
        if selected is not None and dataset_id not in selected:
            continue
        yield from _iter_jsonl_gz(path)


def _iter_sqlite_winners(db_path: Path) -> Iterator[dict[str, Any]]:
    with sqlite3.connect(db_path) as connection:
        for (record_json,) in connection.execute("SELECT record_json FROM winners ORDER BY content_hash"):
            value = json.loads(record_json)
            if isinstance(value, dict):
                yield value


def _signature_int(record: Mapping[str, Any]) -> int:
    raw = record.get("simhash")
    try:
        return int(str(raw), 16)
    except (TypeError, ValueError):
        digest = hashlib.blake2b(normalized_record_text(record).encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big")


def near_deduplicate(
    records: Iterable[Mapping[str, Any]],
    *,
    hamming_threshold: int = 3,
    bucket_bits: int = 16,
    max_records: int | None = 250000,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Screen near duplicates with LSH buckets, never pairwise all-v-all."""

    bands: dict[tuple[int, int], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    accepted: list[dict[str, Any]] = []
    near_duplicates: list[dict[str, Any]] = []
    inspected = 0
    considered = 0
    band_count = max(1, 64 // max(1, bucket_bits))
    for record_value in records:
        if record_value.get("eligibility") is not True:
            continue
        considered += 1
        if max_records is not None and inspected >= max_records:
            continue
        signature = _signature_int(record_value)
        duplicate_of: dict[str, Any] | None = None
        for band_index in range(band_count):
            key = (band_index, (signature >> (band_index * bucket_bits)) & ((1 << bucket_bits) - 1))
            for previous_signature, previous in bands[key]:
                if hamming_distance(signature, previous_signature) <= hamming_threshold:
                    duplicate_of = previous
                    break
            if duplicate_of is not None:
                break
        if duplicate_of is not None:
            near_duplicates.append({**dict(record_value), "duplicate_of": duplicate_of.get("record_id"), "duplicate_kind": "near"})
        else:
            record = dict(record_value)
            accepted.append(record)
            for band_index in range(band_count):
                key = (band_index, (signature >> (band_index * bucket_bits)) & ((1 << bucket_bits) - 1))
                bands[key].append((signature, record))
        inspected += 1
    return accepted, near_duplicates, {
        "considered_eligible": considered,
        "inspected": inspected,
        "max_records": max_records,
        "capped": max_records is not None and considered > inspected,
        "algorithm": "simhash_lsh",
        "hamming_threshold": hamming_threshold,
        "bucket_bits": bucket_bits,
    }


def run_dedup_sqlite(
    audit_dir: str | Path,
    output_dir: str | Path,
    *,
    hamming_threshold: int = 3,
    bucket_bits: int = 16,
    near_max_records: int | None = 250000,
    dataset_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Run exact dedup with a disk-backed index and write bounded artifacts."""

    selected_dataset_ids = (
        sorted({str(dataset_id) for dataset_id in dataset_ids}) if dataset_ids is not None else None
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    db_path = output / "exact_index.sqlite3"
    connection = sqlite3.connect(db_path)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS winners (content_hash TEXT PRIMARY KEY, record_json TEXT NOT NULL)"
    )
    connection.execute("CREATE TABLE IF NOT EXISTS counters (name TEXT PRIMARY KEY, value INTEGER NOT NULL)")
    # The audit input is a snapshot for this run; never merge stale runtime rows.
    connection.execute("DELETE FROM winners")
    connection.execute("DELETE FROM counters")
    connection.commit()
    eligible = exact_duplicates = 0
    for record in iter_audit_records(audit_dir, dataset_ids=selected_dataset_ids):
        if record.get("eligibility") is not True:
            continue
        eligible += 1
        content_hash = str(record.get("content_hash") or exact_content_hash(record))
        record["content_hash"] = content_hash
        record["duplicate_cluster"] = f"exact:{content_hash}"
        existing = connection.execute(
            "SELECT record_json FROM winners WHERE content_hash = ?", (content_hash,)
        ).fetchone()
        if existing is None:
            connection.execute(
                "INSERT INTO winners(content_hash, record_json) VALUES (?, ?)",
                (content_hash, json.dumps(record, ensure_ascii=False, separators=(",", ":"))),
            )
        else:
            exact_duplicates += 1
            previous = json.loads(existing[0])
            if _winner_key(record) > _winner_key(previous):
                connection.execute(
                    "UPDATE winners SET record_json = ? WHERE content_hash = ?",
                    (json.dumps(record, ensure_ascii=False, separators=(",", ":")), content_hash),
                )
        if eligible % 5000 == 0:
            connection.commit()
    connection.commit()
    winner_count = int(connection.execute("SELECT COUNT(*) FROM winners").fetchone()[0])
    connection.close()

    winners_path = output / "exact_winners.jsonl.gz"
    with gzip.open(winners_path, "wt", encoding="utf-8") as handle:
        for winner in _iter_sqlite_winners(db_path):
            handle.write(json.dumps(winner, ensure_ascii=False, separators=(",", ":")) + "\n")
    _, near_duplicates, near_stats = near_deduplicate(
        _iter_sqlite_winners(db_path),
        hamming_threshold=hamming_threshold,
        bucket_bits=bucket_bits,
        max_records=near_max_records,
    )
    near_path = output / "near_duplicates.jsonl.gz"
    with gzip.open(near_path, "wt", encoding="utf-8") as handle:
        for record in near_duplicates:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = {
        "dataset_ids": selected_dataset_ids,
        "eligible_records": eligible,
        "exact_unique_records": winner_count,
        "exact_duplicate_records": exact_duplicates,
        "near_duplicate_records": len(near_duplicates),
        "near": near_stats,
        "artifacts": [str(winners_path), str(near_path), str(db_path)],
    }
    (output / "dedup_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
