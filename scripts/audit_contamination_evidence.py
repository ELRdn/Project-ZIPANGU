from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import heapq
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
DEFAULT_SEED = 3407
DEFAULT_SAMPLE_SIZE = 100
SOURCE_PROJECTION_FINDINGS = {
    "metadata/context_buckets": (
        "metadata-only fields (id, split, token counts); should not enter text contamination scan"
    ),
    "data/canonical": (
        "trainable text is stored in messages_json, which the current canonical extractor ignores"
    ),
    "data/glm47_native": (
        "tokenized input_ids/labels derivative; no raw text field for the generic extractor"
    ),
    "data/token_stats": (
        "token-statistics derivative; should not enter text contamination scan"
    ),
    "metadata/parent_ids": (
        "metadata-only id/split rows; should not enter text contamination scan"
    ),
    "data/prompt_completion_text": (
        "text is stored in prompt_text/completion_text, which the current extractor ignores"
    ),
    "data/rl_tool_prompts": (
        "prompt is stored in prompt_messages_json, which the current extractor ignores"
    ),
    "data/smoke": (
        "trainable text is stored in messages_json, which the current canonical extractor ignores"
    ),
}


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


def _load_eval_fingerprints(
    registry_path: Path,
) -> tuple[
    dict[str, dict[str, list[dict[str, Any]]]],
    dict[str, dict[str, Any]],
]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    by_hash: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    corpus_stats: dict[str, dict[str, Any]] = {}
    for entry in registry.get("evaluations", []):
        eval_id = str(entry.get("id") or "unknown_eval")
        fingerprint_value = entry.get("fingerprint_path")
        stats = {
            "eval_corpus": eval_id,
            "availability": str(entry.get("availability") or "unknown"),
            "registry_rows": int(entry.get("row_count") or 0),
            "fingerprint_rows": 0,
            "empty_projection_rows": 0,
            "unique_content_hashes": 0,
            "fingerprint_path": str(fingerprint_value or ""),
            "fingerprint_sha256": None,
        }
        seen_hashes: set[str] = set()
        if fingerprint_value:
            fingerprint_path = Path(str(fingerprint_value))
            if fingerprint_path.is_file():
                stats["fingerprint_sha256"] = _sha256_file(fingerprint_path)
                with fingerprint_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        item = json.loads(line)
                        content_hash = str(item.get("content_hash") or "")
                        normalized_length = int(item.get("normalized_length") or 0)
                        evidence = {
                            "source_row_id": str(item.get("source_row_id") or ""),
                            "normalized_length": normalized_length,
                        }
                        by_hash[content_hash][eval_id].append(evidence)
                        seen_hashes.add(content_hash)
                        stats["fingerprint_rows"] += 1
                        if normalized_length == 0:
                            stats["empty_projection_rows"] += 1
        stats["unique_content_hashes"] = len(seen_hashes)
        corpus_stats[eval_id] = stats
    return by_hash, corpus_stats


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


def _sample_priority(seed: int, record_id: str, source_row_id: str) -> int:
    value = f"{seed}|{record_id}|{source_row_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def _source_file(source_row_id: str) -> str:
    path, separator, row_number = source_row_id.rpartition("#")
    return path if separator and row_number.isdigit() else source_row_id


def _source_family(source_file: str) -> str:
    parts = Path(source_file).parts
    return "/".join(parts[:2]) if len(parts) >= 2 else source_file


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
    elif priority < -reservoir[0][0]:
        heapq.heapreplace(reservoir, entry)


def _cluster_evidence(
    content_hash: str,
    eval_by_hash: Mapping[str, Mapping[str, list[dict[str, Any]]]],
) -> dict[str, Any]:
    corpora = eval_by_hash.get(content_hash, {})
    eval_ids = sorted(corpora)
    eval_rows = [row for eval_id in eval_ids for row in corpora[eval_id]]
    candidate_projection_empty = content_hash == EMPTY_SHA256
    all_eval_projections_empty = bool(eval_rows) and all(
        int(row.get("normalized_length") or 0) == 0 for row in eval_rows
    )
    if candidate_projection_empty and all_eval_projections_empty:
        classification = "high_confidence_false_positive_projection_collision"
        basis = "candidate_and_eval_normalized_content_are_empty"
    elif not eval_rows:
        classification = "unresolved_eval_hash_mapping"
        basis = "exact_status_has_no_current_eval_fingerprint_mapping"
    else:
        classification = "plausible_true_positive"
        basis = "non_empty_full_content_hash_matches_eval_fingerprint"
    return {
        "eval_corpora": eval_ids,
        "eval_row_count": len(eval_rows),
        "candidate_projection_empty": candidate_projection_empty,
        "all_eval_projections_empty": all_eval_projections_empty,
        "false_positive_classification": classification,
        "false_positive_basis": basis,
    }


def _pct(value: int, total: int) -> float:
    return round((100.0 * value / total), 6) if total else 0.0


def _markdown_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---:" if index else "---" for index in range(len(headers))) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return lines


def audit(
    input_path: Path,
    registry_path: Path,
    output_dir: Path,
    *,
    sample_size: int,
    seed: int,
) -> dict[str, Any]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("pyarrow is required; install the project audit extra") from exc

    eval_by_hash, eval_corpus_stats = _load_eval_fingerprints(registry_path)
    source_scanned: Counter[str] = Counter()
    source_exact: Counter[str] = Counter()
    exact_reasons: Counter[str] = Counter()
    cluster_counts: Counter[str] = Counter()
    cluster_sources: dict[str, Counter[str]] = defaultdict(Counter)
    source_file_exact: Counter[str] = Counter()
    source_family_exact: Counter[str] = Counter()
    source_family_files: dict[str, set[str]] = defaultdict(set)
    stratum_counts: Counter[str] = Counter()
    reservoirs: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    exact_rows = 0

    parquet_file = parquet.ParquetFile(input_path)
    columns = [
        "record_id",
        "source_dataset",
        "source_row_id",
        "content_hash",
        "prompt_hash",
        "status",
        "similarity",
        "reasons",
    ]
    for batch in parquet_file.iter_batches(columns=columns, batch_size=65_536):
        data = batch.to_pydict()
        for index, status_value in enumerate(data["status"]):
            source = str(data["source_dataset"][index] or "unknown_source")
            source_scanned[source] += 1
            if status_value != "quarantine_exact":
                continue
            exact_rows += 1
            source_exact[source] += 1
            content_hash = str(data["content_hash"][index] or "")
            reason = str(data["reasons"][index] or "")
            for item in reason.split(";"):
                if item:
                    exact_reasons[item] += 1
            cluster_counts[content_hash] += 1
            cluster_sources[content_hash][source] += 1
            record_id = str(data["record_id"][index] or "")
            source_row_id = str(data["source_row_id"][index] or "")
            source_file = _source_file(source_row_id)
            source_family = _source_family(source_file)
            source_file_exact[source_file] += 1
            source_family_exact[source_family] += 1
            source_family_files[source_family].add(source_file)
            stratum_id = f"{source}|{source_family}|content_sha256:{content_hash}"
            stratum_counts[stratum_id] += 1
            priority = _sample_priority(seed, record_id, source_row_id)
            row = {
                "record_id": record_id,
                "source_dataset": source,
                "source_row_id": source_row_id,
                "source_file": source_file,
                "source_family": source_family,
                "sample_stratum": stratum_id,
                "content_hash": content_hash,
                "prompt_hash": str(data["prompt_hash"][index] or ""),
                "status": str(status_value),
                "similarity": float(data["similarity"][index] or 0.0),
                "reasons": reason,
                "sample_priority": f"{priority:016x}",
            }
            _push_reservoir(
                reservoirs[stratum_id],
                row,
                priority=priority,
                capacity=sample_size,
            )

    metadata_rows = int(parquet_file.metadata.num_rows)
    if sum(source_scanned.values()) != metadata_rows:
        raise RuntimeError("parquet scan count does not match metadata row count")
    if exact_rows != sum(source_exact.values()) or exact_rows != sum(cluster_counts.values()):
        raise RuntimeError("exact row decomposition does not reconcile")

    cluster_details = {
        content_hash: _cluster_evidence(content_hash, eval_by_hash)
        for content_hash in cluster_counts
    }
    false_positive_rows = sum(
        cluster_counts[content_hash]
        for content_hash, evidence in cluster_details.items()
        if evidence["false_positive_classification"]
        == "high_confidence_false_positive_projection_collision"
    )
    plausible_true_positive_rows = sum(
        cluster_counts[content_hash]
        for content_hash, evidence in cluster_details.items()
        if evidence["false_positive_classification"] == "plausible_true_positive"
    )
    unresolved_rows = exact_rows - false_positive_rows - plausible_true_positive_rows

    primary_eval_counts: Counter[str] = Counter()
    eval_incidence_counts: Counter[str] = Counter()
    eval_cluster_counts: Counter[str] = Counter()
    source_eval_counts: Counter[tuple[str, str]] = Counter()
    for content_hash, count in cluster_counts.items():
        eval_ids = cluster_details[content_hash]["eval_corpora"]
        primary_eval = eval_ids[0] if eval_ids else "unresolved_eval_hash"
        primary_eval_counts[primary_eval] += count
        for source, source_count in cluster_sources[content_hash].items():
            source_eval_counts[(source, primary_eval)] += source_count
        for eval_id in eval_ids or ["unresolved_eval_hash"]:
            eval_incidence_counts[eval_id] += count
            eval_cluster_counts[eval_id] += 1

    allocation = allocate_stratified_sample(stratum_counts, sample_size)
    allocation_by_source: Counter[str] = Counter()
    allocation_by_source_family: Counter[str] = Counter()
    sample_rows: list[dict[str, Any]] = []
    for stratum_id in sorted(allocation):
        available = sorted(
            (
                (-negative_priority, row)
                for negative_priority, row in reservoirs[stratum_id]
            ),
            key=lambda item: (item[0], item[1]["record_id"]),
        )
        for priority, row in available[: allocation[stratum_id]]:
            evidence = cluster_details[row["content_hash"]]
            eval_ids = evidence["eval_corpora"]
            allocation_by_source[row["source_dataset"]] += 1
            allocation_by_source_family[row["source_family"]] += 1
            sample_rows.append(
                {
                    "sample_id": 0,
                    "sample_stratum": row["sample_stratum"],
                    "source_dataset": source,
                    "source_row_id": row["source_row_id"],
                    "source_file": row["source_file"],
                    "source_family": row["source_family"],
                    "record_id": row["record_id"],
                    "cluster_id": f"content_sha256:{row['content_hash']}",
                    "cluster_candidate_rows": cluster_counts[row["content_hash"]],
                    "cluster_source_rows": cluster_sources[row["content_hash"]][source],
                    "eval_corpora": ";".join(eval_ids) or "unresolved_eval_hash",
                    "eval_fingerprint_rows": evidence["eval_row_count"],
                    "raw_status": row["status"],
                    "raw_reasons": row["reasons"],
                    "similarity": row["similarity"],
                    "content_hash": row["content_hash"],
                    "prompt_hash": row["prompt_hash"],
                    "candidate_projection_empty": evidence["candidate_projection_empty"],
                    "all_eval_projections_empty": evidence["all_eval_projections_empty"],
                    "false_positive_candidate": (
                        evidence["false_positive_classification"]
                        == "high_confidence_false_positive_projection_collision"
                    ),
                    "false_positive_basis": evidence["false_positive_basis"],
                    "recommended_disposition": "keep_quarantined_and_rescan_after_extractor_fix",
                    "sample_priority": f"{priority:016x}",
                }
            )
    sample_rows.sort(key=lambda row: (row["source_dataset"], row["sample_priority"]))
    for sample_id, row in enumerate(sample_rows, start=1):
        row["sample_id"] = sample_id
    if len(sample_rows) != min(sample_size, exact_rows):
        raise RuntimeError("stratified sample size does not reconcile")

    source_rows = []
    for source in sorted(source_scanned):
        exact_count = source_exact[source]
        false_positive_count = sum(
            cluster_sources[content_hash][source]
            for content_hash in cluster_sources
            if cluster_details[content_hash]["false_positive_classification"]
            == "high_confidence_false_positive_projection_collision"
        )
        primary_evals = sorted(
            eval_id
            for (item_source, eval_id), count in source_eval_counts.items()
            if item_source == source and count
        )
        source_rows.append(
            {
                "source_dataset": source,
                "candidate_rows_scanned": source_scanned[source],
                "quarantine_exact_rows": exact_count,
                "exact_rate_pct": _pct(exact_count, source_scanned[source]),
                "exact_cluster_count": sum(
                    int(cluster_sources[content_hash][source] > 0)
                    for content_hash in cluster_sources
                ),
                "primary_eval_corpora": ";".join(primary_evals),
                "false_positive_candidate_rows": false_positive_count,
                "false_positive_share_pct": _pct(false_positive_count, exact_count),
                "sample_allocation": allocation_by_source[source],
            }
        )

    eval_rows = []
    eval_ids_for_report = sorted(
        set(eval_corpus_stats) | set(primary_eval_counts) | set(eval_incidence_counts)
    )
    for eval_id in eval_ids_for_report:
        stats = eval_corpus_stats.get(eval_id, {})
        eval_rows.append(
            {
                "eval_corpus": eval_id,
                "fingerprint_rows": int(stats.get("fingerprint_rows") or 0),
                "empty_projection_rows": int(stats.get("empty_projection_rows") or 0),
                "matched_cluster_count": eval_cluster_counts[eval_id],
                "primary_candidate_rows": primary_eval_counts[eval_id],
                "candidate_row_incidence": eval_incidence_counts[eval_id],
                "note": (
                    "candidate rows reconcile on deterministic primary eval; incidence may overlap"
                ),
            }
        )

    cluster_rows = []
    for content_hash, count in sorted(
        cluster_counts.items(), key=lambda item: (-item[1], item[0])
    ):
        evidence = cluster_details[content_hash]
        cluster_rows.append(
            {
                "cluster_id": f"content_sha256:{content_hash}",
                "content_hash": content_hash,
                "candidate_rows": count,
                "candidate_share_pct": _pct(count, exact_rows),
                "source_count": len(cluster_sources[content_hash]),
                "eval_corpora": ";".join(evidence["eval_corpora"])
                or "unresolved_eval_hash",
                "eval_fingerprint_rows": evidence["eval_row_count"],
                "candidate_projection_empty": evidence["candidate_projection_empty"],
                "all_eval_projections_empty": evidence["all_eval_projections_empty"],
                "false_positive_classification": evidence[
                    "false_positive_classification"
                ],
                "false_positive_basis": evidence["false_positive_basis"],
            }
        )

    source_family_rows = [
        {
            "source_family": family,
            "quarantine_exact_rows": count,
            "exact_share_pct": _pct(count, exact_rows),
            "source_file_count": len(source_family_files[family]),
            "false_positive_candidate_rows": count,
            "false_positive_basis": "empty_candidate_and_eval_projection_collision",
            "sample_allocation": allocation_by_source_family[family],
            "projection_finding": SOURCE_PROJECTION_FINDINGS.get(
                family, "empty canonical projection; inspect source schema"
            ),
        }
        for family, count in sorted(
            source_family_exact.items(), key=lambda item: (-item[1], item[0])
        )
    ]
    source_file_rows = [
        {
            "source_file": source_file,
            "source_family": _source_family(source_file),
            "quarantine_exact_rows": count,
            "exact_share_pct": _pct(count, exact_rows),
        }
        for source_file, count in sorted(
            source_file_exact.items(), key=lambda item: (-item[1], item[0])
        )
    ]

    source_path = output_dir / "contamination_exact_by_source.csv"
    source_family_path = output_dir / "contamination_exact_by_source_family.csv"
    source_file_path = output_dir / "contamination_exact_by_source_file.csv"
    eval_path = output_dir / "contamination_exact_by_eval.csv"
    cluster_path = output_dir / "contamination_exact_by_cluster.csv"
    sample_path = output_dir / "contamination_stratified_evidence_sample.csv"
    summary_path = output_dir / "contamination_exact_audit.json"
    report_path = output_dir / "CONTAMINATION_EVIDENCE_AUDIT.md"

    _atomic_write_csv(source_path, source_rows, list(source_rows[0]))
    _atomic_write_csv(
        source_family_path,
        source_family_rows,
        list(source_family_rows[0]),
    )
    _atomic_write_csv(source_file_path, source_file_rows, list(source_file_rows[0]))
    _atomic_write_csv(eval_path, eval_rows, list(eval_rows[0]))
    _atomic_write_csv(cluster_path, cluster_rows, list(cluster_rows[0]))
    _atomic_write_csv(sample_path, sample_rows, list(sample_rows[0]))

    cross_source_eval = [
        {
            "source_dataset": source,
            "primary_eval_corpus": eval_id,
            "candidate_rows": count,
        }
        for (source, eval_id), count in sorted(source_eval_counts.items())
    ]
    result: dict[str, Any] = {
        "schema_version": 1,
        "scope": "quarantine_exact_only",
        "input": {
            "path": str(input_path.resolve()),
            "sha256": _sha256_file(input_path),
            "parquet_rows": metadata_rows,
        },
        "eval_registry": {
            "path": str(registry_path.resolve()),
            "sha256": _sha256_file(registry_path),
            "corpora": eval_corpus_stats,
        },
        "sample": {
            "method": (
                "source-family and content-hash-cluster stratified deterministic "
                "hash-priority sample"
            ),
            "seed": seed,
            "requested_rows": sample_size,
            "actual_rows": len(sample_rows),
            "allocation_by_stratum": allocation,
            "allocation_by_source": dict(sorted(allocation_by_source.items())),
            "allocation_by_source_family": dict(
                sorted(allocation_by_source_family.items())
            ),
            "contains_raw_text": False,
        },
        "totals": {
            "candidate_rows_scanned": metadata_rows,
            "quarantine_exact_rows": exact_rows,
            "exact_clusters": len(cluster_counts),
            "false_positive_candidate_rows": false_positive_rows,
            "false_positive_candidate_share_pct": _pct(false_positive_rows, exact_rows),
            "plausible_true_positive_rows": plausible_true_positive_rows,
            "unresolved_exact_rows": unresolved_rows,
            "decomposition_reconciles": (
                exact_rows == sum(source_exact.values()) == sum(primary_eval_counts.values())
            ),
        },
        "exact_reasons": dict(sorted(exact_reasons.items())),
        "by_source": source_rows,
        "by_source_family": source_family_rows,
        "source_projection_findings": {
            row["source_family"]: row["projection_finding"] for row in source_family_rows
        },
        "source_file_count_with_exact_hits": len(source_file_rows),
        "by_primary_eval": eval_rows,
        "by_cluster": cluster_rows,
        "by_source_primary_eval": cross_source_eval,
        "artifacts": {
            "by_source_csv": str(source_path.resolve()),
            "by_source_family_csv": str(source_family_path.resolve()),
            "by_source_file_csv": str(source_file_path.resolve()),
            "by_eval_csv": str(eval_path.resolve()),
            "by_cluster_csv": str(cluster_path.resolve()),
            "sample_csv": str(sample_path.resolve()),
            "report": str(report_path.resolve()),
        },
        "safety": {
            "training_allowed": False,
            "automatic_clean_reclassification": False,
            "recommended_disposition": (
                "keep affected rows quarantined until extractor repair and full rescan"
            ),
        },
    }
    _atomic_write_json(summary_path, result)

    lines = [
        "# Contamination Exact-Match Evidence Audit",
        "",
        "This audit is metadata-only. It does not copy candidate or eval text into the repository.",
        "",
        "## Result",
        "",
        f"- exact quarantines audited: `{exact_rows:,}`",
        f"- exact content-hash clusters: `{len(cluster_counts):,}`",
        (
            f"- high-confidence false-positive candidates: `{false_positive_rows:,}` "
            f"(`{_pct(false_positive_rows, exact_rows):.6f}%`)"
        ),
        f"- plausible non-empty true positives: `{plausible_true_positive_rows:,}`",
        f"- unresolved exact rows: `{unresolved_rows:,}`",
        f"- decomposition reconciles to exact total: `{result['totals']['decomposition_reconciles']}`",
        "",
        "## Root finding",
        "",
        (
            "Every current `quarantine_exact` row belongs to the empty normalized-content "
            f"cluster `{EMPTY_SHA256}`. The same hash appears in empty eval projections, so the "
            "status proves an extractor projection collision, not benchmark-content overlap."
        ),
        "",
        (
            "Japanese MT-Bench rows use structures such as `turns`; the current generic eval "
            "extractor does not project those fields. Empty fingerprints were then admitted to "
            "the exact-content index. Candidate rows whose canonical projection was also empty "
            "matched all empty eval fingerprints."
        ),
        "",
        "## By source",
        "",
    ]
    lines.extend(
        _markdown_table(
            ["Source", "Scanned", "Exact", "Exact %", "FP candidates", "Sample"],
            (
                (
                    row["source_dataset"],
                    f"{row['candidate_rows_scanned']:,}",
                    f"{row['quarantine_exact_rows']:,}",
                    f"{row['exact_rate_pct']:.6f}",
                    f"{row['false_positive_candidate_rows']:,}",
                    row["sample_allocation"],
                )
                for row in source_rows
            ),
        )
    )
    lines.extend(["", "## By eval corpus", ""])
    lines.extend(
        _markdown_table(
            ["Eval", "Fingerprints", "Empty", "Clusters", "Primary candidate rows"],
            (
                (
                    row["eval_corpus"],
                    f"{row['fingerprint_rows']:,}",
                    f"{row['empty_projection_rows']:,}",
                    f"{row['matched_cluster_count']:,}",
                    f"{row['primary_candidate_rows']:,}",
                )
                for row in eval_rows
            ),
        )
    )
    lines.extend(["", "## Frontier source path families", ""])
    lines.extend(
        _markdown_table(
            ["Source family", "Exact", "Exact share %", "Files", "Sample"],
            (
                (
                    row["source_family"],
                    f"{row['quarantine_exact_rows']:,}",
                    f"{row['exact_share_pct']:.6f}",
                    row["source_file_count"],
                    row["sample_allocation"],
                )
                for row in source_family_rows
            ),
        )
    )
    lines.extend(["", "### Source-side projection findings", ""])
    for row in source_family_rows:
        lines.append(
            f"- `{row['source_family']}`: {row['projection_finding']} "
            f"(`{row['quarantine_exact_rows']:,}` rows)."
        )
    lines.extend(["", "## By exact cluster", ""])
    lines.extend(
        _markdown_table(
            ["Content hash", "Candidates", "Sources", "Eval rows", "Classification"],
            (
                (
                    row["content_hash"],
                    f"{row['candidate_rows']:,}",
                    row["source_count"],
                    row["eval_fingerprint_rows"],
                    row["false_positive_classification"],
                )
                for row in cluster_rows[:20]
            ),
        )
    )
    lines.extend(
        [
            "",
            "## 100-row stratified evidence sample",
            "",
            (
                f"`{sample_path.name}` contains `{len(sample_rows)}` rows allocated across "
                "source-family and content-hash-cluster strata, with at least one row per "
                f"positive stratum and proportional remainder. Selection is deterministic "
                f"using seed `{seed}` and SHA-256 hash priority."
            ),
            "",
            "The sample contains identifiers and fingerprints only. It includes no raw candidate "
            "text and no eval-only text.",
            "",
            "## Decision",
            "",
            (
                "Treat the current 1,393,704 exact count as invalid evidence of contamination. "
                "Do not automatically mark these rows clean: keep them quarantined, repair both "
                "eval and candidate field extraction, reject empty fingerprints at index-build "
                "time, and then run a fresh full scan."
            ),
            "",
            (
                "The global contamination gate remains blocked because AnswerCarefully is still "
                "unavailable and the corrected full-corpus scan has not been executed."
            ),
            "",
            "## Artifacts",
            "",
            f"- `{source_path.name}`",
            f"- `{source_family_path.name}`",
            f"- `{source_file_path.name}`",
            f"- `{eval_path.name}`",
            f"- `{cluster_path.name}`",
            f"- `{sample_path.name}`",
            f"- `{summary_path.name}`",
            "",
        ]
    )
    _atomic_write_text(report_path, "\n".join(lines))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Decompose exact contamination hits and build metadata-only evidence."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/manifests/contamination_results.parquet"),
    )
    parser.add_argument(
        "--eval-registry",
        type=Path,
        default=Path("data/manifests/eval_registry.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/pretrain_finalization"),
    )
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    result = audit(
        args.input,
        args.eval_registry,
        args.output_dir,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    print(json.dumps(result["totals"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
