"""The independent stages after source acquisition and tokenization."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
import heapq
from itertools import chain
import json
from pathlib import Path
from typing import Any

import yaml

from .core import (
    JP_HEAVY_BUCKETS,
    FinalizationBlocked,
    atomic_write_json,
    atomic_write_text,
    load_yaml,
    sha256_text,
)
from .runner import (
    BALANCED_BUCKETS,
    TRAINING_CANDIDATE_IDS,
    _available_eval_entries,
    _canonical,
    _iter_locations,
    _load_token_rows,
    _required_eval_missing,
    _write_stage_report,
)
from .selection import (
    candidate_manifest,
    fable_length_bucket,
    is_known_nemotron_split,
    nemotron_split_conflict,
    quality_threshold_comparison,
    stratified_sample_by_token_mass,
    validate_fable_trace,
    write_candidate_parquet,
)


def _candidate_rows_from_ids(runner: Any, ids: set[str], metadata: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    if not ids:
        return selected
    remaining = set(ids)
    for location, source_row in _iter_locations(runner.repo_root, runner.dataset_root):
        # record_id is defined solely by these two metadata fields.  Check it
        # before canonicalizing so reconstructing a few hundred selected rows
        # does not repeat expensive text/language/quality work for all 4.6M
        # source rows.
        identifier = sha256_text(
            f"{source_row.dataset_id}:{source_row.source_row_id}"
        )
        if identifier not in remaining:
            continue
        record = _canonical(location, source_row)
        if str(record.get("record_id")) != identifier:
            raise FinalizationBlocked(
                f"candidate record-id mismatch: {source_row.dataset_id}:{source_row.source_row_id}"
            )
        record.update(dict(metadata[identifier]))
        selected.append(record)
        remaining.remove(identifier)
        if not remaining:
            break
    if remaining:
        raise FinalizationBlocked(
            f"candidate source rows missing after selection: {len(remaining)}"
        )
    selected.sort(key=lambda row: str(row.get("record_id", "")))
    return selected

def _stage_nemotron(runner: Any) -> dict[str, Any]:
    rows_count = 0
    unknown_rows = 0
    japanese_rows = 0
    counts: Counter[str] = Counter()
    tokens: Counter[str] = Counter()
    quality_sum: Counter[str] = Counter()
    quality_count: Counter[str] = Counter()
    reasoning_languages = Counter()
    final_languages = Counter()
    hashes = Counter()
    for row in _load_token_rows(runner.temp_root):
        if row.get("source_dataset") != "nemotron_sft_multilingual_v2":
            continue
        rows_count += 1
        source_path = str(row.get("source_path", ""))
        source_split = str(row.get("source_split", ""))
        if (
            not is_known_nemotron_split(source_path, source_split)
            or nemotron_split_conflict(source_path, source_split)
        ):
            unknown_rows += 1
        bucket = str(row.get("nemotron_bucket") or "nemotron_non_ja")
        if bucket.startswith("nemotron_ja_"):
            japanese_rows += 1
        counts[bucket] += 1
        tokens[bucket] += int(row.get("training_formatted_tokens") or 0)
        quality_sum[bucket] += float(row.get("quality_score") or 0.0)
        quality_count[bucket] += 1
        reasoning_languages[str(row.get("language_reasoning") or "unknown")] += 1
        final_languages[str(row.get("language_assistant") or "unknown")] += 1
        content_hash = str(row.get("content_hash") or "")
        if content_hash:
            hashes[content_hash] += 1
    if not rows_count:
        raise FinalizationBlocked("no Nemotron token metadata is available")
    if unknown_rows:
        return {
            "status": "BLOCKED",
            "blocked_reasons": ["unknown_nemotron_split"],
            "details": {"rows": rows_count, "unknown_rows": unknown_rows},
        }
    duplicate_rows = sum(value - 1 for value in hashes.values() if value > 1)
    report = _write_stage_report(
        runner.repo_root,
        "NEMOTRON_JA_REPORT.md",
        [
            "# Nemotron Japanese Subset Report",
            "",
            "Classification is based on the filename/schema split. Unknown split or contradictory metadata is blocked; no language-based reclassification is inferred.",
            "",
            f"- total rows: {rows_count:,}",
            f"- Japanese rows: {japanese_rows:,}",
            f"- exact duplicate rate: {duplicate_rows / rows_count:.6%}",
            "",
            "| Split bucket | Rows | Formatted token mass | Mean quality |",
            "| --- | ---: | ---: | ---: |",
        ]
        + [
            f"| {bucket} | {counts[bucket]:,} | {tokens[bucket]:,} | {quality_sum[bucket] / quality_count[bucket] if quality_count[bucket] else 'n/a'} |"
            for bucket in sorted(counts)
        ]
        + [
            "",
            f"- reasoning language metadata: {json.dumps(dict(reasoning_languages), ensure_ascii=False, sort_keys=True)}",
            f"- final answer language metadata: {json.dumps(dict(final_languages), ensure_ascii=False, sort_keys=True)}",
        ],
    )
    summary_path = runner.temp_root / "finalization" / "nemotron_summary.json"
    summary = {
        "status": "PASS",
        "rows": rows_count,
        "japanese_rows": japanese_rows,
        "row_counts": dict(counts),
        "token_mass": dict(tokens),
        "quality_mean": {
            key: quality_sum[key] / quality_count[key] if quality_count[key] else None
            for key in quality_count
        },
        "reasoning_languages": dict(reasoning_languages),
        "final_languages": dict(final_languages),
        "exact_duplicate_rows": duplicate_rows,
    }
    atomic_write_json(summary_path, summary)
    return {"status": "PASS", "artifacts": [report, summary_path], "details": summary}

def _stage_fable(runner: Any) -> dict[str, Any]:
    token_rows = {
        str(row.get("record_id")): row
        for row in _load_token_rows(runner.temp_root)
        if row.get("source_dataset") == "fable_5_premium"
    }
    if not token_rows:
        raise FinalizationBlocked("no Fable token metadata is available")
    bucket_counts: Counter[str] = Counter()
    invalid: Counter[str] = Counter()
    segmentation = Counter()
    scanned = 0
    for location, source_row in _iter_locations(
        runner.repo_root,
        runner.dataset_root,
        dataset_ids=("fable_5_premium",),
    ):
        record = _canonical(location, source_row)
        metadata = token_rows.get(str(record.get("record_id")))
        if metadata is None:
            continue
        scanned += 1
        bucket = str(metadata.get("fable_bucket") or fable_length_bucket(int(metadata.get("training_formatted_tokens") or 0)))
        bucket_counts[bucket] += 1
        validation = validate_fable_trace(record)
        if not validation["valid"]:
            invalid.update(validation["errors"])
        if bucket in {"long", "extra_long"}:
            segmentation["long_trace_unsegmented"] += 1
    report = _write_stage_report(
        runner.repo_root,
        "FABLE_LONG_TRACE_REPORT.md",
        [
            "# Fable Premium Long-Trace Report",
            "",
            "Raw traces are not truncated. Extra-long traces are excluded from the 1M candidate unless a future reviewed segmentation pass proves turn-boundary and tool-pair integrity.",
            "",
            f"- rows inspected: `{scanned:,}`",
            f"- invalid trace rows: `{sum(invalid.values()):,}`",
            f"- validation errors: `{json.dumps(dict(invalid), ensure_ascii=False, sort_keys=True)}`",
            "",
            "| Bucket | Rows |",
            "| --- | ---: |",
        ]
        + [f"| `{bucket}` | {bucket_counts[bucket]:,} |" for bucket in ("short", "medium", "long", "extra_long")]
        + ["", f"- segmentation states: `{json.dumps(dict(segmentation), ensure_ascii=False, sort_keys=True)}`"],
    )
    summary_path = runner.temp_root / "finalization" / "fable_summary.json"
    summary = {
        "status": "PASS" if not invalid else "BLOCKED",
        "rows": scanned,
        "bucket_counts": dict(bucket_counts),
        "invalid_errors": dict(invalid),
        "segmentation": dict(segmentation),
        "extra_long_allowed_for_raw_candidate": False,
    }
    atomic_write_json(summary_path, summary)
    return {
        "status": "BLOCKED" if invalid else "PASS",
        "artifacts": [report, summary_path],
        "blocked_reasons": ["fable_trace_integrity_failure"] if invalid else [],
        "details": summary,
    }


def _stage_quality(runner: Any) -> dict[str, Any]:
    rows = _load_token_rows(runner.temp_root)
    try:
        first = next(rows)
    except StopIteration:
        raise FinalizationBlocked("no token metadata is available for quality thresholds")
    result = quality_threshold_comparison(chain((first,), rows))
    output = runner.repo_root / "reports" / "pretrain_finalization" / "i-quality-thresholds-recommendation.yaml"
    payload = {
        "schema_version": 1,
        "status": "draft_recommendation",
        "approval_required": True,
        "seed": int(runner.config.get("seed", 3407)),
        "thresholds": result.get("thresholds", []),
        "recommended_threshold": result.get("recommended_threshold"),
        "minimum_retention": result.get("minimum_retention", {}),
        "policy": {
            "diversity_entropy_ratio_min": 0.70,
            "contamination_exclusions": ["quarantine_exact", "quarantine_near"],
            "fable_extra_long_direct_inclusion": False,
            "ace_reasoning_sft_ready_required": True,
        },
    }
    atomic_write_text(output, yaml.safe_dump(payload, allow_unicode=True, sort_keys=False))
    report = _write_stage_report(
        runner.repo_root,
        "QUALITY_THRESHOLD_REPORT.md",
        [
            "# Quality Threshold Comparison",
            "",
            "This is a draft recommendation only. It does not approve a source or change the registry.",
            "",
            f"- recommended threshold: `{result.get('recommended_threshold') or 'none'}`",
            "",
            "| Threshold | Rows | Formatted mass | Diversity ratio | Missing minimum sources | Passes |",
            "| ---: | ---: | ---: | ---: | --- | --- |",
        ]
        + [
            f"| {item['threshold']} | {item['rows']:,} | {item['formatted_token_mass']:,} | {item['diversity_entropy_ratio']:.4f} | {', '.join(item['missing_minimum_sources']) or 'none'} | {item['passes']} |"
            for item in result["thresholds"]
        ],
    )
    return {"status": "PASS", "artifacts": [output, report], "details": result}


def _license_parts(license_raw: Any) -> list[str]:
    if isinstance(license_raw, (list, tuple, set)):
        return [str(item).casefold() for item in license_raw if str(item).strip()]
    return [str(license_raw or "unknown").casefold()]


def _is_mixed_license(license_raw: Any) -> bool:
    parts = _license_parts(license_raw)
    return len(parts) > 1 or any("mixed" in part for part in parts)


def _stage_license(runner: Any) -> dict[str, Any]:
    _, locations = runner._locations()
    candidates: list[dict[str, Any]] = []
    for location in locations:
        if location.policy.dataset_id not in TRAINING_CANDIDATE_IDS:
            continue
        card_license = location.card_metadata.get("license", location.policy.license_expected)
        revision_known = location.revision not in {"unknown", ""}
        license_known = not _is_mixed_license(card_license) and not (
            not _license_parts(card_license)
            or all(part in {"unknown", "", "none"} for part in _license_parts(card_license))
        )
        evidence_completeness = "package_metadata_present" if revision_known and license_known else "incomplete"
        candidates.append(
            {
                "dataset": location.policy.dataset_id,
                "subset": "dataset_root",
                "upstream_source": location.policy.repo,
                "revision": location.revision,
                "license_raw": card_license,
                "license_url_if_metadata_contains": location.card_metadata.get("license_url"),
                "redistribution_notes": "Human review required; no license conclusion is inferred by this runner.",
                "derivative_model_notes": "Training use is not authorized by this artifact.",
                "attribution_required": "unknown",
                "mixed_license": _is_mixed_license(card_license),
                "unknown_license": not _license_parts(card_license) or all(part in {"unknown", "", "none"} for part in _license_parts(card_license)),
                "evidence_completeness": evidence_completeness,
                "human_decision": None,
                "decided_by": None,
                "decided_at": None,
                "decision_rationale": None,
                "recommended_action": "HUMAN_DECISION_REQUIRED",
            }
        )
    json_path = runner.repo_root / "data" / "manifests" / "source_approval_candidates.json"
    atomic_write_json(
        json_path,
        {
            "schema_version": 2,
            "approval_required": True,
            "ai_may_set_human_decision": False,
            "sources": candidates,
        },
    )
    review_count = len(candidates)
    queue = _write_stage_report(
        runner.repo_root,
        "HUMAN_APPROVAL_QUEUE.md",
        [
            "# Human Approval Queue",
            "",
            "TRAINING_ALLOWED=false",
            "",
            "## WHAT",
            "",
            "Review the local training-source license and provenance candidates. This queue is advisory; it does not approve a source or change registry state.",
            "",
            "## EVIDENCE",
            "",
            "| Dataset | Upstream | Package revision | License metadata | Evidence | Human decision |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        + [
            f"| `{item['dataset']}` | `{item['upstream_source']}` | `{item['revision']}` | `{item['license_raw']}` | `{item['evidence_completeness']}` | _unset_ |"
            for item in candidates
        ]
        + [
            "",
            f"- human review required: `{review_count}`",
            "- human decisions recorded: `0`",
            "",
            "## HUMAN DECISION STATUS",
            "",
            "No source decision is recorded by this runner. Use `SOURCE_APPROVAL_MATRIX.md`; every choice remains human-owned and unset.",
            "",
            "## RISK",
            "",
            "Approving from repository metadata alone could authorize incompatible or untraceable training material. Final approval remains a human action outside this runner.",
        ],
    )
    return {"status": "PASS", "artifacts": [json_path, queue], "details": {"sources": len(candidates), "review": review_count, "human_decisions": 0}}


def _quality_threshold(runner: Any) -> int:
    payload: Any = None
    paths = (
        runner.repo_root / "configs" / "datasets" / "i-quality-thresholds.yaml",
        runner.repo_root / "configs" / "datasets" / "c-i-v1-quality-thresholds.yaml",
    )
    for path in paths:
        if not path.is_file():
            continue
        try:
            payload = load_yaml(path)
            break
        except (OSError, ValueError, yaml.YAMLError):
            payload = None
    value = payload.get("recommended_threshold") if isinstance(payload, Mapping) else None
    return int(value) if isinstance(value, (int, float)) else 75


def _source_pool(runner: Any, threshold: int, *, pool_limit: int = 8_000) -> dict[str, list[dict[str, Any]]]:
    pools: dict[str, list[tuple[int, int, dict[str, Any]]]] = defaultdict(list)
    seed = int(runner.config.get("seed", 3407))
    # Fable Premium contains tool-call traces.  Its current canonical form can
    # lose assistant messages whose content is empty but whose tool_calls are
    # semantically required.  Never admit those rows while the dedicated
    # integrity stage is blocked; a research candidate must fail closed rather
    # than silently train on an invalid role sequence.
    fable_integrity_passed = runner.runtime.stage("fable").get("status") == "PASS"
    sequence = 0
    for row in _load_token_rows(runner.temp_root):
        sequence += 1
        if row.get("training_formatted_tokens") is None:
            continue
        if str(row.get("source_split", "")).casefold() in {
            "eval",
            "test",
            "validation",
            "dev",
        }:
            continue
        if float(row.get("quality_score") or 0) < threshold:
            continue
        if str(row.get("contamination_status")) not in {
            "clean",
            "not_checked_missing_eval_source",
            "not_scanned",
        }:
            # Manual-review and quarantine hits are never auto-selected.
            continue
        if row.get("source_dataset") == "ace_reason_math_japanese" and row.get("reasoning_sft_ready") is not True:
            continue
        if row.get("source_dataset") == "fable_5_premium":
            if not fable_integrity_passed:
                continue
            if row.get("fable_bucket") != "short":
                # No automatic turn-boundary segmentation is performed here.
                continue
        bucket = str(row.get("selection_bucket") or "general_japanese")
        score = int(sha256_text(f"{seed}:{row.get('record_id', sequence)}"), 16)
        heapq.heappush(pools[bucket], (-score, sequence, row))
        if len(pools[bucket]) > pool_limit:
            heapq.heappop(pools[bucket])
    return {
        bucket: [item[2] for item in sorted(values, key=lambda item: (-item[0], item[1]))]
        for bucket, values in pools.items()
    }


def _write_candidate(
    runner: Any,
    *,
    name: str,
    bucket_targets: Mapping[str, float],
    pools: Mapping[str, list[dict[str, Any]]],
    contamination_global: str,
    license_status: str,
) -> tuple[Path, dict[str, Any]]:
    pool_rows = [row for values in pools.values() for row in values]
    selected_metadata, selection_summary = stratified_sample_by_token_mass(
        pool_rows,
        target_tokens=int(runner.config["candidate"]["target_tokens"]),
        bucket_targets=bucket_targets,
        seed=int(runner.config.get("seed", 3407)),
        template_cap=float(runner.config["candidate"].get("template_family_cap", 0.05)),
    )
    metadata_by_id = {str(row.get("record_id")): row for row in selected_metadata}
    selected_rows = _candidate_rows_from_ids(runner, set(metadata_by_id), metadata_by_id)
    output_dir = runner.shared_candidate_root / name
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = write_candidate_parquet(output_dir / "data.parquet", selected_rows)
    source_revisions = {
        str(row.get("source_dataset")): str(row.get("source_revision", "unknown"))
        for row in selected_rows
    }
    manifest = candidate_manifest(
        selected_rows,
        recipe_name=name,
        source_revisions=source_revisions,
        quality_thresholds={"recommended": _quality_threshold(runner)},
        dedup_config={"exact": True, "near": "simhash_lsh_bounded", "template_family_cap": 0.05},
        contamination_status=contamination_global,
        license_status=license_status,
        seed=int(runner.config.get("seed", 3407)),
        model_id=runner.model_id,
        base_model_id=runner.base_model_id,
    )
    manifest["selection"] = selection_summary
    manifest_path = output_dir / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    return output_dir, {"parquet": parquet_path, "manifest": manifest_path, "selection": selection_summary, "rows": selected_rows}


def _stage_candidate(runner: Any) -> dict[str, Any]:
    threshold = _quality_threshold(runner)
    pools = _source_pool(runner, threshold, pool_limit=int(runner.config["candidate"].get("pool_limit", 8_000)))
    if not pools:
        raise FinalizationBlocked("no eligible tokenized candidate pool is available")
    entries = _available_eval_entries(runner.repo_root)
    contamination_summary = runner._read_json(
        runner.temp_root / "finalization" / "contamination_summary.json",
        {},
    )
    contamination_global = str(contamination_summary.get("status") or "")
    if not contamination_global:
        contamination_global = (
            "not_checked_missing_eval_source"
            if _required_eval_missing(entries)
            else "not_scanned"
        )
    approval_path = runner.repo_root / "data" / "manifests" / "source_approval_candidates.json"
    approval = runner._read_json(approval_path, {})
    # Evidence collection never authorizes training. Human source decisions are
    # recorded separately and must not be inferred from card metadata here.
    license_status = "review_required"
    artifacts: list[Path] = []
    _, heavy = _write_candidate(
        runner,
        name="pilot-1m-jp-heavy",
        bucket_targets=JP_HEAVY_BUCKETS,
        pools=pools,
        contamination_global=contamination_global,
        license_status=license_status,
    )
    artifacts.extend([heavy["parquet"], heavy["manifest"]])
    target_tokens = int(runner.config["candidate"]["target_tokens"])

    def fill_status(selection: Mapping[str, Any]) -> dict[str, Any]:
        bucket_rows = selection.get("buckets", {})
        selected_tokens = sum(
            int(item.get("selected_tokens") or 0)
            for item in bucket_rows.values()
            if isinstance(item, Mapping)
        )
        severe_underfills = [
            str(bucket)
            for bucket, item in bucket_rows.items()
            if isinstance(item, Mapping)
            and int(item.get("selected_tokens") or 0)
            < int(item.get("target_tokens") or 0) * 0.90
        ]
        return {
            "target_tokens": target_tokens,
            "selected_tokens": selected_tokens,
            "fill_ratio": round(selected_tokens / target_tokens, 6) if target_tokens else 0.0,
            "severely_underfilled_buckets": severe_underfills,
            "status": (
                "PASS"
                if selected_tokens >= target_tokens * 0.95 and not severe_underfills
                else "BLOCKED"
            ),
        }

    heavy_fill = fill_status(heavy["selection"])
    japanese_buckets = {"general_japanese", "instruction_extraction", "japanese_math_reasoning", "japanese_stem_code"}
    frontier_buckets = {"frontier_reasoning", "agent_code_retention"}
    japanese_mass = sum(int(row.get("training_formatted_tokens") or 0) for bucket in japanese_buckets for row in pools.get(bucket, []))
    frontier_mass = sum(int(row.get("training_formatted_tokens") or 0) for bucket in frontier_buckets for row in pools.get(bucket, []))
    target_side = int(runner.config["candidate"]["target_tokens"]) // 2
    balanced_details: dict[str, Any] = {"japanese_pool_tokens": japanese_mass, "frontier_pool_tokens": frontier_mass, "target_each_side": target_side}
    if japanese_mass >= target_side and frontier_mass >= target_side:
        _, balanced = _write_candidate(
            runner,
            name="pilot-1m-balanced",
            bucket_targets=BALANCED_BUCKETS,
            pools=pools,
            contamination_global=contamination_global,
            license_status=license_status,
        )
        artifacts.extend([balanced["parquet"], balanced["manifest"]])
        balanced_fill = fill_status(balanced["selection"])
        balanced_details["selection"] = balanced["selection"]
        balanced_details["fill"] = balanced_fill
        balanced_details["status"] = balanced_fill["status"]
        if balanced_fill["status"] != "PASS":
            balanced_details["reason"] = "candidate_target_underfilled"
    else:
        balanced_details["status"] = "BLOCKED"
        balanced_details["reason"] = "japanese_or_frontier_side_underfilled"
    summary_path = runner.temp_root / "finalization" / "candidate_summary.json"
    atomic_write_json(
        summary_path,
        {
            "jp_heavy": heavy["selection"],
            "jp_heavy_fill": heavy_fill,
            "balanced": balanced_details,
        },
    )
    artifacts.append(summary_path)
    status = (
        "PASS"
        if heavy_fill["status"] == "PASS" and balanced_details["status"] == "PASS"
        else "BLOCKED"
    )
    blocked_reasons: list[str] = []
    if heavy_fill["status"] != "PASS":
        blocked_reasons.append("candidate_target_underfilled")
    if balanced_details["status"] != "PASS":
        blocked_reasons.append(str(balanced_details.get("reason") or "candidate_target_underfilled"))
    return {
        "status": status,
        "artifacts": artifacts,
        "blocked_reasons": sorted(set(blocked_reasons)),
        "details": {
            "jp_heavy": heavy["selection"],
            "jp_heavy_fill": heavy_fill,
            "balanced": balanced_details,
        },
    }
