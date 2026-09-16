"""Special-source checks, quality recommendations, and token-mass selection."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from .core import FinalizationBlocked, JP_HEAVY_BUCKETS, MIN_RETENTION, QUALITY_THRESHOLDS


NEMOTRON_SPLITS = {
    "code_ja": "nemotron_ja_code",
    "math_ja": "nemotron_ja_math",
    "stem_ja": "nemotron_ja_stem",
}
FABLE_BUCKETS = ("short", "medium", "long", "extra_long")
CONTAMINATION_ELIGIBLE = {
    "clean",
    "not_checked_missing_eval_source",
    "not_scanned",
}


def classify_nemotron_split(source_path: str, source_split: str | None = None) -> str:
    value = str(source_split or "").casefold()
    if value in NEMOTRON_SPLITS:
        return NEMOTRON_SPLITS[value]
    lower = source_path.casefold().replace("-", "_")
    for split, bucket in NEMOTRON_SPLITS.items():
        if f"_{split}_" in f"_{lower}_" or lower.endswith(f"_{split}.jsonl") or lower.endswith(f"_{split}.parquet"):
            return bucket
    if any(token in lower for token in ("code_ja", "math_ja", "stem_ja")):
        for split, bucket in NEMOTRON_SPLITS.items():
            if split in lower:
                return bucket
    if value and value not in NEMOTRON_SPLITS:
        return "nemotron_non_ja"
    return "nemotron_non_ja"


def is_known_nemotron_split(source_path: str, source_split: str | None = None) -> bool:
    """Return whether filename/schema metadata identifies a supported split."""

    value = str(source_split or "").casefold()
    if value in NEMOTRON_SPLITS or re.fullmatch(r"(?:code|math|stem)_[a-z]{2}", value):
        return True
    lower = source_path.casefold().replace("-", "_")
    if any(f"_{split}_" in f"_{lower}_" for split in NEMOTRON_SPLITS):
        return True
    return bool(
        re.search(
            r"(?:^|[_/])(?:code|math|stem)_[a-z]{2}(?:[_/.-]|$)",
            lower,
        )
    )


def nemotron_split_conflict(source_path: str, source_split: str | None = None) -> bool:
    """Detect contradictory language/category metadata in a Nemotron file."""

    lower = source_path.casefold().replace("-", "_")
    path_labels = {
        match.group(1) + "_" + match.group(2)
        for match in re.finditer(
            r"(?:^|[_/])(code|math|stem)_([a-z]{2})(?:[_/.-]|$)",
            lower,
        )
    }
    if len(path_labels) > 1:
        return True
    value = str(source_split or "").casefold()
    if path_labels and (
        value in NEMOTRON_SPLITS
        or re.fullmatch(r"(?:code|math|stem)_[a-z]{2}", value)
    ):
        return value not in path_labels
    return False


def fable_length_bucket(formatted_tokens: int) -> str:
    value = int(formatted_tokens)
    if value <= 4_096:
        return "short"
    if value <= 8_192:
        return "medium"
    if value <= 32_768:
        return "long"
    return "extra_long"


def _roles(messages: Any) -> list[str]:
    return [str(item.get("role", "")).casefold() for item in messages if isinstance(item, Mapping)] if isinstance(messages, list) else []


def validate_fable_trace(record: Mapping[str, Any]) -> dict[str, Any]:
    messages = record.get("messages")
    roles = _roles(messages)
    errors: list[str] = []
    if not messages or not roles:
        errors.append("messages_missing")
    if any(role not in {"system", "user", "assistant", "tool"} for role in roles):
        errors.append("unknown_role")
    if "user" not in roles:
        errors.append("user_missing")
    if "assistant" not in roles:
        errors.append("assistant_missing")
    for index, role in enumerate(roles):
        if role == "tool" and (index == 0 or roles[index - 1] != "assistant"):
            errors.append("tool_without_preceding_assistant")
    if roles and roles[-1] == "tool":
        errors.append("trace_ends_with_tool_result")
    return {
        "valid": not errors,
        "role_sequence": roles,
        "tool_pair_integrity": not any("tool_" in error or error == "trace_ends_with_tool_result" for error in errors),
        "assistant_supervision": "assistant" in roles,
        "errors": errors,
    }


def safe_segment_trace(
    messages: Sequence[Mapping[str, Any]],
    *,
    max_tokens: int,
    token_counter: Any,
) -> tuple[list[list[Mapping[str, Any]]] | None, str]:
    """Segment only at complete message boundaries; never truncate strings."""

    if not messages:
        return None, "messages_missing"
    segments: list[list[Mapping[str, Any]]] = []
    current: list[Mapping[str, Any]] = []
    for message in messages:
        current.append(message)
        try:
            count = int(
                token_counter.count_record({"messages": current}).get(
                    "training_formatted_tokens"
                )
                or 0
            )
        except Exception:
            return None, "tokenization_failed"
        while count > max_tokens:
            roles = _roles(current)
            safe_cuts = [
                index
                for index in range(1, len(current))
                if roles[index - 1] in {"assistant", "tool"} and roles[index] != "tool"
            ]
            if not safe_cuts:
                return None, "long_trace_unsegmented"
            cut = max(safe_cuts)
            segments.append(current[:cut])
            current = current[cut:]
            try:
                count = int(
                    token_counter.count_record({"messages": current}).get(
                        "training_formatted_tokens"
                    )
                    or 0
                )
            except Exception:
                return None, "tokenization_failed"
            if not current:
                return None, "long_trace_unsegmented"
    if current:
        segments.append(current)
    return segments, "segmented" if len(segments) > 1 else "single_turn_boundary_safe"

def selection_bucket(record: Mapping[str, Any]) -> str:
    source = str(record.get("source_dataset", record.get("dataset", ""))).casefold()
    category = str(record.get("category", "")).casefold()
    special = str(record.get("nemotron_bucket", "")).casefold()
    if source == "fable_5_premium" or "agent" in category or "coding_agent" in category:
        return "agent_code_retention"
    if source in {"gpt_5_6_traces", "fable_5_5_distillation", "frontier_multi_teacher"} or "frontier" in category:
        return "frontier_reasoning"
    if special == "nemotron_ja_code" or "code" in category or "stem" in category:
        return "japanese_stem_code"
    if special == "nemotron_ja_math" or "math" in category or "reason" in category:
        return "japanese_math_reasoning"
    if "instruction" in category or "extraction" in category or source in {"magpie_sft_v1", "extraction_wiki_ja"}:
        return "instruction_extraction"
    return "general_japanese"


def _entropy(values: Mapping[str, float]) -> float:
    total = sum(values.values())
    if total <= 0:
        return 0.0
    return -sum((value / total) * math.log(value / total) for value in values.values() if value > 0)


def quality_threshold_comparison(
    rows: Iterable[Mapping[str, Any]],
    *,
    thresholds: Sequence[int] = QUALITY_THRESHOLDS,
    minimum_retention: Mapping[str, float] = MIN_RETENTION,
) -> dict[str, Any]:
    """Compare thresholds in one streaming pass without retaining all rows."""

    threshold_values = sorted({int(item) for item in thresholds}, reverse=True)
    total_by_source: Counter[str] = Counter()
    eligible_templates: Counter[str] = Counter()
    selected_rows: Counter[int] = Counter()
    selected_mass: Counter[int] = Counter()
    mass_by_threshold: dict[int, Counter[str]] = {
        threshold: Counter() for threshold in threshold_values
    }
    templates_by_threshold: dict[int, Counter[str]] = {
        threshold: Counter() for threshold in threshold_values
    }
    for raw in rows:
        if not isinstance(raw, Mapping):
            continue
        formatted = raw.get("training_formatted_tokens")
        if formatted is None:
            continue
        row = dict(raw)
        if str(row.get("source_split", "")).casefold() in {
            "eval",
            "test",
            "validation",
            "dev",
        }:
            continue
        mass = int(formatted or 0)
        source = str(row.get("source_dataset", row.get("dataset", "unknown")))
        contamination = str(row.get("contamination_status", "clean"))
        total_by_source[source] += mass
        if contamination in CONTAMINATION_ELIGIBLE:
            template = str(row.get("template_family_hash", "unknown"))
            eligible_templates[template] += mass
        quality = float(row.get("quality_score") or 0)
        if contamination not in CONTAMINATION_ELIGIBLE:
            continue
        template = str(row.get("template_family_hash", "unknown"))
        for threshold in threshold_values:
            if quality < threshold:
                continue
            selected_rows[threshold] += 1
            selected_mass[threshold] += mass
            mass_by_threshold[threshold][source] += mass
            templates_by_threshold[threshold][template] += mass

    eligible_entropy = _entropy(eligible_templates)
    comparisons: list[dict[str, Any]] = []
    for threshold in threshold_values:
        mass_by_source = mass_by_threshold[threshold]
        retention = {
            source: round(mass_by_source[source] / total_by_source[source], 6)
            if total_by_source[source]
            else 0.0
            for source in sorted(total_by_source)
        }
        missing_minimum = [
            source
            for source, minimum in minimum_retention.items()
            if source in total_by_source and retention.get(source, 0.0) < float(minimum)
        ]
        selected_entropy = _entropy(templates_by_threshold[threshold])
        diversity_ratio = selected_entropy / eligible_entropy if eligible_entropy else (1.0 if selected_entropy == 0.0 and selected_rows[threshold] else 0.0)
        comparisons.append(
            {
                "threshold": threshold,
                "rows": selected_rows[threshold],
                "formatted_token_mass": selected_mass[threshold],
                "retention_by_source": retention,
                "missing_minimum_sources": missing_minimum,
                "diversity_entropy": round(selected_entropy, 6),
                "eligible_entropy": round(eligible_entropy, 6),
                "diversity_entropy_ratio": round(diversity_ratio, 6),
                "passes": not missing_minimum and diversity_ratio >= 0.70,
            }
        )
    recommendation = next((item["threshold"] for item in comparisons if item["passes"]), None)
    return {
        "thresholds": comparisons,
        "recommended_threshold": recommendation,
        "recommendation_status": "draft_recommendation",
        "approval_required": True,
        "minimum_retention": dict(minimum_retention),
    }

def _stable_score(row: Mapping[str, Any], seed: int) -> str:
    identity = str(row.get("record_id") or row.get("source_row_id") or row.get("content_hash") or "")
    return hashlib.sha256(f"{seed}:{identity}".encode("utf-8")).hexdigest()


def _group_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        str(row.get("source_dataset", row.get("dataset", "unknown"))),
        str(row.get("selection_bucket", selection_bucket(row))),
        str(row.get("category", "unknown")),
        str(row.get("task_type", "unknown")),
        str(row.get("language", row.get("language_overall", "unknown"))),
        str(row.get("teacher_model", row.get("teacher", "unknown"))),
        str(row.get("prompt_length_bin", "unknown")),
        str(row.get("answer_length_bin", "unknown")),
        str(row.get("template_family_hash", "unknown")),
    )


def stratified_sample_by_token_mass(
    rows: Iterable[Mapping[str, Any]],
    *,
    target_tokens: int = 1_000_000,
    bucket_targets: Mapping[str, float] = JP_HEAVY_BUCKETS,
    seed: int = 3407,
    template_cap: float = 0.05,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select by token mass using deterministic strata and hash order.

    Hash order gives a reproducible sample without sorting by quality.  The
    group round-robin keeps source/category/language strata represented, and
    the template cap is enforced independently inside each target bucket.
    """

    grouped: dict[str, dict[tuple[str, ...], list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for raw in rows:
        row = dict(raw)
        bucket = str(row.get("selection_bucket") or selection_bucket(row))
        row["selection_bucket"] = bucket
        row["_stable_score"] = _stable_score(row, seed)
        grouped[bucket][_group_key(row)].append(row)
    for bucket_groups in grouped.values():
        for group_rows in bucket_groups.values():
            group_rows.sort(key=lambda item: (str(item["_stable_score"]), str(item.get("record_id", ""))))

    selected: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"target_tokens": int(target_tokens), "seed": seed, "buckets": {}, "underfilled_buckets": []}
    for bucket, ratio in bucket_targets.items():
        target = int(round(target_tokens * float(ratio)))
        groups = [
            bucket_groups
            for key, bucket_groups in sorted(grouped.get(bucket, {}).items(), key=lambda item: str(item[0]))
        ]
        pointers = [0] * len(groups)
        bucket_rows: list[dict[str, Any]] = []
        bucket_mass = 0
        family_mass: Counter[str] = Counter()
        cap = max(1, int(target * template_cap))
        while groups and bucket_mass < target:
            progressed = False
            for index, group_rows in enumerate(groups):
                if pointers[index] >= len(group_rows):
                    continue
                candidate = group_rows[pointers[index]]
                pointers[index] += 1
                progressed = True
                mass = int(candidate.get("training_formatted_tokens") or candidate.get("formatted_tokens") or 0)
                family = str(candidate.get("template_family_hash", "unknown"))
                if (
                    mass <= 0
                    or bucket_mass + mass > target
                    or family_mass[family] + mass > cap
                ):
                    continue
                bucket_rows.append(candidate)
                bucket_mass += mass
                family_mass[family] += mass
                if bucket_mass >= target:
                    break
            if not progressed:
                break
        selected.extend(bucket_rows)
        summary["buckets"][bucket] = {
            "target_tokens": target,
            "selected_rows": len(bucket_rows),
            "selected_tokens": bucket_mass,
            "underfilled": bucket_mass < target,
            "template_mass": dict(family_mass),
        }
        if bucket_mass < target:
            summary["underfilled_buckets"].append(bucket)
    for row in selected:
        row.pop("_stable_score", None)
    return selected, summary


def candidate_manifest(
    rows: Iterable[Mapping[str, Any]],
    *,
    recipe_name: str,
    source_revisions: Mapping[str, str],
    quality_thresholds: Mapping[str, Any],
    dedup_config: Mapping[str, Any],
    contamination_status: str,
    license_status: str,
    format_validation_status: str = "PENDING",
    seed: int = 3407,
    model_id: str = "ZIPANGU-K-I-4B",
    base_model_id: str | None = None,
) -> dict[str, Any]:
    values = [dict(row) for row in rows]
    source_mass: Counter[str] = Counter()
    category_mass: Counter[str] = Counter()
    for row in values:
        mass = int(row.get("training_formatted_tokens") or row.get("formatted_tokens") or 0)
        source_mass[str(row.get("source_dataset", row.get("dataset", "unknown")))] += mass
        category_mass[str(row.get("category", "unknown"))] += mass
    total = sum(source_mass.values())
    return {
        "schema_version": 1,
        "recipe": recipe_name,
        "recipe_generation": "I",
        "model_id": model_id,
        "base_model": base_model_id,
        "canonical_candidate": True,
        "TRAINING_ALLOWED": False,
        "HUMAN_APPROVAL_REQUIRED": True,
        "status": "UNAPPROVED_RESEARCH_CANDIDATE",
        "row_count": len(values),
        "exact_formatted_token_count": total,
        "source_token_mass": dict(source_mass),
        "source_token_percent": {key: round(value / total, 6) if total else 0.0 for key, value in sorted(source_mass.items())},
        "category_token_mass": dict(category_mass),
        "seed": seed,
        "source_revisions": dict(source_revisions),
        "quality_thresholds": dict(quality_thresholds),
        "dedup_config": dict(dedup_config),
        "contamination_status": contamination_status,
        "license_status": license_status,
        "format_validation_status": format_validation_status,
    }


def write_candidate_parquet(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    try:
        import pyarrow as pa
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise FinalizationBlocked("candidate Parquet output requires pyarrow") from exc
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        messages = row.get("messages")
        tools = row.get("tools")
        output_rows.append(
            {
                "record_id": row.get("record_id"),
                "source_dataset": row.get("source_dataset", row.get("dataset")),
                "source_row_id": row.get("source_row_id"),
                "source_config": row.get("source_config"),
                "source_split": row.get("source_split"),
                "category": row.get("category"),
                "selection_bucket": row.get("selection_bucket"),
                "language": row.get("language", row.get("language_overall")),
                "teacher_model": row.get("teacher_model", row.get("teacher")),
                "system": row.get("system", ""),
                "user": row.get("user", ""),
                "assistant_final": row.get("assistant_final", ""),
                "reasoning": row.get("reasoning", ""),
                "messages_json": json.dumps(messages, ensure_ascii=False, separators=(",", ":")),
                "tools_json": json.dumps(tools, ensure_ascii=False, separators=(",", ":")) if tools is not None else None,
                "training_formatted_tokens": int(row.get("training_formatted_tokens") or row.get("formatted_tokens") or 0),
                "supervised_assistant_tokens": row.get("supervised_assistant_tokens"),
                "quality_score": row.get("quality_score"),
                "content_hash": row.get("content_hash"),
                "contamination_status": row.get("contamination_status"),
                "template_family_hash": row.get("template_family_hash"),
                "fable_bucket": row.get("fable_bucket"),
                "reasoning_sft_ready": row.get("reasoning_sft_ready"),
                "tool_sft_ready": row.get("tool_sft_ready"),
            }
        )
    table = pa.Table.from_pylist(output_rows)
    parquet.write_table(table, destination, compression="zstd")
    return destination


def validate_training_record(record: Mapping[str, Any], *, max_sequence_length: int = 4_096) -> dict[str, Any]:
    messages = record.get("messages")
    if messages is None and record.get("messages_json"):
        try:
            messages = json.loads(str(record["messages_json"]))
        except json.JSONDecodeError:
            messages = None
    roles = _roles(messages)
    errors: list[str] = []
    if "assistant" not in roles:
        errors.append("assistant_role_missing")
    if "user" not in roles:
        errors.append("user_role_missing")
    if any(role not in {"system", "user", "assistant", "tool"} for role in roles):
        errors.append("unknown_role")
    assistant_values = [
        str(item.get("content", "")) for item in messages or []
        if isinstance(item, Mapping) and str(item.get("role", "")).casefold() == "assistant"
    ]
    if not any(value.strip() for value in assistant_values):
        errors.append("final_answer_missing")
    if any(value.lower().count("<think>") != value.lower().count("</think>") for value in assistant_values):
        errors.append("think_tag_unbalanced")
    if str(record.get("assistant_final", "")).strip() == "":
        errors.append("final_answer_field_missing")
    for index, role in enumerate(roles):
        if role == "tool" and (index == 0 or roles[index - 1] != "assistant"):
            errors.append("tool_result_pair_missing")
    formatted = record.get("training_formatted_tokens", record.get("formatted_tokens"))
    formatted_value = int(formatted) if formatted is not None else None
    if formatted_value is not None and formatted_value > max_sequence_length:
        errors.append("max_sequence_exceeded_without_safe_segmentation")
    if str(record.get("system", "")).count("<|im_start|>") > 1 or str(record.get("assistant_final", "")).count("<|im_end|>") > 1:
        errors.append("duplicated_bos_or_eos_marker")
    supervised = record.get("supervised_assistant_tokens")
    return {
        "valid": not errors,
        "errors": sorted(set(errors)),
        "roles": roles,
        "formatted_tokens": formatted_value,
        "supervised_tokens": int(supervised) if supervised is not None else None,
        "supervision_available": supervised is not None,
        "truncated": False,
    }


def validate_training_records(records: Iterable[Mapping[str, Any]], *, max_sequence_length: int = 4_096) -> dict[str, Any]:
    total = 0
    valid = 0
    errors: Counter[str] = Counter()
    supervision_values: list[float] = []
    truncation = 0
    for record in records:
        total += 1
        result = validate_training_record(record, max_sequence_length=max_sequence_length)
        if result["valid"]:
            valid += 1
        errors.update(result["errors"])
        if result["supervision_available"] and result["formatted_tokens"]:
            supervision_values.append(result["supervised_tokens"] / result["formatted_tokens"])
        truncation += int(result["truncated"])
    return {
        "status": "PASS" if total and valid == total else "BLOCKED",
        "rows": total,
        "valid_rows": valid,
        "invalid_rows": total - valid,
        "error_counts": dict(errors),
        "truncation_rate": truncation / total if total else 0.0,
        "supervision_token_rate_mean": sum(supervision_values) / len(supervision_values) if supervision_values else None,
        "sample_policy": "deterministic first 100-500 rows after hash ordering",
    }
