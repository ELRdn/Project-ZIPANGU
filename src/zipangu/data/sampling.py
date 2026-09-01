"""Deterministic token-mass recipe planning and pilot selection."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, Mapping


def deterministic_key(record: Mapping[str, Any], seed: int = 3407) -> str:
    value = f"{seed}:{record.get('record_id')}:{record.get('content_hash')}:{record.get('source_row_id')}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def recipe_mass_plan(recipe: Mapping[str, Any], source_stats: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    sampling_mass = recipe.get("sampling_mass") if isinstance(recipe.get("sampling_mass"), Mapping) else {}
    buckets = recipe.get("buckets") if isinstance(recipe.get("buckets"), Mapping) else {}
    plan: dict[str, Any] = {"tokenizer_pending": True, "buckets": {}, "total_mass": 0.0}
    for bucket_name, mass in sampling_mass.items():
        bucket = buckets.get(bucket_name, {})
        candidates = bucket.get("candidates", []) if isinstance(bucket, Mapping) else []
        available = []
        for dataset_id in candidates:
            stats = source_stats.get(str(dataset_id), {})
            tokens = float(stats.get("eligible_token_count", 0) or 0)
            if tokens > 0:
                available.append((str(dataset_id), tokens))
        total_tokens = sum(tokens for _, tokens in available)
        bucket_plan = {
            "target_mass": float(mass),
            "available_token_mass": total_tokens,
            "candidates": [],
        }
        for dataset_id, tokens in available:
            share = float(mass) * tokens / total_tokens if total_tokens else 0.0
            bucket_plan["candidates"].append(
                {"dataset_id": dataset_id, "available_tokens": tokens, "target_mass": round(share, 8)}
            )
        plan["buckets"][str(bucket_name)] = bucket_plan
        plan["total_mass"] += float(mass) if isinstance(mass, (int, float)) else 0.0
    return plan


def select_by_token_mass(
    records: Iterable[Mapping[str, Any]],
    *,
    target_tokens: int,
    seed: int = 3407,
) -> list[dict[str, Any]]:
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    ordered = sorted(
        (dict(record) for record in records if record.get("eligibility") is True),
        key=lambda record: deterministic_key(record, seed),
    )
    selected: list[dict[str, Any]] = []
    token_total = 0
    for record in ordered:
        selected.append(record)
        token_total += int(record.get("token_count") or 0)
        if token_total >= target_tokens:
            break
    return selected


def storage_preflight(*, free_bytes: int, estimated_output_bytes: int, safety_ratio: float = 0.15) -> dict[str, Any]:
    if free_bytes < 0 or estimated_output_bytes < 0:
        raise ValueError("storage values must be non-negative")
    required = int(estimated_output_bytes * (1.0 + safety_ratio))
    return {
        "free_bytes": free_bytes,
        "estimated_output_bytes": estimated_output_bytes,
        "safety_reserve_bytes": required,
        "allowed": free_bytes >= required,
    }
