"""Audit-side provenance helpers; no registry status mutation is performed."""

from __future__ import annotations

from collections.abc import Mapping
import json
from typing import Any

from ..provenance import canonical_json_hash


def provenance_confidence(record: Mapping[str, Any]) -> str:
    revision = record.get("source_revision")
    license_value = record.get("license_raw")
    if revision not in (None, "", "unknown") and license_value not in (None, "", "unknown"):
        return "high"
    if revision not in (None, "") or license_value not in (None, ""):
        return "partial"
    return "unknown"


def audit_manifest_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Strip content while retaining row identity and audit decisions."""

    return {
        "record_id": record.get("record_id"),
        "source_dataset": record.get("source_dataset"),
        "source_config": record.get("source_config"),
        "source_split": record.get("source_split"),
        "source_row_id": record.get("source_row_id"),
        "source_path": (record.get("provenance_raw") or {}).get("source_path"),
        "source_revision": record.get("source_revision"),
        "category": record.get("category"),
        "sub_category": record.get("sub_category"),
        "language_user": record.get("language_user"),
        "language_assistant": record.get("language_assistant"),
        "language_reasoning": record.get("language_reasoning"),
        "language_overall": record.get("language_overall"),
        "teacher_model": record.get("teacher_model"),
        "license_raw": record.get("license_raw"),
        "vendor_specific": record.get("vendor_specific"),
        "reasoning_sft_ready": record.get("reasoning_sft_ready"),
        "tool_sft_ready": record.get("tool_sft_ready"),
        "char_count": record.get("char_count", 0),
        "token_count": record.get("token_count", 0),
        "tokenizer_pending": record.get("tokenizer_pending", True),
        "quality_score": record.get("quality_score", 0.0),
        "quality_breakdown": record.get("quality_breakdown", {}),
        "factuality_verified": record.get("factuality_verified", False),
        "content_hash": record.get("content_hash"),
        "simhash": record.get("simhash"),
        "provenance_confidence": provenance_confidence(record),
        "contamination_status": record.get("contamination_status"),
        "contamination_reasons": record.get("contamination_reasons", []),
        "eligibility": record.get("eligibility") is True,
        "exclusion_reasons": record.get("exclusion_reasons", []),
    }


def canonical_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    return canonical_json_hash(dict(manifest))


def dump_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
