"""Deterministic, non-semantic quality scoring for audit triage."""

from __future__ import annotations

from collections import Counter
import math
import re
from typing import Any, Mapping


_WORD_RE = re.compile(r"\w+", re.UNICODE)
_BOILERPLATE_RE = re.compile(r"(?:もちろんです|お手伝いします|以下に|ご紹介します)", re.I)


def _text(record: Mapping[str, Any], field: str) -> str:
    value = record.get(field)
    return value if isinstance(value, str) else ""


def _bounded(value: float, maximum: float) -> float:
    return max(0.0, min(maximum, value))


def score_record(record: Mapping[str, Any]) -> dict[str, Any]:
    user = _text(record, "user").strip()
    assistant = _text(record, "assistant_final").strip()
    reasoning = _text(record, "reasoning").strip()
    messages = record.get("messages")
    roles = [item.get("role") for item in messages if isinstance(item, Mapping)] if isinstance(messages, list) else []
    structure_ok = bool(user and assistant and roles.count("user") >= 1 and roles.count("assistant") >= 1)
    structural = 20.0 if structure_ok else 0.0
    if structure_ok:
        structural -= min(8.0, max(0, len(roles) - len(set(range(len(roles))))) * 0.0)

    total_text = user + assistant + reasoning
    replacement_ratio = total_text.count("\ufffd") / max(1, len(total_text))
    lines = [line.strip() for line in assistant.splitlines() if line.strip()]
    repetition_penalty = 0.0
    if lines:
        repetition_penalty = min(10.0, max(Counter(lines).values()) / len(lines) * 10.0 - 2.0)
    linguistic = _bounded(20.0 - replacement_ratio * 500.0 - max(0.0, repetition_penalty), 20.0)

    answer_ratio = len(assistant) / max(1, len(user))
    completeness = 7.0
    if len(assistant) >= 40:
        completeness += 6.0
    if answer_ratio >= 0.35:
        completeness += 4.0
    if reasoning:
        completeness += 3.0
    completeness = _bounded(completeness, 20.0)

    unique_ratio = len(set(total_text)) / max(1, len(total_text))
    boilerplate = len(_BOILERPLATE_RE.findall(assistant))
    density = _bounded(15.0 * (0.45 + min(0.55, unique_ratio * 2.0)) - boilerplate * 0.8, 15.0)

    category = str(record.get("category", "")).casefold()
    useful_categories = ("general", "instruction", "extraction", "math", "reason", "stem", "code", "agent", "writing", "frontier")
    task_usefulness = 7.0 + (8.0 if any(token in category for token in useful_categories) else 3.0)
    if record.get("vendor_specific"):
        task_usefulness -= 1.0
    task_usefulness = _bounded(task_usefulness, 15.0)

    tools = record.get("tools")
    format_integrity = 10.0
    if tools is not None and not isinstance(tools, (list, dict, str)):
        format_integrity -= 5.0
    if any(role not in {"system", "user", "assistant", "tool", "developer"} for role in roles):
        format_integrity -= 5.0
    format_integrity = _bounded(format_integrity, 10.0)

    breakdown = {
        "structural_validity": round(structural, 3),
        "linguistic_cleanliness": round(linguistic, 3),
        "completeness": round(completeness, 3),
        "information_density": round(density, 3),
        "task_usefulness": round(task_usefulness, 3),
        "format_tool_integrity": round(format_integrity, 3),
    }
    total = round(sum(breakdown.values()), 3)
    return {
        "quality_score": total,
        "quality_breakdown": breakdown,
        "factuality_verified": False,
        "quality_method": "deterministic_structural_heuristic_v1",
    }


def quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)
