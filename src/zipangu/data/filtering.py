"""Conservative hard filters.  They mark records; they never delete raw rows."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping


SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
    re.compile(r"\b(?:ghp|github_pat|xox[baprs])-[-A-Za-z0-9_]{12,}\b", re.I),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\b(?:api[_ -]?key|secret[_ -]?key|access[_ -]?token)\s*[:=]\s*['\"]?[A-Za-z0-9_\-/+=]{16,}", re.I),
)
PRIVATE_PATH_PATTERNS = (
    re.compile(r"[A-Z]:\\Users\\[^\\\s]+\\", re.I),
    re.compile(r"/home/[^/\s]+/"),
    re.compile(r"/Users/[^/\s]+/"),
)
REFUSAL_RE = re.compile(r"(?:I\s+can(?:not|'t)|申し訳ありません|お手伝いできません|対応できません)", re.I)
PLACEHOLDER_RE = re.compile(r"^(?:n/?a|none|null|todo|tbd|placeholder|未回答|回答なし)[.!。\s]*$", re.I)
BENCHMARK_RE = re.compile(r"(?:answercarefully|mt[- ]?bench|gsm8k|humaneval|mmlu|private holdout|評価用split)", re.I)
STACK_RE = re.compile(r"(?:Traceback \(most recent call last\)|Exception:|Error:\s*\w+Error)", re.I)


@dataclass(frozen=True)
class FilterResult:
    eligible: bool
    reasons: tuple[str, ...]


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return ""


def _combined_text(record: Mapping[str, Any]) -> str:
    parts = [
        _as_text(record.get("system")),
        _as_text(record.get("user")),
        _as_text(record.get("assistant_final")),
        _as_text(record.get("reasoning")),
    ]
    return "\n".join(part for part in parts if part)


def _metadata_text(record: Mapping[str, Any]) -> str:
    values = []
    for key in ("raw_metadata_json", "provenance_raw", "source_split", "source_config"):
        value = record.get(key)
        if value is not None:
            values.append(str(value))
    return "\n".join(values)


def _repeated_content(text: str, threshold: float) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 4:
        return False
    counts: dict[str, int] = {}
    for line in lines:
        counts[line] = counts.get(line, 0) + 1
    return max(counts.values()) / len(lines) >= threshold


def hard_filter(
    record: Mapping[str, Any],
    *,
    thresholds: Mapping[str, Any] | None = None,
    source_role: str = "train_candidate",
) -> FilterResult:
    config = thresholds or {}
    reasons: list[str] = []
    user = _as_text(record.get("user")).strip()
    assistant = _as_text(record.get("assistant_final")).strip()
    full_text = _combined_text(record)
    metadata_text = _metadata_text(record)

    if record.get("_parse_error") or record.get("conversion_error"):
        reasons.append("broken_json")
    if not user:
        reasons.append("empty_user")
    if not assistant:
        reasons.append("empty_assistant")
    if str(record.get("source_split", "")).casefold() in {"eval", "test", "validation"}:
        reasons.append("eval_split_not_train")
    if source_role in {"eval_only", "reference_only", "future_cpt_candidate"}:
        reasons.append("forbidden_source_role")

    if assistant and PLACEHOLDER_RE.fullmatch(assistant):
        reasons.append("placeholder_answer")
    if assistant and len(assistant) <= int(config.get("refusal_only_max_chars", 240)) and REFUSAL_RE.search(assistant):
        reasons.append("refusal_only")
    if assistant and STACK_RE.search(assistant) and len(assistant) < 1200:
        reasons.append("stack_trace_only")
    replacement_limit = float(config.get("max_replacement_char_ratio", 0.01))
    if full_text and full_text.count("\ufffd") / len(full_text) > replacement_limit:
        reasons.append("unicode_corruption")
    control_count = sum(1 for char in full_text if ord(char) < 32 and char not in "\n\r\t")
    if full_text and control_count / len(full_text) > float(config.get("max_control_char_ratio", 0.02)):
        reasons.append("control_character_corruption")
    if _repeated_content(assistant, float(config.get("repetition_line_ratio", 0.65))):
        reasons.append("meaningless_repetition")
    lower_assistant = assistant.casefold()
    if len(assistant) > 40 and lower_assistant.count("<html") > 0 and lower_assistant.count("</html") == 0:
        reasons.append("html_navigation_garbage")
    if any(pattern.search(full_text) for pattern in SECRET_PATTERNS):
        reasons.append("secret_like_string")
    private_hits = sum(len(pattern.findall(full_text)) for pattern in PRIVATE_PATH_PATTERNS)
    if private_hits >= int(config.get("private_path_match_threshold", 2)):
        reasons.append("private_path_exposure")
    explicit_eval = metadata_text.casefold()
    if bool(config.get("benchmark_signal_is_quarantine", True)) and BENCHMARK_RE.search(explicit_eval):
        reasons.append("benchmark_or_eval_signal")
    if source_role == "eval_only":
        reasons.append("eval_only_source")

    unique_reasons = tuple(dict.fromkeys(reasons))
    return FilterResult(eligible=not unique_reasons, reasons=unique_reasons)


def apply_filter(record: Mapping[str, Any], result: FilterResult) -> dict[str, Any]:
    updated = dict(record)
    updated["eligibility"] = result.eligible
    updated["exclusion_reasons"] = list(result.reasons)
    return updated
