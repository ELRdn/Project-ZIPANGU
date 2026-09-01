"""Local-only benchmark contamination checks."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping

from .dedup import normalized_record_text


DEFAULT_BENCHMARK_SIGNALS = (
    "answercarefully",
    "mt-bench",
    "mt bench",
    "llm-jp instruction",
    "private holdout",
    "gsm8k",
    "humaneval",
    "mmlu",
)


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _ngrams(text: str, size: int = 13) -> set[str]:
    normalized = _normalize(text)
    if len(normalized) <= size:
        return {normalized} if normalized else set()
    return {normalized[index : index + size] for index in range(len(normalized) - size + 1)}


def contamination_scan(
    candidates: Iterable[Mapping[str, Any]],
    evaluations: Iterable[Mapping[str, Any]] | None,
    *,
    benchmark_signals: Iterable[str] = DEFAULT_BENCHMARK_SIGNALS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return candidate status without calling the network or an LLM judge."""

    evaluation_list = [dict(item) for item in evaluations] if evaluations is not None else []
    if not evaluation_list:
        output = []
        for candidate in candidates:
            updated = dict(candidate)
            updated["contamination_status"] = "not_checked_missing_eval_source"
            output.append(updated)
        return output, {
            "status": "not_checked_missing_eval_source",
            "evaluation_records": 0,
            "exact_matches": 0,
            "substring_matches": 0,
            "ngram_matches": 0,
            "benchmark_signal_matches": 0,
        }

    eval_hashes = {str(item.get("content_hash")) for item in evaluation_list if item.get("content_hash")}
    eval_texts = [_normalize(normalized_record_text(item)) for item in evaluation_list]
    eval_ngrams: set[str] = set()
    for text in eval_texts:
        eval_ngrams.update(_ngrams(text))
    signals = tuple(signal.casefold() for signal in benchmark_signals)
    counts = Counter()
    output: list[dict[str, Any]] = []
    for candidate in candidates:
        updated = dict(candidate)
        text = _normalize(normalized_record_text(candidate))
        candidate_hash = str(candidate.get("content_hash") or "")
        reasons: list[str] = []
        if candidate_hash and candidate_hash in eval_hashes:
            reasons.append("exact_eval_match")
            counts["exact_matches"] += 1
        if text and any(text in eval_text or eval_text in text for eval_text in eval_texts if len(eval_text) >= 40):
            reasons.append("substring_eval_match")
            counts["substring_matches"] += 1
        candidate_grams = _ngrams(text)
        if candidate_grams and candidate_grams.intersection(eval_ngrams):
            reasons.append("ngram_eval_match")
            counts["ngram_matches"] += 1
        if any(signal in text for signal in signals):
            reasons.append("benchmark_name_signal")
            counts["benchmark_signal_matches"] += 1
        updated["contamination_status"] = "quarantine_contamination" if reasons else "clear"
        updated["contamination_reasons"] = reasons
        output.append(updated)
    return output, {
        "status": "clear" if not any(item.get("contamination_status") != "clear" for item in output) else "matches_found",
        "evaluation_records": len(evaluation_list),
        **counts,
    }


def benchmark_signal(text: str, signals: Iterable[str] = DEFAULT_BENCHMARK_SIGNALS) -> bool:
    lowered = text.casefold()
    return any(signal.casefold() in lowered for signal in signals)
