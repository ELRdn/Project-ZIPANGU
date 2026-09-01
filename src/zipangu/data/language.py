"""Dependency-free language and token-length heuristics."""

from __future__ import annotations

import math
import re
from typing import Any, Mapping


_LATIN_RE = re.compile(r"[A-Za-z]")
_HIRAGANA_RE = re.compile(r"[\u3040-\u309f]")
_KATAKANA_RE = re.compile(r"[\u30a0-\u30ff\u31f0-\u31ff]")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_CODE_RE = re.compile(r"```|</?[A-Za-z][^>]*>|\b(def|class|import|function|const|SELECT|FROM)\b|[{};]", re.I)


def estimate_tokens(text: str) -> int:
    """Conservative pre-tokenizer estimate; never downloads the base tokenizer."""

    if not text:
        return 0
    return max(1, math.ceil(len(text) / 2.0))


def _char_stats(text: str) -> dict[str, int]:
    compact = "".join(text.split())
    total = len(compact)
    return {
        "total": total,
        "latin": len(_LATIN_RE.findall(compact)),
        "hiragana": len(_HIRAGANA_RE.findall(compact)),
        "katakana": len(_KATAKANA_RE.findall(compact)),
        "cjk": len(_CJK_RE.findall(compact)),
        "code_markers": len(_CODE_RE.findall(text)),
    }


def classify_text(text: str, metadata_language: Any = None) -> tuple[str, dict[str, float]]:
    stats = _char_stats(text)
    total = max(1, stats["total"])
    japanese_signal = (stats["hiragana"] + stats["katakana"]) / total
    cjk_signal = stats["cjk"] / total
    english_signal = stats["latin"] / total
    code_signal = min(1.0, stats["code_markers"] / max(1, total / 80))
    metadata = str(metadata_language or "").casefold()
    metadata_ja = metadata in {"ja", "jp", "japanese"} or "ja" in metadata.split("-")
    metadata_en = metadata in {"en", "english"} or "en" in metadata.split("-")

    if metadata_ja and (japanese_signal >= 0.03 or cjk_signal >= 0.08):
        label = "mixed" if english_signal >= 0.15 else "japanese"
    elif japanese_signal >= 0.08 or (japanese_signal >= 0.03 and cjk_signal >= 0.15):
        label = "mixed" if english_signal >= 0.15 else "japanese"
    elif code_signal >= 0.45 and english_signal >= 0.10:
        label = "code-heavy"
    elif metadata_en or english_signal >= 0.35:
        label = "english"
    else:
        label = "unknown"

    return label, {
        "japanese": round(min(1.0, japanese_signal + (cjk_signal * 0.25)), 6),
        "english": round(min(1.0, english_signal), 6),
        "code": round(code_signal, 6),
    }


def analyze_record_languages(record: Mapping[str, Any], metadata_language: Any = None) -> dict[str, Any]:
    texts = {
        "user": str(record.get("user") or ""),
        "assistant": str(record.get("assistant_final") or ""),
        "reasoning": str(record.get("reasoning") or ""),
    }
    labels: dict[str, str] = {}
    for field, text in texts.items():
        labels[field], _ = classify_text(text, metadata_language)
    whole = "\n".join(text for text in texts.values() if text)
    overall, ratios = classify_text(whole, metadata_language)
    chars = max(1, len("".join(whole.split())))
    japanese_chars = len(_HIRAGANA_RE.findall(whole)) + len(_KATAKANA_RE.findall(whole))
    english_chars = len(_LATIN_RE.findall(whole))
    return {
        "language_user": labels["user"],
        "language_assistant": labels["assistant"],
        "language_reasoning": labels["reasoning"] if texts["reasoning"] else "unknown",
        "language_overall": overall,
        "japanese_ratio": round(min(1.0, japanese_chars / chars), 6),
        "english_ratio": round(min(1.0, english_chars / chars), 6),
        "code_ratio": ratios["code"],
    }
