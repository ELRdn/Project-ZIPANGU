"""Conversion from heterogeneous local rows to the ZIPANGU canonical shape."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Any, Mapping, Sequence

from ..provenance import canonical_json_hash
from .adapters import SourceRow
from .language import analyze_record_languages, estimate_tokens


CANONICAL_FIELDS = (
    "record_id",
    "source_dataset",
    "source_config",
    "source_split",
    "source_row_id",
    "source_revision",
    "category",
    "sub_category",
    "language_user",
    "language_assistant",
    "language_reasoning",
    "language_overall",
    "system",
    "user",
    "assistant_final",
    "reasoning",
    "messages",
    "tools",
    "teacher_model",
    "provenance_raw",
    "license_raw",
    "vendor_specific",
    "reasoning_sft_ready",
    "tool_sft_ready",
    "char_count",
    "token_count",
    "tokenizer_pending",
    "quality_score",
    "quality_breakdown",
    "factuality_verified",
    "quality_method",
    "content_hash",
    "simhash",
    "duplicate_cluster",
    "contamination_status",
    "eligibility",
    "exclusion_reasons",
    "raw_metadata_json",
)


_CONTENT_KEYS = frozenset(
    {
        "messages",
        "conversations",
        "conversation",
        "dialogue",
        "row_json",
        "input",
        "instruction",
        "prompt",
        "question",
        "problem",
        "user",
        "output",
        "response",
        "answer",
        "solution",
        "assistant",
        "reasoning",
        "reasoning_content",
        "r1_generation",
        "analysis",
        "thought",
        "_event_stream",
    }
)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.I | re.S)
_ANALYSIS_RE = re.compile(r"<analysis>(.*?)</analysis>", re.I | re.S)


def _json_value(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8", errors="replace")
        except Exception:
            return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("{", "[")):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    return value


def _text(value: Any) -> str:
    value = _json_value(value)
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        for key in ("content", "text", "value", "output", "answer", "solution"):
            if key in value:
                return _text(value[key])
    if isinstance(value, list):
        return "\n".join(part for part in (_text(item) for item in value) if part)
    return ""


def _role(value: Any) -> str:
    role = str(value or "").casefold()
    return {"developer": "system", "human": "user", "model": "assistant"}.get(role, role)


def _message_from_item(item: Any) -> tuple[str, str] | None:
    if not isinstance(item, Mapping):
        return None
    role = _role(item.get("role") or item.get("author") or item.get("from") or item.get("type"))
    if role in {"message", "text", "content"}:
        role = _role(item.get("sender") or item.get("speaker"))
    if role == "tool_result":
        role = "tool"
    content = _text(item.get("content", item.get("text", item.get("message", item.get("value")))))
    if role not in {"system", "user", "assistant", "tool"} or not content:
        return None
    return role, content


def _messages_from_candidate(candidate: Any) -> list[dict[str, str]]:
    candidate = _json_value(candidate)
    if not isinstance(candidate, list):
        return []
    messages: list[dict[str, str]] = []
    for item in candidate:
        parsed = _message_from_item(item)
        if parsed:
            messages.append({"role": parsed[0], "content": parsed[1]})
    return messages


def _event_messages(events: Any) -> list[dict[str, str]]:
    events = events if isinstance(events, list) else []
    messages: list[dict[str, str]] = []
    for event in events:
        if not isinstance(event, Mapping):
            continue
        message = event.get("message")
        if isinstance(message, Mapping):
            parsed = _message_from_item(message)
            if parsed:
                messages.append({"role": parsed[0], "content": parsed[1]})
                continue
        event_type = str(event.get("type", "")).casefold()
        if event_type in {"user", "assistant", "tool", "tool_result"}:
            parsed = _message_from_item({"role": event_type, "content": event.get("content", event.get("text"))})
            if parsed:
                messages.append({"role": parsed[0], "content": parsed[1]})
    return messages


def _unwrap_raw(raw: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(raw)
    nested = _json_value(raw.get("row_json"))
    if isinstance(nested, Mapping):
        merged = dict(nested)
        merged.update({key: value for key, value in raw.items() if key != "row_json"})
        return merged
    return result


def _extract_messages(raw: Mapping[str, Any]) -> list[dict[str, str]]:
    if "_event_stream" in raw:
        return _event_messages(raw.get("_event_stream"))
    for key in ("messages", "conversations", "conversation", "dialogue"):
        messages = _messages_from_candidate(raw.get(key))
        if messages:
            return messages
    if "message" in raw:
        messages = _messages_from_candidate(raw.get("message"))
        if messages:
            return messages
    user = (
        _text(raw.get("user"))
        or _text(raw.get("input"))
        or _text(raw.get("instruction"))
        or _text(raw.get("prompt"))
        or _text(raw.get("question"))
        or _text(raw.get("problem"))
    )
    assistant = (
        _text(raw.get("assistant"))
        or _text(raw.get("output"))
        or _text(raw.get("response"))
        or _text(raw.get("answer"))
        or _text(raw.get("solution"))
    )
    messages = []
    if user:
        messages.append({"role": "user", "content": user})
    if assistant:
        messages.append({"role": "assistant", "content": assistant})
    return messages


def _extract_reasoning(raw: Mapping[str, Any], assistant: str) -> tuple[str, str]:
    reasoning = ""
    for key in ("reasoning_content", "reasoning", "r1_generation", "analysis", "thought"):
        value = _text(raw.get(key))
        if value:
            reasoning = value
            break
    if not reasoning:
        match = _THINK_RE.search(assistant) or _ANALYSIS_RE.search(assistant)
        if match:
            reasoning = match.group(1).strip()
    final = _THINK_RE.sub("", assistant)
    final = _ANALYSIS_RE.sub("", final).strip()
    return reasoning, final


def _nested_metadata(raw: Mapping[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key, value in raw.items():
        if key in _CONTENT_KEYS or str(key).startswith("_"):
            continue
        if key in {"metadata", "quality_scores", "gen_usr_configs", "gen_asst_configs"}:
            metadata[key] = value
        elif isinstance(value, (str, int, float, bool)) or value is None:
            metadata[key] = value
    return metadata


def _find_value(raw: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in raw and raw[key] not in (None, ""):
            return raw[key]
    for nested_key in ("metadata", "meta"):
        nested = raw.get(nested_key)
        if isinstance(nested, Mapping):
            for key in keys:
                if key in nested and nested[key] not in (None, ""):
                    return nested[key]
    return None


def _hash_messages(messages: Sequence[Mapping[str, Any]]) -> str:
    normalized = []
    for message in messages:
        normalized.append(
            {
                "role": str(message.get("role", "")).strip().casefold(),
                "content": " ".join(str(message.get("content", "")).split()),
            }
        )
    return hashlib.sha256(
        json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def simhash_text(text: str, *, max_chars: int = 4096, max_grams: int = 256) -> int:
    """Return a stable 64-bit SimHash using bounded text and gram windows."""

    normalized = unicodedata.normalize("NFKC", text)
    if max_chars > 0 and len(normalized) > max_chars:
        half = max_chars // 2
        normalized = normalized[:half] + "\n<simhash-window>\n" + normalized[-half:]
    gram_count = max(0, len(normalized) - 2)
    stride = max(1, (gram_count + max_grams - 1) // max_grams) if max_grams > 0 else 1
    grams = [normalized[index : index + 3] for index in range(0, gram_count, stride)]
    if not grams:
        grams = [normalized]
    vector = [0] * 64
    for gram in grams:
        digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        for bit in range(64):
            vector[bit] += 1 if value & (1 << bit) else -1
    result = 0
    for bit, score in enumerate(vector):
        if score >= 0:
            result |= 1 << bit
    return result


def canonicalize_source_row(source_row: SourceRow, *, source_metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    raw = _unwrap_raw(source_row.raw)
    messages = _extract_messages(raw)
    system = "\n".join(item["content"] for item in messages if item["role"] == "system").strip()
    user = "\n".join(item["content"] for item in messages if item["role"] == "user").strip()
    assistant_messages = [item["content"] for item in messages if item["role"] == "assistant"]
    assistant = assistant_messages[-1].strip() if assistant_messages else ""
    reasoning, assistant_final = _extract_reasoning(raw, assistant)
    metadata_language = _find_value(raw, ("language", "lang", "language_code"))
    language = analyze_record_languages(
        {"user": user, "assistant_final": assistant_final, "reasoning": reasoning},
        metadata_language,
    )
    combined = "\n".join(item["content"] for item in messages)
    content_hash = _hash_messages(messages)
    source_metadata = source_metadata or {}
    teacher = _find_value(raw, ("teacher_model", "model", "teacher", "generator"))
    if teacher is None and isinstance(raw.get("gen_asst_configs"), Mapping):
        teacher = raw["gen_asst_configs"].get("input_generator")
    if teacher is None and isinstance(raw.get("gen_usr_configs"), Mapping):
        teacher = raw["gen_usr_configs"].get("input_generator")
    tools = _find_value(raw, ("tools", "tool_calls", "functions"))
    category = str(_find_value(raw, ("category", "domain", "task_type", "source_category")) or source_row.dataset_id)
    sub_category = str(source_row.source_split)
    raw_metadata = _nested_metadata(raw)
    raw_metadata["source_path"] = source_row.source_path
    raw_metadata["raw_row_index"] = source_row.row_index
    if "source" in raw:
        raw_metadata["source"] = raw["source"]
    license_raw = _find_value(raw, ("license", "license_id"))
    if license_raw is None:
        license_raw = source_metadata.get("license")
    source_revision = str(
        _find_value(raw, ("source_revision", "revision", "commit_hash"))
        or source_metadata.get("revision")
        or "unknown"
    )
    record_id = hashlib.sha256(f"{source_row.dataset_id}:{source_row.source_row_id}".encode("utf-8")).hexdigest()
    record = {
        "record_id": record_id,
        "source_dataset": source_row.dataset_id,
        "source_config": source_row.source_config,
        "source_split": source_row.source_split,
        "source_row_id": source_row.source_row_id,
        "source_revision": source_revision,
        "category": category,
        "sub_category": sub_category,
        **language,
        "system": system,
        "user": user,
        "assistant_final": assistant_final,
        "reasoning": reasoning,
        "messages": messages,
        "tools": tools,
        "teacher_model": str(teacher) if teacher is not None else "unknown",
        "provenance_raw": {
            "source_repo": source_row.repo,
            "source_path": source_row.source_path,
            "source_row_id": source_row.source_row_id,
        },
        "license_raw": license_raw if license_raw is not None else "unknown",
        "vendor_specific": bool(source_metadata.get("vendor_specific", False)),
        "reasoning_sft_ready": bool(reasoning) or "math" not in category.casefold(),
        "tool_sft_ready": bool(tools) or any(item["role"] == "tool" for item in messages),
        "char_count": len(combined),
        "token_count": estimate_tokens(combined),
        "tokenizer_pending": True,
        "quality_score": 0.0,
        "quality_breakdown": {},
        "factuality_verified": False,
        "quality_method": "pending",
        "content_hash": content_hash,
        "simhash": f"{simhash_text(combined):016x}",
        "duplicate_cluster": None,
        "contamination_status": "not_checked_missing_eval_source",
        "eligibility": True,
        "exclusion_reasons": [],
        "raw_metadata_json": json.dumps(raw_metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    }
    # Keep the public field order stable while allowing future additive fields.
    return {field: record.get(field) for field in CANONICAL_FIELDS}


def canonical_record_fingerprint(record: Mapping[str, Any]) -> str:
    return canonical_json_hash(
        {
            "source_dataset": record.get("source_dataset"),
            "source_row_id": record.get("source_row_id"),
            "content_hash": record.get("content_hash"),
        }
    )
