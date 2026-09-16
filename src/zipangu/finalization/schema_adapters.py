"""Explicit schema projections for contamination scanning.

The contamination gate must never infer that an arbitrary Parquet file is a
trainable text source.  Candidate repositories can contain several materialized
views of the same rows plus metadata and pre-tokenized derivatives.  This module
names the accepted schema for every candidate/eval dataset and makes exclusion
an explicit, auditable result.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import json
import re
from typing import Any

from ..data.adapters import SourceRow


CANDIDATE_SCHEMA_ADAPTERS: dict[str, str] = {
    "extraction_wiki_ja": "conversation_pairs_v1",
    "magpie_sft_v1": "conversation_pairs_v1",
    "nemotron_sft_multilingual_v2": "messages_v1",
    "math_japanese_8k": "instruction_output_v1",
    "ace_reason_math_japanese": "problem_answer_v1",
    "gpt_5_6_traces": "row_json_v1",
    "fable_5_5_distillation": "messages_v1",
    "claude_fable_code": "grouped_event_stream_v1",
    "frontier_multi_teacher": "frontier_authoritative_canonical_v1",
    "fable_5_premium": "messages_v1",
}

EVAL_SCHEMA_ADAPTERS: dict[str, str] = {
    "japanese_mt_bench": "mt_bench_turns_and_choices_v1",
    "llm_jp_instruction_eval": "llm_jp_text_output_v1",
    "answer_carefully": "answer_carefully_v2_fields",
    "zipangu_private_holdout": "local_holdout_fields_v1",
    "english_general_regression": "local_holdout_fields_v1",
    "japanese_math_holdout": "local_holdout_fields_v1",
    "japanese_writing_holdout": "local_holdout_fields_v1",
    "code_reasoning_sanity": "local_holdout_fields_v1",
}

_FRONTIER_CANONICAL = re.compile(
    r"^data/canonical/(?:"
    r"train-\d{5}-of-00006|"
    r"validation-00000-of-00001|"
    r"test-00000-of-00001"
    r")\.parquet$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class CandidateSchemaAdaptation:
    source_row: SourceRow
    adapter_id: str
    disposition: str
    reason: str


@dataclass(frozen=True)
class EvalProjection:
    content: str
    prompt: str
    adapter_id: str


def _json_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(part for item in value if (part := _text(item)))
    return ""


def _frontier_adaptation(source_row: SourceRow) -> CandidateSchemaAdaptation:
    source_path = source_row.source_path.replace("\\", "/")
    lower = source_path.casefold()
    if lower.startswith("metadata/"):
        return CandidateSchemaAdaptation(
            source_row=source_row,
            adapter_id="frontier_metadata_excluded_v1",
            disposition="excluded_metadata",
            reason="frontier_metadata_view_not_trainable_text",
        )
    if not _FRONTIER_CANONICAL.fullmatch(source_path):
        return CandidateSchemaAdaptation(
            source_row=source_row,
            adapter_id="frontier_derivative_excluded_v1",
            disposition="excluded_derivative",
            reason="frontier_non_authoritative_materialized_view",
        )

    raw = dict(source_row.raw)
    messages = _json_value(raw.get("messages_json"))
    if isinstance(messages, list):
        raw["messages"] = messages
    tools = _json_value(raw.get("tools_json"))
    if isinstance(tools, list):
        raw["tools"] = tools
    return CandidateSchemaAdaptation(
        source_row=replace(source_row, raw=raw),
        adapter_id="frontier_messages_json_v1",
        disposition="scan",
        reason="frontier_authoritative_canonical_view",
    )


def adapt_candidate_source_row(source_row: SourceRow) -> CandidateSchemaAdaptation:
    """Return the named schema adapter and scan disposition for one source row."""

    adapter_id = CANDIDATE_SCHEMA_ADAPTERS.get(source_row.dataset_id)
    if adapter_id is None:
        return CandidateSchemaAdaptation(
            source_row=source_row,
            adapter_id="unregistered_candidate_schema",
            disposition="excluded_unregistered_schema",
            reason="candidate_dataset_has_no_explicit_schema_adapter",
        )
    if source_row.dataset_id == "frontier_multi_teacher":
        return _frontier_adaptation(source_row)
    return CandidateSchemaAdaptation(
        source_row=source_row,
        adapter_id=adapter_id,
        disposition="scan",
        reason="explicit_candidate_schema_adapter",
    )


def _mt_bench_projection(row: Mapping[str, Any]) -> tuple[str, str]:
    turns = row.get("turns")
    if isinstance(turns, list):
        prompt = _text(turns)
        return prompt, prompt
    choices = row.get("choices")
    if isinstance(choices, list):
        content = "\n".join(
            part
            for choice in choices
            if isinstance(choice, Mapping)
            if (part := _text(choice.get("turns")))
        )
        return content, ""
    return "", ""


def _named_fields_projection(row: Mapping[str, Any]) -> tuple[str, str]:
    prompt = ""
    for key in ("text", "prompt", "instruction", "question", "problem", "input", "user"):
        prompt = _text(row.get(key))
        if prompt:
            break
    answer = ""
    for key in ("output", "answer", "response", "assistant_final", "assistant", "solution"):
        answer = _text(row.get(key))
        if answer:
            break
    content = "\n".join(part for part in (prompt, answer) if part)
    return content, prompt


def project_eval_record(eval_id: str, row: Mapping[str, Any]) -> EvalProjection:
    """Project one eval row through its registered, dataset-specific schema."""

    adapter_id = EVAL_SCHEMA_ADAPTERS.get(eval_id)
    if adapter_id is None:
        return EvalProjection(content="", prompt="", adapter_id="unregistered_eval_schema")
    if eval_id == "japanese_mt_bench":
        content, prompt = _mt_bench_projection(row)
    else:
        content, prompt = _named_fields_projection(row)
    return EvalProjection(content=content, prompt=prompt, adapter_id=adapter_id)
