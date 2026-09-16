from __future__ import annotations

import json
from pathlib import Path

import pytest

import zipangu.finalization.runner as runner_module
from zipangu.data.adapters import SourceRow
from zipangu.data.discovery import DatasetLocation, SourcePolicy, read_local_revision

from zipangu.finalization.contamination import (
    ContaminationIndex,
    contamination_status,
    content_text,
    minhash_signature,
    simhash,
)
from zipangu.finalization.baseline import _required_eval_sources_missing
from zipangu.finalization.core import FinalizationBlocked, RuntimeStore, sha256_text
from zipangu.finalization.eval import (
    EvalUnavailable,
    fingerprint_eval_source,
    iter_eval_rows,
    write_eval_registry,
)
from zipangu.finalization.schema_adapters import project_eval_record
from zipangu.finalization.orchestrator import FinalizationRunner
from zipangu.finalization.runner import (
    _batched_contamination_payloads,
    _compact_token_record,
    _contamination_parts_resume_info,
    _contamination_record,
    _contamination_resume_info,
    _iter_parquet,
    _ordered_contamination_rows,
    _parquet_writer,
    _prepare_contamination_parts,
)
from zipangu.finalization.stages import _source_pool, _stage_license
from zipangu.finalization.selection import (
    classify_nemotron_split,
    fable_length_bucket,
    is_known_nemotron_split,
    quality_threshold_comparison,
    safe_segment_trace,
    candidate_manifest,
    stratified_sample_by_token_mass,
    validate_training_record,
)
from zipangu.finalization.tokenization import TokenCounter, TokenStats


class FakeTokenizer:
    def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
        return {"input_ids": list(range(max(1, len(text.split()))))}

    def apply_chat_template(self, messages: list[dict[str, str]], *, tokenize: bool, return_dict: bool = False, return_assistant_tokens_mask: bool = False, **_: object):
        values = [str(item.get("content", "")) for item in messages]
        length = sum(max(1, len(value.split())) for value in values)
        if not tokenize:
            return "\n".join(values)
        result = {"input_ids": list(range(length))}
        if return_dict and return_assistant_tokens_mask:
            result["assistant_masks"] = [
                1 if item.get("role") == "assistant" else 0
                for item in messages
                for _ in range(max(1, len(str(item.get("content", "")).split())))
            ]
        return result if return_dict else list(range(length))


class NoGenerationTokenizer(FakeTokenizer):
    chat_template = "{% for message in messages %}{{ message.content }}{% endfor %}"

    def __init__(self) -> None:
        self.template_calls = 0

    def apply_chat_template(self, *args: object, **kwargs: object):
        self.template_calls += 1
        return super().apply_chat_template(*args, **kwargs)


class RejectAssistantOnlyTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: object):
        if messages and all(item.get("role") == "assistant" for item in messages):
            raise RuntimeError("No user query found in messages")
        return super().apply_chat_template(messages, **kwargs)


def _record(text: str = "こんにちは。") -> dict[str, object]:
    return {
        "record_id": "row-1",
        "messages": [
            {"role": "user", "content": text},
            {"role": "assistant", "content": "これは十分な最終回答です。"},
        ],
        "user": text,
        "assistant_final": "これは十分な最終回答です。",
        "source_dataset": "math_japanese_8k",
        "category": "japanese_math",
    }


def test_read_local_revision_supports_huggingface_line_metadata(tmp_path: Path) -> None:
    metadata_dir = tmp_path / ".cache" / "huggingface" / "download"
    metadata_dir.mkdir(parents=True)
    revision = "b385b781defb9bf4266175cd8d7e7c53af0290b8"
    (metadata_dir / "data.metadata").write_text(
        f"{revision}\n\"etag-value\"\n1788566400.0\n",
        encoding="utf-8",
    )

    assert read_local_revision(tmp_path) == revision


def test_read_local_revision_keeps_json_compatibility_and_fails_closed_on_mixed(tmp_path: Path) -> None:
    metadata_dir = tmp_path / ".cache" / "huggingface" / "download"
    metadata_dir.mkdir(parents=True)
    first = "a" * 40
    second = "b" * 40
    (metadata_dir / "first.metadata").write_text(
        json.dumps({"commit_hash": first}),
        encoding="utf-8",
    )
    (metadata_dir / "second.metadata").write_text(
        f"{second}\netag\n0\n",
        encoding="utf-8",
    )

    assert read_local_revision(tmp_path) == f"mixed:{first},{second}"


def test_license_stage_records_evidence_but_never_selects_a_human_decision(tmp_path: Path) -> None:
    policy = SourcePolicy.from_mapping(
        {
            "id": "extraction_wiki_ja",
            "repo": "owner/source",
            "local_dir": "owner__source",
            "category": "general",
            "role": "train_candidate",
        },
        0,
    )
    location = DatasetLocation(
        policy=policy,
        root=tmp_path,
        path=tmp_path,
        revision="a" * 40,
        card_metadata={"license": "apache-2.0"},
    )

    class Runner:
        repo_root = tmp_path

        @staticmethod
        def _locations():
            return tmp_path, [location]

    result = _stage_license(Runner())
    payload = json.loads(
        (tmp_path / "data" / "manifests" / "source_approval_candidates.json").read_text(
            encoding="utf-8"
        )
    )

    assert result["status"] == "PASS"
    assert payload["ai_may_set_human_decision"] is False
    assert payload["sources"][0]["human_decision"] is None
    assert payload["sources"][0]["recommended_action"] == "HUMAN_DECISION_REQUIRED"


def test_source_approval_matrix_has_five_unset_human_choices() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    decisions = json.loads(
        (repo_root / "data" / "manifests" / "source_approval_decisions.json").read_text(
            encoding="utf-8"
        )
    )
    audit = json.loads(
        (repo_root / "data" / "manifests" / "source_license_provenance_audit.json").read_text(
            encoding="utf-8"
        )
    )

    assert decisions["owner"] == "human_reviewer"
    assert decisions["ai_may_choose_or_apply_decision"] is False
    assert len(decisions["allowed_decisions"]) == 5
    assert len(decisions["sources"]) == 10
    assert len({item["dataset"] for item in decisions["sources"]}) == 10
    assert all(item["decision"] is None for item in decisions["sources"])
    assert all(item["decided_by"] is None for item in decisions["sources"])
    assert audit["summary"]["package_revisions_known"] == 10
    assert audit["summary"]["human_decisions_recorded"] == 0
    assert all(len(item["package_revision"]) == 40 for item in audit["sources"])
    assert all(item["human_decision"] is None for item in audit["sources"])


def test_hashes_and_lsh_inputs_are_deterministic() -> None:
    text = "NFKC  日本語の fingerprint です。"
    assert minhash_signature(text) == minhash_signature(text)
    assert simhash(text) == simhash(text)
    index = ContaminationIndex()
    index.add("eval-1", "同じ問題を詳しく説明してください。", prompt="同じ問題")
    result = index.lookup("同じ問題を詳しく説明してください。", prompt="同じ問題")
    assert result["status"] == "quarantine_exact"
    assert result["matches"][0]["id"] == "eval-1"


def test_random_simhash_baseline_does_not_create_manual_review() -> None:
    shared = "共通フレーズABCDEFGHIJKLMN "
    eval_text = shared + ("猫と天気の説明。" * 30)
    unrelated_candidate = shared + ("量子力学と宇宙論。" * 30)
    index = ContaminationIndex()
    index.add("eval-1", eval_text)
    result = index.lookup(unrelated_candidate)
    assert result["status"] == "clean"
    assert result["matches"] == []


def test_calibrated_near_duplicate_is_still_quarantined() -> None:
    eval_text = "日本の首都はどこですか。理由と歴史的背景を三つの観点から具体的に説明してください。"
    candidate = "日本の首都はどこですか。理由と歴史的背景を二つの観点から具体的に説明してください。"
    index = ContaminationIndex()
    index.add("eval-1", eval_text)
    result = index.lookup(candidate)
    assert result["status"] == "quarantine_near"
    assert result["matches"][0]["id"] == "eval-1"


def test_contamination_index_rejects_empty_content() -> None:
    index = ContaminationIndex()
    with pytest.raises(ValueError, match="empty"):
        index.add("eval-empty", "   \n")


def test_empty_candidate_projection_is_excluded_before_lookup() -> None:
    index = ContaminationIndex()
    index.add("eval-1", "実在する評価プロンプト")
    result = contamination_status(
        {"messages": [], "source_dataset": "source"},
        index,
        missing_required_eval=False,
    )
    assert result["status"] == "excluded_empty_projection"
    assert result["reasons"] == ["empty_candidate_projection"]
    assert result["content_hash"] == ""
    assert result["prompt_hash"] == ""


def test_mt_bench_eval_fingerprint_projects_turns(tmp_path: Path) -> None:
    source = tmp_path / "mt-bench"
    source.mkdir()
    (source / "questions.jsonl").write_text(
        json.dumps({"question_id": 1, "turns": ["質問1", "質問2"]}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    result = fingerprint_eval_source(
        {
            "id": "japanese_mt_bench",
            "name": "Japanese MT-Bench",
            "version": "test",
            "split": "embedded",
            "source": "local",
            "source_revision": "test",
            "license_raw": "test",
            "required": True,
        },
        source,
        tmp_path / "fingerprints",
    )
    fingerprint = json.loads(
        (tmp_path / "fingerprints" / "japanese_mt_bench.jsonl")
        .read_text(encoding="utf-8")
        .strip()
    )
    assert result["row_count"] == 1
    assert fingerprint["normalized_length"] > 0
    assert fingerprint["content_hash"] != sha256_text("")


def test_mt_bench_reference_answer_projects_choice_turns() -> None:
    projection = project_eval_record(
        "japanese_mt_bench",
        {
            "question_id": 1,
            "choices": [
                {"index": 0, "turns": ["回答1", "回答2"]},
                {"index": 1, "turns": ["別回答1", "別回答2"]},
            ],
        },
    )
    assert projection.adapter_id == "mt_bench_turns_and_choices_v1"
    assert projection.content == "回答1\n回答2\n別回答1\n別回答2"
    assert projection.prompt == ""


def test_llm_jp_eval_projects_text_and_output() -> None:
    projection = project_eval_record(
        "llm_jp_instruction_eval",
        {"text": "評価指示", "output": "参照回答"},
    )
    assert projection.adapter_id == "llm_jp_text_output_v1"
    assert projection.content == "評価指示\n参照回答"
    assert projection.prompt == "評価指示"


def test_eval_fingerprint_rejects_empty_projection(tmp_path: Path) -> None:
    source = tmp_path / "unknown-eval"
    source.mkdir()
    (source / "test.jsonl").write_text('{"unknown": "metadata"}\n', encoding="utf-8")
    with pytest.raises(EvalUnavailable, match="empty projection"):
        fingerprint_eval_source(
            {
                "id": "zipangu_private_holdout",
                "name": "Holdout",
                "version": "test",
                "split": "holdout",
                "source": "local",
                "source_revision": "test",
                "license_raw": "private",
                "required": False,
            },
            source,
            tmp_path / "fingerprints",
        )


def _frontier_location() -> DatasetLocation:
    policy = SourcePolicy.from_mapping(
        {
            "id": "frontier_multi_teacher",
            "repo": "owner/frontier",
            "local_dir": "frontier",
            "category": "frontier_reasoning",
            "role": "train_candidate",
        },
        0,
    )
    return DatasetLocation(
        policy=policy,
        root=None,
        path=None,
        revision="a" * 40,
        card_metadata={},
    )


def test_frontier_contamination_adapter_reads_messages_json() -> None:
    source_row = SourceRow(
        dataset_id="frontier_multi_teacher",
        repo="owner/frontier",
        source_path="data/canonical/train-00000-of-00006.parquet",
        source_row_id="data/canonical/train-00000-of-00006.parquet#1",
        source_config="default",
        source_split="train",
        row_index=0,
        raw={
            "messages_json": json.dumps(
                [
                    {"role": "user", "content": "評価対象の質問"},
                    {"role": "assistant", "content": "評価対象の回答"},
                ],
                ensure_ascii=False,
            )
        },
    )
    record = _contamination_record(_frontier_location(), source_row)
    assert record["contamination_schema_adapter"] == "frontier_messages_json_v1"
    assert record["contamination_scan_disposition"] == "scan"
    assert content_text(record) == "評価対象の質問\n評価対象の回答"


@pytest.mark.parametrize(
    ("source_path", "expected_disposition"),
    [
        ("metadata/context_buckets/qwen3.parquet", "excluded_metadata"),
        ("metadata/parent_ids/train.parquet", "excluded_metadata"),
        ("data/token_stats/train.parquet", "excluded_derivative"),
        ("data/glm47_native/train.parquet", "excluded_derivative"),
        ("data/smoke/train.parquet", "excluded_derivative"),
        ("data/prompt_completion_text/train.parquet", "excluded_derivative"),
        ("data/sft_balanced/train.parquet", "excluded_derivative"),
        ("data/canonical/train-00000-of-00005.parquet", "excluded_derivative"),
    ],
)
def test_frontier_contamination_adapter_excludes_non_text_artifacts(
    source_path: str,
    expected_disposition: str,
) -> None:
    source_row = SourceRow(
        dataset_id="frontier_multi_teacher",
        repo="owner/frontier",
        source_path=source_path,
        source_row_id=f"{source_path}#1",
        source_config="default",
        source_split="train",
        row_index=0,
        raw={"id": "row-1", "input_ids": [1, 2], "labels": [1, 2]},
    )
    record = _contamination_record(_frontier_location(), source_row)
    assert record["contamination_scan_disposition"] == expected_disposition


def test_excluded_frontier_row_bypasses_fingerprinting() -> None:
    source_row = SourceRow(
        dataset_id="frontier_multi_teacher",
        repo="owner/frontier",
        source_path="metadata/context_buckets/train.parquet",
        source_row_id="metadata/context_buckets/train.parquet#1",
        source_config="default",
        source_split="train",
        row_index=0,
        raw={"messages_json": "[]"},
    )
    rows = list(
        _ordered_contamination_rows(
            [(_frontier_location(), source_row)],
            ContaminationIndex(),
            False,
            workers=1,
        )
    )
    assert len(rows) == 1
    result, status, reasons, _ = rows[0]
    assert status == "excluded_metadata"
    assert result["content_hash"] == ""
    assert result["prompt_hash"] == ""
    assert result["scan_disposition"] == "excluded_metadata"
    assert "frontier_metadata_view_not_trainable_text" in reasons
    assert "schema_adapter:frontier_metadata_excluded_v1" in reasons


def test_all_candidate_datasets_have_explicit_contamination_schema_adapters() -> None:
    from zipangu.finalization.schema_adapters import CANDIDATE_SCHEMA_ADAPTERS

    assert set(CANDIDATE_SCHEMA_ADAPTERS) == set(runner_module.TRAINING_CANDIDATE_IDS)


def test_all_eval_datasets_have_explicit_contamination_schema_adapters() -> None:
    from zipangu.finalization.eval import EVAL_SPECS
    from zipangu.finalization.schema_adapters import EVAL_SCHEMA_ADAPTERS

    assert set(EVAL_SCHEMA_ADAPTERS) == {str(item["id"]) for item in EVAL_SPECS}


def test_missing_eval_never_becomes_clean() -> None:
    result = contamination_status(_record(), None, missing_required_eval=True)
    assert result["status"] == "not_checked_missing_eval_source"
    assert "required_eval_source_missing" in result["reasons"]


def test_parallel_contamination_preserves_sequential_order_and_results() -> None:
    index = ContaminationIndex()
    matching = _record("評価用の一致テキスト")
    index.add_record("eval-1", matching)
    records = []
    for row_index, text in enumerate(("別のテキスト", "評価用の一致テキスト", "さらに別のテキスト")):
        record = _record(text)
        record["record_id"] = f"row-{row_index}"
        records.append((record, "math_japanese_8k", "train.jsonl", f"train.jsonl#{row_index}"))

    sequential = list(
        _ordered_contamination_rows(records, index, False, workers=1, batch_size=1)
    )
    parallel = list(
        _ordered_contamination_rows(records, index, False, workers=2, batch_size=1)
    )

    assert parallel == sequential
    assert [item[0]["record_id"] for item in parallel] == ["row-0", "row-1", "row-2"]
    assert parallel[1][1] == "quarantine_exact"


def test_contamination_batches_respect_raw_character_budget() -> None:
    location = _frontier_location()
    rows = [
        SourceRow(
            dataset_id="frontier_multi_teacher",
            repo="owner/frontier",
            source_path="data/canonical/train-00000-of-00006.parquet",
            source_row_id=f"data/canonical/train-00000-of-00006.parquet#{index}",
            source_config="default",
            source_split="train",
            row_index=index - 1,
            raw={"messages_json": "x" * size},
        )
        for index, size in enumerate((60, 60, 20), start=1)
    ]
    batches = list(
        _batched_contamination_payloads(
            [(location, row) for row in rows],
            64,
            max_characters=100,
        )
    )
    assert [[item[1].row_index for item in batch] for batch in batches] == [
        [0],
        [1, 2],
    ]


def test_contamination_resume_uses_valid_partial_row_count_and_tail(tmp_path: Path) -> None:
    partial = tmp_path / ".contamination_results.parquet.partial"
    rows = [
        {
            "record_id": f"row-{index}",
            "source_dataset": "math_japanese_8k",
            "source_row_id": f"train.jsonl#{index}",
            "content_hash": f"hash-{index}",
            "prompt_hash": "",
            "status": "not_checked_missing_eval_source",
            "similarity": 0.0,
            "reasons": "required_eval_source_missing",
            "matched_eval_ids": "",
        }
        for index in range(3)
    ]
    assert _parquet_writer(partial, rows, batch_size=2) == 3

    resumed = _contamination_resume_info(
        {"resume_from": [{"rows_scanned": 2}]},
        partial,
    )

    assert resumed == (3, "math_japanese_8k", "train.jsonl#2")


def test_contamination_chunk_resume_uses_closed_contiguous_parts(tmp_path: Path) -> None:
    parts = tmp_path / ".contamination_results.parquet.parts"
    parts.mkdir()
    rows = [
        {
            "record_id": f"row-{index}",
            "source_dataset": "math_japanese_8k",
            "source_row_id": f"train.jsonl#{index}",
        }
        for index in range(3)
    ]
    _parquet_writer(parts / "part-000000000000-000000000003.parquet", rows)
    temporary = parts / ".part-000000000003-000000000004.parquet.tmp"
    temporary.write_bytes(b"interrupted")

    resumed = _contamination_parts_resume_info(
        {"resume_from": [{"rows_scanned": 3}]},
        parts,
    )

    assert resumed == (3, "math_japanese_8k", "train.jsonl#2")
    assert not temporary.exists()


def test_fresh_contamination_scan_refuses_implicit_chunk_resume(tmp_path: Path) -> None:
    parts = tmp_path / ".contamination_results.parquet.parts"
    parts.mkdir()
    (parts / "part-000000000000-000000000001.parquet").write_bytes(b"stale")
    with pytest.raises(FinalizationBlocked, match="refuses existing chunks"):
        _prepare_contamination_parts(parts, resume=False)
    assert (parts / "part-000000000000-000000000001.parquet").is_file()


def test_eval_registry_is_training_forbidden(tmp_path: Path) -> None:
    path = write_eval_registry(
        tmp_path,
        [{
            "id": "answer_carefully",
            "name": "AnswerCarefully",
            "version": "v2.0",
            "split": "test",
            "row_count": 0,
            "source": "llm-jp/AnswerCarefully",
            "source_revision": "v2.0",
            "local_path": "E:/zipangu-eval/sources/AnswerCarefully-v2.0-test",
            "license_raw": "unknown",
            "content_hash": None,
            "availability": "blocked_access_denied",
            "contamination_scan_ready": False,
            "training_forbidden": True,
            "required": True,
        }],
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["training_forbidden"] is True
    assert payload["evaluations"][0]["training_forbidden"] is True


def test_token_counter_preserves_unavailable_mask() -> None:
    counts = TokenCounter(FakeTokenizer()).count_record(_record())
    assert counts["training_formatted_tokens"] > 0
    assert counts["supervised_assistant_tokens"] is not None
    assert counts["assistant_mask_status"] == "available"


def test_special_dataset_buckets() -> None:
    assert classify_nemotron_split("ultra-v3_code_ja_train.jsonl", "code_ja") == "nemotron_ja_code"
    assert classify_nemotron_split("ultra-v3_math_en_train.jsonl", "math_en") == "nemotron_non_ja"
    assert is_known_nemotron_split("ultra-v3_math_en_train.jsonl", "train") is True
    assert is_known_nemotron_split("mystery.parquet", "train") is False
    assert fable_length_bucket(4_096) == "short"
    assert fable_length_bucket(4_097) == "medium"
    assert fable_length_bucket(32_769) == "extra_long"


def test_quality_threshold_comparison_and_stratified_mass_are_reproducible() -> None:
    rows = []
    for index in range(40):
        rows.append({
            "record_id": str(index),
            "source_dataset": "math_japanese_8k",
            "category": "japanese_math",
            "selection_bucket": "japanese_math_reasoning",
            "training_formatted_tokens": 100,
            "quality_score": 80 + index % 20,
            "template_family_hash": f"template-{index % 8}",
            "language": "japanese",
        })
    comparison = quality_threshold_comparison(rows, minimum_retention={"math_japanese_8k": 0.25})
    assert [item["threshold"] for item in comparison["thresholds"]] == [95, 90, 85, 80, 75]
    first, summary = stratified_sample_by_token_mass(
        rows,
        target_tokens=1_000,
        bucket_targets={"japanese_math_reasoning": 1.0},
        seed=3407,
    )
    second, _ = stratified_sample_by_token_mass(
        rows,
        target_tokens=1_000,
        bucket_targets={"japanese_math_reasoning": 1.0},
        seed=3407,
    )
    assert [item["record_id"] for item in first] == [item["record_id"] for item in second]
    assert summary["buckets"]["japanese_math_reasoning"]["selected_tokens"] <= 1_000


def test_training_record_requires_final_answer() -> None:
    record = _record()
    assert validate_training_record(record)["valid"] is True
    record["assistant_final"] = ""
    assert "final_answer_field_missing" in validate_training_record(record)["errors"]


def test_token_counter_skips_duplicate_mask_tokenization_when_template_has_no_generation_marker() -> None:
    tokenizer = NoGenerationTokenizer()
    counts = TokenCounter(tokenizer).count_record(_record())
    assert tokenizer.template_calls == 1
    assert counts["supervised_assistant_tokens"] is None
    assert counts["assistant_mask_status"] == "unavailable_tokenizer_mask"


def test_token_counter_quarantines_row_with_invalid_chat_template() -> None:
    record = _record()
    record["messages"] = [{"role": "assistant", "content": "answer only"}]
    counts = TokenCounter(RejectAssistantOnlyTokenizer()).count_records([record])[0]
    assert counts["raw_content_tokens"] > 0
    assert counts["training_formatted_tokens"] is None
    assert counts["supervised_assistant_tokens"] is None
    assert counts["assistant_mask_status"] == "unavailable_chat_template"


def test_compact_token_record_drops_conversation_text() -> None:
    compact = _compact_token_record(
        {
            "record_id": "row-1",
            "source_dataset": "source",
            "quality_score": 95,
            "messages": [{"role": "user", "content": "large private payload"}],
            "assistant_final": "large private payload",
        }
    )
    assert compact["record_id"] == "row-1"
    assert compact["quality_score"] == 95
    assert "messages" not in compact
    assert "assistant_final" not in compact


def test_oversized_token_record_is_fail_closed(monkeypatch) -> None:
    monkeypatch.setattr(runner_module, "MAX_TOKENIZATION_CHARACTERS", 10)
    policy = SourcePolicy.from_mapping(
        {
            "id": "source",
            "repo": "owner/source",
            "local_dir": "source",
            "category": "frontier_reasoning",
            "role": "candidate",
        },
        0,
    )
    location = DatasetLocation(
        policy=policy,
        root=None,
        path=None,
        revision="a" * 40,
        card_metadata={},
    )
    source_row = SourceRow(
        dataset_id="source",
        repo="owner/source",
        source_path="data.parquet",
        source_row_id="data.parquet#1",
        source_config="default",
        source_split="train",
        row_index=0,
        raw={"messages": [{"role": "user", "content": "x" * 11}]},
    )
    record, counts = runner_module._oversized_token_result((location, source_row))
    assert record["raw_character_count"] == 15
    assert record["quality_score"] == 0.0
    assert counts["training_formatted_tokens"] is None
    assert counts["assistant_mask_status"] == "unavailable_oversized_record"


def test_candidate_pool_excludes_fable_when_integrity_stage_is_blocked(monkeypatch) -> None:
    rows = [
        {
            "record_id": "fable-row",
            "source_dataset": "fable_5_premium",
            "source_split": "train",
            "selection_bucket": "agent_code_retention",
            "fable_bucket": "short",
            "training_formatted_tokens": 100,
            "quality_score": 99,
            "contamination_status": "not_checked_missing_eval_source",
        },
        {
            "record_id": "safe-row",
            "source_dataset": "math_japanese_8k",
            "source_split": "train",
            "selection_bucket": "japanese_math_reasoning",
            "training_formatted_tokens": 100,
            "quality_score": 99,
            "contamination_status": "not_checked_missing_eval_source",
            "reasoning_sft_ready": True,
        },
    ]

    class Runtime:
        @staticmethod
        def stage(name: str) -> dict[str, str]:
            assert name == "fable"
            return {"status": "BLOCKED"}

    class Runner:
        config = {"seed": 3407}
        runtime = Runtime()
        temp_root = Path("unused")

    monkeypatch.setattr("zipangu.finalization.stages._load_token_rows", lambda _: iter(rows))
    pools = _source_pool(Runner(), 75)
    assert [row["record_id"] for row in pools["japanese_math_reasoning"]] == ["safe-row"]
    assert "agent_code_retention" not in pools


def test_candidate_reconstruction_canonicalizes_only_selected_rows(monkeypatch) -> None:
    from zipangu.data.adapters import SourceRow
    from zipangu.finalization.stages import _candidate_rows_from_ids

    rows = [
        SourceRow(
            dataset_id="source",
            repo="owner/source",
            source_path="data.jsonl",
            source_row_id=f"data.jsonl#{index}",
            source_config="default",
            source_split="train",
            row_index=index,
            raw={"prompt": f"prompt {index}", "answer": f"answer {index}"},
        )
        for index in range(3)
    ]
    selected_id = runner_module.hashlib.sha256(
        b"source:data.jsonl#2"
    ).hexdigest()
    calls: list[str] = []

    monkeypatch.setattr(
        "zipangu.finalization.stages._iter_locations",
        lambda *_args, **_kwargs: iter((object(), row) for row in rows),
    )

    def canonical(_location: object, row: SourceRow) -> dict[str, object]:
        calls.append(row.source_row_id)
        return {"record_id": selected_id, "source_row_id": row.source_row_id}

    monkeypatch.setattr("zipangu.finalization.stages._canonical", canonical)

    class Runner:
        repo_root = Path(".")
        dataset_root = Path("unused")

    result = _candidate_rows_from_ids(
        Runner(),
        {selected_id},
        {selected_id: {"quality_score": 99}},
    )
    assert calls == ["data.jsonl#2"]
    assert result[0]["quality_score"] == 99


def test_runtime_store_resume_and_candidate_gate(tmp_path: Path) -> None:
    store = RuntimeStore(tmp_path / "runtime.json")
    assert store.start("tokenize", fingerprint="fp", cfg_hash="cfg") is True
    store.finish("tokenize", "PASS")
    assert store.can_resume("tokenize", "fp", "cfg") is True
    assert store.start("contamination", fingerprint="fp", cfg_hash="cfg") is True
    store.finish(
        "contamination",
        "BLOCKED",
        artifacts=[tmp_path / "contamination.parquet"],
        blocked_reasons=["required_eval_source_missing"],
    )
    assert store.can_resume("contamination", "fp", "cfg") is True
    assert store.can_resume("contamination", "changed", "cfg") is False
    assert store.state["training_allowed"] is False


def test_runtime_details_cannot_overwrite_stage_status(tmp_path: Path) -> None:
    store = RuntimeStore(tmp_path / "runtime.json")
    store.start("contamination", fingerprint="fp", cfg_hash="cfg")
    store.finish(
        "contamination",
        "BLOCKED",
        artifacts=[tmp_path / "result.parquet"],
        details={"status": "not_checked_missing_eval_source", "rows": 3},
    )
    stage = store.stage("contamination")
    assert stage["status"] == "BLOCKED"
    assert stage["details"]["status"] == "not_checked_missing_eval_source"
    assert stage["rows"] == 3


def test_runtime_repeated_resume_preserves_prior_checkpoints(tmp_path: Path) -> None:
    store = RuntimeStore(tmp_path / "runtime.json")
    store.start("tokenize", fingerprint="fp", cfg_hash="cfg")
    store.checkpoint("tokenize", {"rows_scanned": 10})
    store.start("tokenize", fingerprint="fp", cfg_hash="cfg", resume=True)
    assert [item["rows_scanned"] for item in store.stage("tokenize")["resume_from"]] == [10]
    store.checkpoint("tokenize", {"rows_scanned": 20})
    store.start("tokenize", fingerprint="fp", cfg_hash="cfg", resume=True)
    assert [item["rows_scanned"] for item in store.stage("tokenize")["resume_from"]] == [10, 20]


def test_parquet_reader_resumes_at_exact_row(tmp_path: Path) -> None:
    path = tmp_path / "rows.parquet"
    _parquet_writer(path, ({"index": index} for index in range(5)))
    assert [row["index"] for row in _iter_parquet(path, start_row=3)] == [3, 4]


def test_preflight_exit_codes(tmp_path: Path) -> None:
    config = Path(__file__).resolve().parents[1] / "configs" / "pretrain" / "finalization.yaml"
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()
    runner = FinalizationRunner(
        repo_root=Path(__file__).resolve().parents[1],
        dataset_root=dataset_root,
        eval_root=tmp_path / "eval",
        cache_root=tmp_path / "cache",
        model_root=tmp_path / "models",
        temp_root=tmp_path / "temp",
        config_path=config,
    )
    assert runner.run("preflight") == 0
    missing_runner = FinalizationRunner(
        repo_root=Path(__file__).resolve().parents[1],
        dataset_root=tmp_path / "missing",
        eval_root=tmp_path / "eval2",
        cache_root=tmp_path / "cache2",
        model_root=tmp_path / "models2",
        temp_root=tmp_path / "temp2",
        config_path=config,
    )
    assert missing_runner.run("preflight") == 2
def test_runtime_recovers_stale_stage_and_preserves_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "runtime.json"
    first = RuntimeStore(path)
    first.start("contamination", fingerprint="fp", cfg_hash="cfg")
    first.checkpoint("contamination", {"rows_scanned": 100, "source_path": "part.parquet"})
    recovered = RuntimeStore(path)
    assert recovered.stage("contamination")["status"] == "PARTIAL"
    assert "previous_process_interrupted_or_stale" in recovered.stage("contamination")["blocked_reasons"]
    recovered.start("contamination", fingerprint="fp", cfg_hash="cfg", resume=True)
    assert recovered.stage("contamination")["resume_from"] == [
        {"rows_scanned": 100, "source_path": "part.parquet"}
    ]


def test_candidate_manifest_is_never_training_allowed() -> None:
    manifest = candidate_manifest(
        [_record()],
        recipe_name="pilot-1m-jp-heavy",
        source_revisions={"math_japanese_8k": "unknown"},
        quality_thresholds={"recommended": 75},
        dedup_config={"exact": True},
        contamination_status="not_checked_missing_eval_source",
        license_status="review_required",
    )
    assert manifest["TRAINING_ALLOWED"] is False
    assert manifest["HUMAN_APPROVAL_REQUIRED"] is True
    assert manifest["status"] == "UNAPPROVED_RESEARCH_CANDIDATE"


def test_eval_test_split_accepts_test_root_without_mixing_train(tmp_path: Path) -> None:
    root = tmp_path / "test"
    root.mkdir()
    (root / "data.jsonl").write_text(
        json.dumps({"prompt": "test prompt", "answer": "test answer"}) + "\n",
        encoding="utf-8",
    )
    rows = list(iter_eval_rows(root, split_hint="test"))
    assert len(rows) == 1


def test_baseline_blocks_when_required_eval_source_is_missing(tmp_path: Path) -> None:
    registry = tmp_path / "data" / "manifests"
    registry.mkdir(parents=True)
    (registry / "eval_registry.json").write_text(
        json.dumps(
            {
                "evaluations": [
                    {"id": "available", "required": True, "availability": "available"},
                    {"id": "answer_carefully", "required": True, "availability": "blocked_access_denied"},
                ]
            }
        ),
        encoding="utf-8",
    )
    assert _required_eval_sources_missing(tmp_path) == ["answer_carefully"]
def test_safe_trace_segmentation_keeps_tool_boundary() -> None:
    messages = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "tool", "content": "three"},
        {"role": "assistant", "content": "four"},
    ]
    segments, state = safe_segment_trace(
        messages,
        max_tokens=3,
        token_counter=TokenCounter(FakeTokenizer()),
    )
    assert state == "segmented"
    assert segments is not None
    assert [item["role"] for item in segments[0]] == ["user", "assistant", "tool"]
    assert segments[1][-1]["role"] == "assistant"
def test_stage_error_maps_to_exit_one(tmp_path: Path) -> None:
    config = Path(__file__).resolve().parents[1] / "configs" / "pretrain" / "finalization.yaml"
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()
    runner = FinalizationRunner(
        repo_root=Path(__file__).resolve().parents[1],
        dataset_root=dataset_root,
        eval_root=tmp_path / "eval",
        cache_root=tmp_path / "cache",
        model_root=tmp_path / "models",
        temp_root=tmp_path / "temp",
        config_path=config,
    )

    def fail(_: object) -> dict[str, object]:
        raise RuntimeError("synthetic stage failure")

    runner._stage_callable = lambda _: fail
    assert runner.run("preflight") == 1
    assert runner.runtime.stage("preflight")["status"] == "ERROR"
def test_token_stats_keep_exact_mass_buckets_with_bounded_quantiles() -> None:
    stats = TokenStats()
    stats.QUANTILE_SAMPLE_LIMIT = 2
    for value in (500, 1_500, 5_000, 9_000, 40_000):
        stats.add(
            {
                "raw_content_tokens": value,
                "training_formatted_tokens": value,
                "supervised_assistant_tokens": value // 2,
            }
        )
    result = stats.as_dict()
    assert result["rows"] == 5
    assert result["training_formatted_tokens"] == 56_000
    assert result["max"] == 40_000
    assert [result[key] for key in ("<=1k", "1k-4k", "4k-8k", "8k-32k", ">32k")] == [1, 1, 1, 1, 1]
    assert result["quantile_sample_size"] == 2
