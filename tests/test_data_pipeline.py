from __future__ import annotations

import gzip
import json
from pathlib import Path

from zipangu.data.adapters import SourceRow
from zipangu.data.canonical import canonicalize_source_row
from zipangu.data.contamination import contamination_scan
from zipangu.data.dedup import exact_deduplicate, hamming_distance, near_deduplicate
from zipangu.data.filtering import apply_filter, hard_filter
from zipangu.data.language import classify_text
from zipangu.data.reports import metadata_only_inventory
from zipangu.data.sampling import deterministic_key, select_by_token_mass, storage_preflight


def source_row(raw: dict, *, dataset_id: str = "fixture") -> SourceRow:
    return SourceRow(
        dataset_id=dataset_id,
        repo="fixture/repo",
        source_path="train.jsonl",
        source_row_id="train.jsonl#1",
        source_config="default",
        source_split="train",
        row_index=0,
        raw=raw,
    )


def canonical_fixture(text: str = "日本語の質問です。") -> dict:
    record = canonicalize_source_row(
        source_row(
            {
                "messages": [
                    {"role": "system", "content": "答えを説明してください。"},
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": "これは十分な説明を含む回答です。"},
                ]
            }
        )
    )
    record["eligibility"] = True
    record["quality_score"] = 80
    return record


def test_canonical_conversion_preserves_roles_and_reasoning() -> None:
    record = canonicalize_source_row(
        source_row(
            {
                "instruction": "計算してください。",
                "output": "<think>途中の推論</think>最終回答です。",
                "model": "teacher-x",
            }
        )
    )
    assert record["user"] == "計算してください。"
    assert record["assistant_final"] == "最終回答です。"
    assert record["reasoning"] == "途中の推論"
    assert record["teacher_model"] == "teacher-x"
    assert len(record["content_hash"]) == 64
    assert record["tokenizer_pending"] is True


def test_invalid_row_is_rejected_without_deleting_it() -> None:
    record = canonicalize_source_row(source_row({"_parse_error": "bad json"}))
    result = hard_filter(record)
    assert result.eligible is False
    assert "broken_json" in result.reasons
    updated = apply_filter(record, result)
    assert updated["eligibility"] is False
    assert "broken_json" in updated["exclusion_reasons"]


def test_language_heuristic_identifies_japanese_and_mixed() -> None:
    assert classify_text("これは日本語の文章です。")[0] == "japanese"
    assert classify_text("これは Japanese explanation です。")[0] == "mixed"
    assert classify_text("def add(a, b): return a + b")[0] == "code-heavy"


def test_exact_and_near_dedup_are_deterministic() -> None:
    first = canonical_fixture()
    second = dict(first)
    second["record_id"] = "second"
    winners, duplicates = exact_deduplicate([first, second])
    assert len(winners) == 1
    assert len(duplicates) == 1
    near_winners, near_duplicates, stats = near_deduplicate(
        [first, {**first, "record_id": "near", "simhash": first["simhash"]}],
        max_records=10,
    )
    assert len(near_winners) == 1
    assert len(near_duplicates) == 1
    assert stats["algorithm"] == "simhash_lsh"
    assert hamming_distance(0b1010, 0b1000) == 1


def test_contamination_match_and_missing_source_status() -> None:
    candidate = canonical_fixture("What is MT-Bench? 日本語")
    clean_missing, missing_summary = contamination_scan([candidate], None)
    assert clean_missing[0]["contamination_status"] == "not_checked_missing_eval_source"
    assert missing_summary["status"] == "not_checked_missing_eval_source"
    evaluation = [canonical_fixture("同じ問題")]
    matched, summary = contamination_scan([evaluation[0]], evaluation)
    assert matched[0]["contamination_status"] == "quarantine_contamination"
    assert summary["exact_matches"] == 1


def test_deterministic_sampling_and_storage_gate() -> None:
    records = [
        {"record_id": str(index), "content_hash": str(index), "source_row_id": str(index), "eligibility": True, "token_count": 10}
        for index in range(5)
    ]
    assert [item["record_id"] for item in select_by_token_mass(records, target_tokens=20)] == [
        item["record_id"] for item in select_by_token_mass(records, target_tokens=20)
    ]
    assert deterministic_key(records[0]) == deterministic_key(records[0])
    assert storage_preflight(free_bytes=100, estimated_output_bytes=80)["allowed"] is True
    assert storage_preflight(free_bytes=100, estimated_output_bytes=90)["allowed"] is False


def test_inventory_projection_omits_nested_samples() -> None:
    projected = metadata_only_inventory({"schema": {"messages": ["list"]}, "samples": [{"content": "secret"}], "files": [{"samples": [{"prompt": "secret"}], "rows": 1}]})
    assert projected["schema"]["messages"] == ["list"]
    assert "samples" not in projected
    assert "samples" not in projected["files"][0]


def test_audit_manifest_artifact_is_content_free(tmp_path: Path) -> None:
    path = tmp_path / "record.jsonl.gz"
    record = canonical_fixture()
    from zipangu.data.provenance import audit_manifest_record

    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(audit_manifest_record(record), ensure_ascii=False) + "\n")
    value = json.loads(gzip.open(path, "rt", encoding="utf-8").read())
    assert "assistant_final" not in value
    assert value["source_row_id"]
