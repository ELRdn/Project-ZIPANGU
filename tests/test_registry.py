from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from zipangu.provenance import canonical_json_hash
from zipangu.registry import (
    SCHEMA_VERSION,
    split_issues,
    validate_eval_config,
    validate_recipe_against_registry,
    validate_registry,
    validate_train_config,
)


MODEL = "empero-ai/Qwen3.8-9B-Distill"


def dataset(
    dataset_id: str,
    *,
    role: str = "train_candidate",
    status: str = "quarantine",
    repo: str | None = "org/example",
    category: str = "japanese_math",
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": dataset_id,
        "repo": repo,
        "role": role,
        "category": category,
        "status": status,
    }
    if provenance is not None:
        item["provenance"] = provenance
    return item


def registry(*items: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "datasets": list(items)}


def recipe(*candidate_ids: str, forbidden: list[str] | None = None) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "name": "test-recipe",
        "model": MODEL,
        "objective": "sft_reasoning_distillation",
        "sampling_mass": {"training": 1.0},
        "buckets": {
            "training": {
                "candidates": list(candidate_ids),
                "target_weight": 1.0,
            }
        },
        "english_trace_policy": {
            "keep_original_subset": True,
            "create_japanese_rewrite_subset": True,
            "preserve_pair_provenance": True,
        },
        "eval_only_forbidden": forbidden or [],
    }


def eval_config() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "name": "test-eval",
        "target_model": {"candidate": "candidate", "reference": "reference", "base": MODEL},
        "primary": [{"id": "primary_eval", "repeat": 3, "role": "eval_only"}],
        "secondary": ["secondary_eval"],
        "contamination_gate": {
            "exact_hash": True,
            "substring": True,
            "ngram": True,
            "minhash": True,
            "manual_review_top_hits": True,
        },
        "preserve_raw_generations": True,
        "publish_raw_generations": False,
        "victory": {
            "minimum_primary_wins": 1,
            "total_primary": 1,
            "require_general_regression_gate": True,
        },
        "regression_tolerance": None,
    }


def valid_manifest(dataset_id: str, repo: str, category: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "source_repo": repo,
        "source_config": "default",
        "source_revision": "abc123",
        "source_split": "train",
        "license": "Apache-2.0",
        "language": "ja",
        "category": category,
        "transform_chain": ["none"],
        "contamination_status": "clear",
        "records": [{"source_row_id": "row-1", "content_hash": "a" * 64}],
    }


def write_manifest(
    tmp_path: Path,
    dataset_id: str,
    repo: str,
    category: str,
) -> tuple[dict[str, Any], str]:
    manifest = valid_manifest(dataset_id, repo, category)
    manifest_path = tmp_path / "configs" / "datasets" / "manifests" / f"{dataset_id}.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    relative_path = manifest_path.relative_to(tmp_path).as_posix()
    return manifest, relative_path


def test_eval_only_must_be_locked() -> None:
    issues = validate_registry(
        registry(
            dataset(
                "eval",
                role="eval_only",
                status="quarantine",
                repo=None,
                category="evaluation",
            )
        )
    )
    assert any(issue.code == "protected_dataset_not_locked" for issue in issues)


def test_registry_rejects_duplicate_and_unknown_values() -> None:
    issues = validate_registry(
        registry(
            dataset("duplicate"),
            dataset("duplicate", role="unknown", status="unknown"),
        )
    )
    codes = {issue.code for issue in issues}
    assert {"duplicate_dataset_id", "unknown_role", "unknown_status"} <= codes


def test_quarantine_is_structurally_valid_but_blocked() -> None:
    issues = validate_recipe_against_registry(
        recipe("train_a"),
        registry(dataset("train_a")),
    )
    errors, blocked = split_issues(issues)
    assert not errors
    assert any(issue.code == "training_source_not_approved" for issue in blocked)


def test_forbidden_training_source_is_an_error() -> None:
    issues = validate_recipe_against_registry(
        recipe("eval_a", forbidden=["eval_a"]),
        registry(dataset("eval_a", role="eval_only", status="locked", repo=None, category="evaluation")),
    )
    errors, _ = split_issues(issues)
    assert any(issue.code == "forbidden_training_source" for issue in errors)


def test_recipe_rejects_mass_mismatch_and_duplicate_candidates() -> None:
    candidate = dataset("train_a")
    bad_recipe = recipe("train_a", "train_a")
    bad_recipe["sampling_mass"] = {"training": 0.9}
    issues = validate_recipe_against_registry(bad_recipe, registry(candidate))
    codes = {issue.code for issue in issues}
    assert {"sampling_mass_sum_invalid", "duplicate_candidate_id"} <= codes


def test_approved_source_requires_and_validates_sidecar(tmp_path: Path) -> None:
    source_repo = "org/train-a"
    category = "japanese_math"
    manifest, relative_path = write_manifest(tmp_path, "train_a", source_repo, category)
    approved = dataset(
        "train_a",
        status="approved",
        repo=source_repo,
        category=category,
        provenance={
            "manifest_path": relative_path,
            "manifest_sha256": canonical_json_hash(manifest),
        },
    )
    issues = validate_recipe_against_registry(
        recipe("train_a"),
        registry(approved),
        repo_root=tmp_path,
    )
    assert not issues


def test_unknown_manifest_provenance_blocks_approved_source(tmp_path: Path) -> None:
    source_repo = "org/train-a"
    category = "japanese_math"
    manifest, relative_path = write_manifest(tmp_path, "train_a", source_repo, category)
    manifest["source_revision"] = "unknown"
    manifest_path = tmp_path / relative_path
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    approved = dataset(
        "train_a",
        status="approved",
        repo=source_repo,
        category=category,
        provenance={
            "manifest_path": relative_path,
            "manifest_sha256": canonical_json_hash(manifest),
        },
    )
    issues = validate_recipe_against_registry(
        recipe("train_a"),
        registry(approved),
        repo_root=tmp_path,
    )
    assert any(issue.code == "manifest_field_unknown" for issue in issues)


def test_manifest_hash_mismatch_blocks_approved_source(tmp_path: Path) -> None:
    source_repo = "org/train-a"
    category = "japanese_math"
    manifest, relative_path = write_manifest(tmp_path, "train_a", source_repo, category)
    approved = dataset(
        "train_a",
        status="approved",
        repo=source_repo,
        category=category,
        provenance={"manifest_path": relative_path, "manifest_sha256": "0" * 64},
    )
    issues = validate_recipe_against_registry(
        recipe("train_a"),
        registry(approved),
        repo_root=tmp_path,
    )
    assert any(issue.code == "manifest_hash_mismatch" for issue in issues)
    assert manifest["dataset_id"] == "train_a"


def test_empty_manifest_records_block_approved_source(tmp_path: Path) -> None:
    source_repo = "org/train-a"
    category = "japanese_math"
    manifest, relative_path = write_manifest(tmp_path, "train_a", source_repo, category)
    manifest["records"] = []
    manifest_path = tmp_path / relative_path
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    approved = dataset(
        "train_a",
        status="approved",
        repo=source_repo,
        category=category,
        provenance={
            "manifest_path": relative_path,
            "manifest_sha256": canonical_json_hash(manifest),
        },
    )
    issues = validate_recipe_against_registry(
        recipe("train_a"),
        registry(approved),
        repo_root=tmp_path,
    )
    assert any(issue.code == "manifest_records_empty" for issue in issues)


def test_recipe_requires_every_eval_only_id_in_forbidden_list() -> None:
    eval_item = dataset("eval_a", role="eval_only", status="locked", repo=None, category="evaluation")
    issues = validate_recipe_against_registry(recipe("train_a"), registry(dataset("train_a"), eval_item))
    assert any(issue.code == "eval_only_not_forbidden" for issue in issues)


def test_eval_config_requires_registered_locked_primary_and_secondary() -> None:
    eval_registry = registry(
        dataset("primary_eval", role="eval_only", status="locked", repo=None, category="evaluation"),
        dataset("secondary_eval", role="eval_only", status="locked", repo=None, category="regression"),
    )
    assert not validate_eval_config(eval_config(), eval_registry)

    bad_config = deepcopy(eval_config())
    bad_config["secondary"] = ["not_registered"]
    issues = validate_eval_config(bad_config, eval_registry)
    assert any(issue.code == "eval_id_unregistered" for issue in issues)


def test_train_config_requires_explicit_dataset_and_target_review(tmp_path: Path) -> None:
    recipe_path = tmp_path / "recipe.yaml"
    recipe_path.write_text("schema_version: 1\nmodel: " + MODEL + "\n", encoding="utf-8")
    train_config = {
        "schema_version": SCHEMA_VERSION,
        "dataset_config": "recipe.yaml",
        "model_name_or_path": MODEL,
        "backend": "reference_trl_peft",
        "method": "qlora",
        "dtype": "bfloat16",
        "run_name": "test-run",
        "target_tokens": 100,
        "sequence": {"max_length": 128, "packing": True},
        "lora": {"r": 4, "alpha": 8, "dropout": 0.05, "target_policy": "auto_review_required"},
        "optimizer": {"name": "adamw_torch", "learning_rate": 0.001, "weight_decay": 0.0, "warmup_ratio": 0.1},
        "logging": {
            "save_environment_manifest": True,
            "save_raw_config": True,
            "save_git_commit": True,
        },
        "safety": {
            "require_dataset_registry_validation": True,
            "require_budget_preflight_on_runpod": True,
            "forbid_eval_only_sources": True,
        },
    }
    issues = validate_train_config(train_config, registry(), repo_root=tmp_path)
    assert any(issue.code == "target_policy_pending" for issue in issues)
