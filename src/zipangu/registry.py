from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from .provenance import canonical_json_hash


SCHEMA_VERSION = 1
FORBIDDEN_TRAIN_ROLES = frozenset({"eval_only", "reference_only", "future_cpt_candidate"})
ALLOWED_ROLES = frozenset(
    {"train_candidate", "future_cpt_candidate", "eval_only", "reference_only"}
)
ALLOWED_STATUSES = frozenset({"quarantine", "locked", "approved"})
ALLOWED_TARGET_POLICIES = frozenset({"auto_review_required", "explicit_reviewed"})


@dataclass(frozen=True)
class ValidationIssue:
    """A stable validation result that separates malformed config from run blockers."""

    code: str
    message: str
    path: str = ""
    level: str = "error"

    def __post_init__(self) -> None:
        if self.level not in {"error", "blocked"}:
            raise ValueError(f"unsupported validation level: {self.level}")

    @property
    def is_error(self) -> bool:
        return self.level == "error"

    @property
    def is_blocked(self) -> bool:
        return self.level == "blocked"

    def render(self) -> str:
        location = f" {self.path}" if self.path else ""
        return f"{self.level.upper()} [{self.code}]{location}: {self.message}"


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(float(value))


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


UNKNOWN_MANIFEST_VALUES = frozenset({"", "unknown", "unspecified", "undetermined", "n/a", "na", "null"})


def _known_manifest_string(value: Any) -> bool:
    return _non_empty_string(value) and value.strip().casefold() not in UNKNOWN_MANIFEST_VALUES


def _issue(
    code: str,
    message: str,
    path: str = "",
    *,
    level: str = "error",
) -> ValidationIssue:
    return ValidationIssue(code=code, message=message, path=path, level=level)


def _repo_path(raw_path: Any, repo_root: Path) -> Path | None:
    if not _non_empty_string(raw_path):
        return None
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return None

    root = repo_root.resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return None
    return resolved


def _validate_manifest(
    item: Mapping[str, Any],
    *,
    repo_root: Path,
    issue_path: str,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    provenance = item.get("provenance")
    if not isinstance(provenance, Mapping):
        return [
            _issue(
                "provenance_missing",
                "approved training sources require a provenance mapping",
                issue_path,
                level="blocked",
            )
        ]

    manifest_path = provenance.get("manifest_path")
    manifest_sha256 = provenance.get("manifest_sha256")
    if not _non_empty_string(manifest_path):
        issues.append(
            _issue(
                "manifest_path_missing",
                "provenance.manifest_path must be a repo-relative path",
                issue_path,
                level="blocked",
            )
        )
    if not _non_empty_string(manifest_sha256) or len(manifest_sha256) != 64:
        issues.append(
            _issue(
                "manifest_hash_invalid",
                "provenance.manifest_sha256 must be a 64-character SHA-256 hex string",
                issue_path,
                level="blocked",
            )
        )
    elif any(character not in "0123456789abcdef" for character in manifest_sha256.lower()):
        issues.append(
            _issue(
                "manifest_hash_invalid",
                "provenance.manifest_sha256 must contain only hexadecimal characters",
                issue_path,
                level="blocked",
            )
        )

    normalized_manifest_path = str(manifest_path).replace("\\", "/") if _non_empty_string(manifest_path) else ""
    expected_manifest_name = f"{item.get('id', '')}.json"
    if normalized_manifest_path and (
        not normalized_manifest_path.startswith("configs/datasets/manifests/")
        or Path(normalized_manifest_path).name != expected_manifest_name
    ):
        issues.append(
            _issue(
                "manifest_path_invalid",
                "approved provenance must use configs/datasets/manifests/<dataset_id>.json",
                issue_path,
                level="blocked",
            )
        )
    manifest_file = _repo_path(manifest_path, repo_root)
    if manifest_file is None:
        if _non_empty_string(manifest_path):
            issues.append(
                _issue(
                    "manifest_path_invalid",
                    "manifest_path must stay inside the repository and be relative",
                    issue_path,
                    level="blocked",
                )
            )
        return issues
    if not manifest_file.is_file():
        issues.append(
            _issue(
                "manifest_missing",
                f"manifest file does not exist: {manifest_path}",
                issue_path,
                level="blocked",
            )
        )
        return issues

    try:
        with manifest_file.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        issues.append(
            _issue(
                "manifest_unreadable",
                f"could not read JSON manifest: {exc}",
                issue_path,
                level="blocked",
            )
        )
        return issues

    if not isinstance(manifest, Mapping):
        return issues + [
            _issue(
                "manifest_root_invalid",
                "manifest root must be a JSON object",
                issue_path,
                level="blocked",
            )
        ]

    expected_hash = provenance.get("manifest_sha256")
    if _non_empty_string(expected_hash) and canonical_json_hash(dict(manifest)) != expected_hash.lower():
        issues.append(
            _issue(
                "manifest_hash_mismatch",
                "manifest SHA-256 does not match canonical manifest content",
                issue_path,
                level="blocked",
            )
        )

    required_manifest_strings = (
        "dataset_id",
        "source_repo",
        "source_config",
        "source_revision",
        "source_split",
        "license",
        "language",
        "category",
        "contamination_status",
    )
    for field in required_manifest_strings:
        value = manifest.get(field)
        if not _non_empty_string(value):
            issues.append(
                _issue(
                    "manifest_field_missing",
                    f"manifest field must be a non-empty string: {field}",
                    issue_path,
                    level="blocked",
                )
            )
        elif field in {"source_revision", "source_split", "license"} and not _known_manifest_string(value):
            issues.append(
                _issue(
                    "manifest_field_unknown",
                    f"manifest field must be known before training: {field}",
                    issue_path,
                    level="blocked",
                )
            )

    if manifest.get("schema_version") != SCHEMA_VERSION:
        issues.append(
            _issue(
                "manifest_schema_version",
                f"manifest schema_version must be {SCHEMA_VERSION}",
                issue_path,
                level="blocked",
            )
        )
    if manifest.get("dataset_id") != item.get("id"):
        issues.append(
            _issue(
                "manifest_dataset_mismatch",
                "manifest dataset_id does not match registry id",
                issue_path,
                level="blocked",
            )
        )
    if _non_empty_string(item.get("repo")) and manifest.get("source_repo") != item.get("repo"):
        issues.append(
            _issue(
                "manifest_repo_mismatch",
                "manifest source_repo does not match registry repo",
                issue_path,
                level="blocked",
            )
        )
    if manifest.get("category") != item.get("category"):
        issues.append(
            _issue(
                "manifest_category_mismatch",
                "manifest category does not match registry category",
                issue_path,
                level="blocked",
            )
        )
    if manifest.get("contamination_status") != "clear":
        issues.append(
            _issue(
                "contamination_not_clear",
                "contamination_status must be clear before training",
                issue_path,
                level="blocked",
            )
        )

    transform_chain = manifest.get("transform_chain")
    if not isinstance(transform_chain, list) or not all(
        _non_empty_string(step) for step in transform_chain
    ):
        issues.append(
            _issue(
                "transform_chain_invalid",
                "transform_chain must be a list of non-empty strings",
                issue_path,
                level="blocked",
            )
        )

    records = manifest.get("records")
    if not isinstance(records, list):
        issues.append(
            _issue(
                "manifest_records_invalid",
                "records must be a list of source_row_id/content_hash objects",
                issue_path,
                level="blocked",
            )
        )
        return issues
    if not records:
        issues.append(
            _issue(
                "manifest_records_empty",
                "approved training manifests must contain at least one record",
                issue_path,
                level="blocked",
            )
        )

    seen_row_ids: set[str] = set()
    for index, record in enumerate(records):
        record_path = f"{issue_path}.records[{index}]"
        if not isinstance(record, Mapping):
            issues.append(
                _issue(
                    "manifest_record_invalid",
                    "record must be an object",
                    record_path,
                    level="blocked",
                )
            )
            continue
        row_id = record.get("source_row_id")
        content_hash = record.get("content_hash")
        if not _non_empty_string(row_id):
            issues.append(
                _issue(
                    "source_row_id_missing",
                    "source_row_id must be a non-empty string",
                    record_path,
                    level="blocked",
                )
            )
        elif row_id in seen_row_ids:
            issues.append(
                _issue(
                    "source_row_id_duplicate",
                    f"duplicate source_row_id: {row_id}",
                    record_path,
                    level="blocked",
                )
            )
        else:
            seen_row_ids.add(row_id)
        if not _non_empty_string(content_hash) or len(content_hash) != 64 or any(
            character not in "0123456789abcdef" for character in content_hash.lower()
        ):
            issues.append(
                _issue(
                    "content_hash_invalid",
                    "content_hash must be a 64-character SHA-256 hex string",
                    record_path,
                    level="blocked",
                )
            )
    return issues


def validate_registry(registry: Mapping[str, Any]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if not isinstance(registry, Mapping):
        return [_issue("registry_root_invalid", "registry root must be a mapping")]
    if registry.get("schema_version") != SCHEMA_VERSION:
        issues.append(
            _issue(
                "registry_schema_version",
                f"registry schema_version must be {SCHEMA_VERSION}",
                "registry.schema_version",
            )
        )

    datasets = registry.get("datasets")
    if not isinstance(datasets, list):
        return issues + [_issue("datasets_invalid", "registry.datasets must be a list", "datasets")]

    seen: set[str] = set()
    for index, item in enumerate(datasets):
        item_path = f"datasets[{index}]"
        if not isinstance(item, Mapping):
            issues.append(_issue("dataset_entry_invalid", "dataset entry must be an object", item_path))
            continue

        dataset_id = item.get("id")
        if not _non_empty_string(dataset_id):
            issues.append(_issue("dataset_id_missing", "dataset id must be a non-empty string", item_path))
            continue
        if dataset_id in seen:
            issues.append(_issue("duplicate_dataset_id", f"duplicate dataset id: {dataset_id}", item_path))
        seen.add(dataset_id)

        role = item.get("role")
        if role not in ALLOWED_ROLES:
            issues.append(_issue("unknown_role", f"unsupported role: {role}", f"{item_path}.role"))
        status = item.get("status")
        if status not in ALLOWED_STATUSES:
            issues.append(_issue("unknown_status", f"unsupported status: {status}", f"{item_path}.status"))

        if not _non_empty_string(item.get("category")):
            issues.append(
                _issue("category_missing", "dataset category must be a non-empty string", item_path)
            )
        if "repo" not in item:
            issues.append(_issue("repo_missing", "dataset repo key is required", item_path))
        elif item.get("repo") is not None and not _non_empty_string(item.get("repo")):
            issues.append(_issue("repo_invalid", "dataset repo must be a string or null", item_path))
        elif role in {"train_candidate", "future_cpt_candidate", "reference_only"} and not _non_empty_string(
            item.get("repo")
        ):
            issues.append(
                _issue(
                    "repo_missing",
                    "non-evaluation datasets require a non-empty repo",
                    item_path,
                )
            )

        if role in {"eval_only", "reference_only"} and status != "locked":
            issues.append(
                _issue(
                    "protected_dataset_not_locked",
                    f"{role} dataset must be locked: {dataset_id}",
                    item_path,
                )
            )
        if role == "future_cpt_candidate" and status != "quarantine":
            issues.append(
                _issue(
                    "future_candidate_not_quarantined",
                    "future_cpt_candidate must remain quarantined until a separate CPT review",
                    item_path,
                )
            )
        if status == "approved" and role != "train_candidate":
            issues.append(
                _issue(
                    "approved_role_invalid",
                    "only train_candidate entries may be approved for training",
                    item_path,
                )
            )
        if "provenance" in item and not isinstance(item.get("provenance"), Mapping):
            issues.append(
                _issue(
                    "provenance_invalid",
                    "provenance must be an object when present",
                    f"{item_path}.provenance",
                )
            )
    return issues


def _registry_by_id(registry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    datasets = registry.get("datasets", [])
    return {
        item["id"]: item
        for item in datasets
        if isinstance(item, Mapping) and _non_empty_string(item.get("id"))
    }


def validate_recipe_against_registry(
    recipe: Mapping[str, Any],
    registry: Mapping[str, Any],
    *,
    repo_root: str | Path = ".",
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    recipe_label = str(recipe.get("name", "recipe")) if isinstance(recipe, Mapping) else "recipe"
    if not isinstance(recipe, Mapping):
        return [_issue("recipe_root_invalid", "recipe root must be a mapping", recipe_label)]
    if recipe.get("schema_version") != SCHEMA_VERSION:
        issues.append(
            _issue(
                "recipe_schema_version",
                f"recipe schema_version must be {SCHEMA_VERSION}",
                recipe_label,
            )
        )
    for field in ("name", "model", "objective"):
        if not _non_empty_string(recipe.get(field)):
            issues.append(
                _issue(
                    "recipe_field_missing",
                    f"recipe field must be a non-empty string: {field}",
                    recipe_label,
                )
            )

    sampling_mass = recipe.get("sampling_mass")
    buckets = recipe.get("buckets")
    if not isinstance(sampling_mass, Mapping):
        issues.append(_issue("sampling_mass_invalid", "sampling_mass must be an object", recipe_label))
        sampling_mass = {}
    if not isinstance(buckets, Mapping):
        issues.append(_issue("buckets_invalid", "buckets must be an object", recipe_label))
        buckets = {}

    total_mass = 0.0
    for bucket_name, mass in sampling_mass.items():
        if not _non_empty_string(bucket_name) or not _is_number(mass) or not 0 < float(mass) <= 1:
            issues.append(
                _issue(
                    "sampling_mass_value_invalid",
                    f"sampling mass must be a number in (0, 1]: {bucket_name}",
                    recipe_label,
                )
            )
        else:
            total_mass += float(mass)
    if sampling_mass and abs(total_mass - 1.0) > 1e-6:
        issues.append(
            _issue(
                "sampling_mass_sum_invalid",
                f"sampling mass must sum to 1.0, got {total_mass:.8f}",
                recipe_label,
            )
        )
    for bucket_name in sampling_mass:
        if bucket_name not in buckets:
            issues.append(
                _issue("sampling_bucket_missing", f"sampling bucket is not defined: {bucket_name}", recipe_label)
            )
    for bucket_name in buckets:
        if bucket_name not in sampling_mass:
            issues.append(
                _issue("sampling_mass_missing", f"bucket has no sampling mass: {bucket_name}", recipe_label)
            )

    registry_by_id = _registry_by_id(registry)
    all_candidates: set[str] = set()
    for bucket_name, bucket in buckets.items():
        bucket_path = f"{recipe_label}.buckets.{bucket_name}"
        if not isinstance(bucket, Mapping):
            issues.append(_issue("bucket_invalid", "bucket must be an object", bucket_path))
            continue
        candidates = bucket.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            issues.append(_issue("candidates_invalid", "bucket candidates must be a non-empty list", bucket_path))
            candidates = []
        target_weight = bucket.get("target_weight")
        if not _is_number(target_weight) or not 0 < float(target_weight) <= 1:
            issues.append(_issue("target_weight_invalid", "target_weight must be a number in (0, 1]", bucket_path))
        elif bucket_name in sampling_mass and abs(float(target_weight) - float(sampling_mass[bucket_name])) > 1e-6:
            issues.append(
                _issue(
                    "target_weight_mismatch",
                    "bucket target_weight must match sampling_mass",
                    bucket_path,
                )
            )

        local_candidates: set[str] = set()
        for candidate in candidates:
            candidate_path = f"{bucket_path}.candidates"
            if not _non_empty_string(candidate):
                issues.append(_issue("candidate_id_invalid", "candidate IDs must be non-empty strings", candidate_path))
                continue
            if candidate in local_candidates:
                issues.append(_issue("duplicate_candidate_id", f"duplicate candidate in bucket: {candidate}", candidate_path))
            local_candidates.add(candidate)
            if candidate in all_candidates:
                issues.append(_issue("candidate_in_multiple_buckets", f"candidate appears in multiple buckets: {candidate}", candidate_path))
            all_candidates.add(candidate)

            item = registry_by_id.get(candidate)
            if item is None:
                issues.append(_issue("unknown_dataset", f"unknown dataset: {candidate}", candidate_path))
                continue
            role = item.get("role")
            if role in FORBIDDEN_TRAIN_ROLES:
                issues.append(
                    _issue(
                        "forbidden_training_source",
                        f"forbidden training source: {candidate}",
                        candidate_path,
                    )
                )
            elif item.get("status") != "approved":
                issues.append(
                    _issue(
                        "training_source_not_approved",
                        f"training source is not approved: {candidate}",
                        candidate_path,
                        level="blocked",
                    )
                )
            else:
                issues.extend(
                    _validate_manifest(
                        item,
                        repo_root=Path(repo_root),
                        issue_path=f"registry.datasets[{candidate}]",
                    )
                )

    english_policy = recipe.get("english_trace_policy")
    if not isinstance(english_policy, Mapping):
        issues.append(_issue("english_trace_policy_invalid", "english_trace_policy must be an object", recipe_label))
    else:
        for field in ("keep_original_subset", "create_japanese_rewrite_subset", "preserve_pair_provenance"):
            if not isinstance(english_policy.get(field), bool):
                issues.append(_issue("english_trace_policy_field", f"{field} must be boolean", recipe_label))

    forbidden = recipe.get("eval_only_forbidden")
    if not isinstance(forbidden, list):
        issues.append(_issue("eval_only_forbidden_invalid", "eval_only_forbidden must be a list", recipe_label))
        forbidden = []
    forbidden_ids: set[str] = set()
    for dataset_id in forbidden:
        if not _non_empty_string(dataset_id):
            issues.append(_issue("forbidden_id_invalid", "eval_only_forbidden IDs must be strings", recipe_label))
            continue
        if dataset_id in forbidden_ids:
            issues.append(_issue("duplicate_forbidden_id", f"duplicate forbidden ID: {dataset_id}", recipe_label))
        forbidden_ids.add(dataset_id)
        item = registry_by_id.get(dataset_id)
        if item is None:
            issues.append(_issue("unknown_forbidden_id", f"unknown eval-only ID: {dataset_id}", recipe_label))
        elif item.get("role") != "eval_only":
            issues.append(_issue("non_eval_forbidden_id", f"forbidden list contains non-eval dataset: {dataset_id}", recipe_label))

    expected_eval_ids = {
        dataset_id
        for dataset_id, item in registry_by_id.items()
        if item.get("role") == "eval_only"
    }
    for dataset_id in sorted(expected_eval_ids - forbidden_ids):
        issues.append(
            _issue(
                "eval_only_not_forbidden",
                f"recipe must forbid registry eval-only dataset: {dataset_id}",
                recipe_label,
            )
        )
    return issues


def _validate_eval_reference(
    reference: Any,
    registry_by_id: Mapping[str, Mapping[str, Any]],
    *,
    path: str,
    primary: bool,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if primary:
        if not isinstance(reference, Mapping):
            return [_issue("primary_entry_invalid", "primary evaluation entry must be an object", path)]
        dataset_id = reference.get("id")
        if reference.get("role") != "eval_only":
            issues.append(_issue("primary_role_invalid", "primary evaluation role must be eval_only", path))
        repeat = reference.get("repeat")
        if not _is_positive_int(repeat):
            issues.append(_issue("repeat_invalid", "repeat must be a positive integer", path))
    else:
        dataset_id = reference
    if not _non_empty_string(dataset_id):
        issues.append(_issue("eval_id_invalid", "evaluation ID must be a non-empty string", path))
        return issues
    item = registry_by_id.get(dataset_id)
    if item is None:
        issues.append(_issue("eval_id_unregistered", f"evaluation ID is not registered: {dataset_id}", path))
    elif item.get("role") != "eval_only" or item.get("status") != "locked":
        issues.append(_issue("eval_source_not_locked", f"evaluation source must be eval_only and locked: {dataset_id}", path))
    return issues


def validate_eval_config(
    eval_config: Mapping[str, Any],
    registry: Mapping[str, Any],
    *,
    config_path: str = "eval",
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if not isinstance(eval_config, Mapping):
        return [_issue("eval_config_root_invalid", "evaluation config root must be an object", config_path)]
    if eval_config.get("schema_version") != SCHEMA_VERSION:
        issues.append(_issue("eval_schema_version", f"evaluation schema_version must be {SCHEMA_VERSION}", config_path))

    target_model = eval_config.get("target_model")
    if not isinstance(target_model, Mapping):
        issues.append(_issue("target_model_invalid", "target_model must be an object", config_path))
    else:
        for field in ("candidate", "reference", "base"):
            if not _non_empty_string(target_model.get(field)):
                issues.append(_issue("target_model_field", f"target_model.{field} must be a non-empty string", config_path))

    registry_by_id = _registry_by_id(registry)
    primary = eval_config.get("primary")
    if not isinstance(primary, list) or not primary:
        issues.append(_issue("primary_invalid", "primary must be a non-empty list", config_path))
        primary = []
    for index, reference in enumerate(primary):
        issues.extend(
            _validate_eval_reference(
                reference,
                registry_by_id,
                path=f"{config_path}.primary[{index}]",
                primary=True,
            )
        )

    secondary = eval_config.get("secondary")
    if not isinstance(secondary, list) or not secondary:
        issues.append(_issue("secondary_invalid", "secondary must be a non-empty list", config_path))
        secondary = []
    for index, reference in enumerate(secondary):
        issues.extend(
            _validate_eval_reference(
                reference,
                registry_by_id,
                path=f"{config_path}.secondary[{index}]",
                primary=False,
            )
        )

    contamination_gate = eval_config.get("contamination_gate")
    required_contamination_checks = ("exact_hash", "substring", "ngram", "minhash", "manual_review_top_hits")
    if not isinstance(contamination_gate, Mapping):
        issues.append(_issue("contamination_gate_invalid", "contamination_gate must be an object", config_path))
    else:
        for field in required_contamination_checks:
            if contamination_gate.get(field) is not True:
                issues.append(_issue("contamination_check_disabled", f"contamination check must be true: {field}", config_path))
    if eval_config.get("preserve_raw_generations") is not True:
        issues.append(_issue("raw_generations_disabled", "preserve_raw_generations must be true", config_path))
    if eval_config.get("publish_raw_generations") is not False:
        issues.append(_issue("raw_generations_publish_enabled", "publish_raw_generations must be false", config_path))

    victory = eval_config.get("victory")
    if not isinstance(victory, Mapping):
        issues.append(_issue("victory_invalid", "victory must be an object", config_path))
    else:
        minimum = victory.get("minimum_primary_wins")
        total = victory.get("total_primary")
        if not _is_positive_int(minimum) or not _is_positive_int(total) or minimum > total:
            issues.append(_issue("victory_threshold_invalid", "victory thresholds must be positive and ordered", config_path))
        elif total != len(primary):
            issues.append(_issue("victory_total_mismatch", "victory.total_primary must match primary count", config_path))
        if victory.get("require_general_regression_gate") is not True:
            issues.append(_issue("regression_gate_disabled", "general regression gate must be true", config_path))

    tolerance = eval_config.get("regression_tolerance")
    if tolerance is not None and (not _is_number(tolerance) or float(tolerance) < 0):
        issues.append(_issue("regression_tolerance_invalid", "regression_tolerance must be null or a non-negative number", config_path))
    return issues


def _load_recipe_for_train_config(
    train_config: Mapping[str, Any],
    repo_root: Path,
    path: str,
) -> tuple[dict[str, Any] | None, list[ValidationIssue]]:
    dataset_config = train_config.get("dataset_config")
    recipe_path = _repo_path(dataset_config, repo_root)
    if recipe_path is None:
        return None, [_issue("dataset_config_invalid", "dataset_config must be a repo-relative path", path)]
    if not recipe_path.is_file():
        return None, [_issue("dataset_config_missing", f"dataset config does not exist: {dataset_config}", path)]
    try:
        return load_yaml(recipe_path), []
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return None, [_issue("dataset_config_unreadable", str(exc), path)]


def validate_train_config(
    train_config: Mapping[str, Any],
    registry: Mapping[str, Any],
    *,
    repo_root: str | Path = ".",
    config_path: str = "train",
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if not isinstance(train_config, Mapping):
        return [_issue("train_config_root_invalid", "train config root must be an object", config_path)]
    if train_config.get("schema_version") != SCHEMA_VERSION:
        issues.append(_issue("train_schema_version", f"train schema_version must be {SCHEMA_VERSION}", config_path))
    for field in ("model_name_or_path", "backend", "method", "dtype", "run_name"):
        if not _non_empty_string(train_config.get(field)):
            issues.append(_issue("train_field_missing", f"train field must be a non-empty string: {field}", config_path))
    target_tokens = train_config.get("target_tokens")
    if not _is_positive_int(target_tokens):
        issues.append(_issue("target_tokens_invalid", "target_tokens must be a positive integer", config_path))

    repo_root_path = Path(repo_root)
    recipe, recipe_issues = _load_recipe_for_train_config(train_config, repo_root_path, config_path)
    issues.extend(recipe_issues)
    if recipe is not None:
        recipe_model = recipe.get("model")
        if _non_empty_string(train_config.get("model_name_or_path")) and train_config.get("model_name_or_path") != recipe_model:
            issues.append(_issue("model_mismatch", "train model_name_or_path must match dataset recipe model", config_path))
        issues.extend(validate_recipe_against_registry(recipe, registry, repo_root=repo_root_path))

    sequence = train_config.get("sequence")
    if not isinstance(sequence, Mapping):
        issues.append(_issue("sequence_invalid", "sequence must be an object", config_path))
    else:
        if not _is_positive_int(sequence.get("max_length")):
            issues.append(_issue("max_length_invalid", "sequence.max_length must be a positive integer", config_path))
        if not isinstance(sequence.get("packing"), bool):
            issues.append(_issue("packing_invalid", "sequence.packing must be boolean", config_path))

    lora = train_config.get("lora")
    if not isinstance(lora, Mapping):
        issues.append(_issue("lora_invalid", "lora must be an object", config_path))
    else:
        for field in ("r", "alpha"):
            if not _is_positive_int(lora.get(field)):
                issues.append(_issue("lora_value_invalid", f"lora.{field} must be a positive integer", config_path))
        dropout = lora.get("dropout")
        if not _is_number(dropout) or not 0 <= float(dropout) < 1:
            issues.append(_issue("lora_dropout_invalid", "lora.dropout must be in [0, 1)", config_path))
        target_policy = lora.get("target_policy")
        if target_policy not in ALLOWED_TARGET_POLICIES:
            issues.append(_issue("target_policy_invalid", f"unsupported target policy: {target_policy}", config_path))
        elif target_policy == "auto_review_required":
            issues.append(
                _issue(
                    "target_policy_pending",
                    "explicit_reviewed target policy is required before training",
                    config_path,
                    level="blocked",
                )
            )
        else:
            target_modules = lora.get("target_modules")
            if not isinstance(target_modules, list) or not target_modules or not all(
                _non_empty_string(module) for module in target_modules
            ):
                issues.append(
                    _issue(
                        "target_modules_missing",
                        "explicit_reviewed target policy requires a non-empty target_modules list",
                        config_path,
                        level="blocked",
                    )
                )

    optimizer = train_config.get("optimizer")
    if not isinstance(optimizer, Mapping):
        issues.append(_issue("optimizer_invalid", "optimizer must be an object", config_path))
    else:
        if not _non_empty_string(optimizer.get("name")):
            issues.append(_issue("optimizer_name_missing", "optimizer.name is required", config_path))
        if not _is_number(optimizer.get("learning_rate")) or float(optimizer["learning_rate"]) <= 0:
            issues.append(_issue("learning_rate_invalid", "optimizer.learning_rate must be positive", config_path))
        if not _is_number(optimizer.get("weight_decay")) or float(optimizer["weight_decay"]) < 0:
            issues.append(_issue("weight_decay_invalid", "optimizer.weight_decay must be non-negative", config_path))
        if not _is_number(optimizer.get("warmup_ratio")) or not 0 <= float(optimizer["warmup_ratio"]) <= 1:
            issues.append(_issue("warmup_ratio_invalid", "optimizer.warmup_ratio must be in [0, 1]", config_path))

    for section_name, required_fields in {
        "logging": ("save_environment_manifest", "save_raw_config", "save_git_commit"),
        "safety": (
            "require_dataset_registry_validation",
            "require_budget_preflight_on_runpod",
            "forbid_eval_only_sources",
        ),
    }.items():
        section = train_config.get(section_name)
        if not isinstance(section, Mapping):
            issues.append(_issue(f"{section_name}_invalid", f"{section_name} must be an object", config_path))
            continue
        for field in required_fields:
            if section.get(field) is not True:
                issues.append(_issue(f"{section_name}_flag_disabled", f"{section_name}.{field} must be true", config_path))
    return issues


def split_issues(issues: Iterable[ValidationIssue]) -> tuple[list[ValidationIssue], list[ValidationIssue]]:
    errors: list[ValidationIssue] = []
    blocked: list[ValidationIssue] = []
    for issue in issues:
        (blocked if issue.is_blocked else errors).append(issue)
    return errors, blocked
