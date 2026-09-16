"""Canonical Project ZIPANGU model-family registry.

The model registry is deliberately separate from dataset recipes. A recipe
describes Generation-I data policy, while this module describes the model
class and the exact upstream base for one experiment. Community
Heretic/Abliterated variants are non-canonical records and cannot resolve as
a canonical base by accident.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


MODEL_REGISTRY_RELATIVE_PATH = Path("configs") / "models" / "registry.yaml"
DEFAULT_MODEL_ID = "ZIPANGU-K-I-4B"
DEFAULT_BASE_MODEL_ID = "empero-ai/Qwen3.8-4B-Distill"
ALLOWED_CLASSES = frozenset({"K", "C", "Z"})
ALLOWED_GENERATIONS = frozenset({"I", "RO", "HA", "NI", "HO", "HE", "TO", "CHI", "RI", "NU"})


class ModelRegistryError(ValueError):
    """The canonical model registry is malformed or ambiguous."""


@dataclass(frozen=True)
class ModelSpec:
    """Resolved metadata for one canonical ZIPANGU model."""

    id: str
    class_code: str
    generation: str
    size: str
    base_repo: str
    revision: str | None
    architecture_family: str
    architecture_type: str
    parameters: str | None
    total_parameters: str | None
    active_parameters: str | None
    research_role: tuple[str, ...]
    training_enabled: bool
    preferred_method: str
    compare_base: bool
    compare_llm_jp: str
    canonical: bool = True
    canonical_id: str | None = None
    variant_type: str | None = None

    @property
    def is_future_only(self) -> bool:
        return not self.training_enabled

    @property
    def artifact_slug(self) -> str:
        return self.id.casefold().replace("zipangu-", "", 1).replace("-", "_")

    @property
    def base_model_id(self) -> str:
        """Compatibility alias for code that calls an upstream repo a model id."""

        return self.base_repo

    def as_dict(self) -> dict[str, Any]:
        architecture: dict[str, Any] = {"family": self.architecture_family, "type": self.architecture_type}
        for key, value in (
            ("parameters", self.parameters),
            ("total_parameters", self.total_parameters),
            ("active_parameters", self.active_parameters),
        ):
            if value is not None:
                architecture[key] = value
        return {
            "schema_version": 1,
            "id": self.id,
            "class": self.class_code,
            "generation": self.generation,
            "size": self.size,
            "base_model": {"repo": self.base_repo, "revision": self.revision},
            "architecture": architecture,
            "research_role": list(self.research_role),
            "training": {"enabled": self.training_enabled, "preferred_method": self.preferred_method},
            "evaluation": {"compare_base": self.compare_base, "compare_llm_jp": self.compare_llm_jp},
            "canonical": self.canonical,
            **({"canonical_id": self.canonical_id} if self.canonical_id else {}),
            **({"variant_type": self.variant_type} if self.variant_type else {}),
        }


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ModelRegistryError(f"could not read model config: {path}") from exc
    if not isinstance(value, Mapping):
        raise ModelRegistryError(f"model config root must be a mapping: {path}")
    return dict(value)


def _required_string(value: Any, field: str, path: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelRegistryError(f"{path}: {field} must be a non-empty string")
    return value.strip()


def _parse_spec(payload: Mapping[str, Any], *, path: Path, canonical: bool = True) -> ModelSpec:
    model_id = _required_string(payload.get("id"), "id", path)
    class_code = _required_string(payload.get("class"), "class", path).upper()
    generation = _required_string(payload.get("generation"), "generation", path).upper()
    if class_code not in ALLOWED_CLASSES:
        raise ModelRegistryError(f"{path}: unsupported model class: {class_code}")
    if generation not in ALLOWED_GENERATIONS:
        raise ModelRegistryError(f"{path}: unsupported generation: {generation}")
    expected_prefix = f"ZIPANGU-{class_code}-{generation}-"
    if not model_id.startswith(expected_prefix):
        raise ModelRegistryError(f"{path}: model id must start with {expected_prefix}: {model_id}")
    base = payload.get("base_model")
    if not isinstance(base, Mapping):
        raise ModelRegistryError(f"{path}: base_model must be a mapping")
    architecture = payload.get("architecture")
    if not isinstance(architecture, Mapping):
        raise ModelRegistryError(f"{path}: architecture must be a mapping")
    training = payload.get("training")
    if not isinstance(training, Mapping):
        raise ModelRegistryError(f"{path}: training must be a mapping")
    evaluation = payload.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ModelRegistryError(f"{path}: evaluation must be a mapping")
    roles = payload.get("research_role", [])
    if not isinstance(roles, list) or not all(isinstance(role, str) and role.strip() for role in roles):
        raise ModelRegistryError(f"{path}: research_role must be a list of non-empty strings")
    enabled = training.get("enabled")
    if not isinstance(enabled, bool):
        raise ModelRegistryError(f"{path}: training.enabled must be boolean")
    compare_base = evaluation.get("compare_base")
    if not isinstance(compare_base, bool):
        raise ModelRegistryError(f"{path}: evaluation.compare_base must be boolean")
    revision = base.get("revision")
    if revision is not None and (not isinstance(revision, str) or not revision.strip()):
        raise ModelRegistryError(f"{path}: base_model.revision must be null or a non-empty string")
    return ModelSpec(
        id=model_id,
        class_code=class_code,
        generation=generation,
        size=_required_string(payload.get("size"), "size", path),
        base_repo=_required_string(base.get("repo"), "base_model.repo", path),
        revision=revision.strip() if isinstance(revision, str) else None,
        architecture_family=_required_string(architecture.get("family"), "architecture.family", path),
        architecture_type=_required_string(architecture.get("type"), "architecture.type", path),
        parameters=str(architecture["parameters"]) if architecture.get("parameters") is not None else None,
        total_parameters=str(architecture["total_parameters"])
        if architecture.get("total_parameters") is not None
        else None,
        active_parameters=str(architecture["active_parameters"])
        if architecture.get("active_parameters") is not None
        else None,
        research_role=tuple(str(role) for role in roles),
        training_enabled=enabled,
        preferred_method=_required_string(training.get("preferred_method"), "training.preferred_method", path),
        compare_base=compare_base,
        compare_llm_jp=str(evaluation.get("compare_llm_jp", "optional")),
        canonical=canonical,
        canonical_id=str(payload.get("canonical_id")) if payload.get("canonical_id") else None,
        variant_type=str(payload.get("variant_type")) if payload.get("variant_type") else None,
    )


def load_model_specs(repo_root: str | Path) -> dict[str, ModelSpec]:
    """Load and validate all canonical model configs."""

    root = Path(repo_root).resolve()
    registry_path = root / MODEL_REGISTRY_RELATIVE_PATH
    registry = _load_yaml(registry_path)
    if registry.get("schema_version") != 1:
        raise ModelRegistryError(f"{registry_path}: schema_version must be 1")
    entries = registry.get("models")
    if not isinstance(entries, list) or not entries:
        raise ModelRegistryError(f"{registry_path}: models must be a non-empty list")
    specs: dict[str, ModelSpec] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ModelRegistryError(f"{registry_path}: model entry must be a mapping")
        config_ref = _required_string(entry.get("config"), "models[].config", registry_path)
        config_path = root / config_ref
        spec = _parse_spec(_load_yaml(config_path), path=config_path)
        if spec.id in specs:
            raise ModelRegistryError(f"duplicate canonical model id: {spec.id}")
        if entry.get("canonical") is not True:
            raise ModelRegistryError(f"{registry_path}: canonical model must set canonical=true: {spec.id}")
        specs[spec.id] = spec
    default_model = registry.get("default_model")
    if default_model not in specs:
        raise ModelRegistryError(f"{registry_path}: default_model is not registered: {default_model}")
    for required_id in ("ZIPANGU-K-I-4B", "ZIPANGU-C-I-9B", "ZIPANGU-Z-I-26B-A4B"):
        if required_id not in specs:
            raise ModelRegistryError(f"{registry_path}: required model is missing: {required_id}")
    return specs


def load_model_spec(repo_root: str | Path, model_id: str | None = None) -> ModelSpec:
    specs = load_model_specs(repo_root)
    selected = model_id or DEFAULT_MODEL_ID
    try:
        return specs[selected]
    except KeyError as exc:
        raise ModelRegistryError(f"unknown canonical ZIPANGU model: {selected}") from exc


def load_variant_registry(repo_root: str | Path) -> dict[str, dict[str, Any]]:
    path = Path(repo_root).resolve() / "configs" / "models" / "variants.yaml"
    payload = _load_yaml(path)
    if payload.get("schema_version") != 1:
        raise ModelRegistryError(f"{path}: schema_version must be 1")
    values = payload.get("variants")
    if not isinstance(values, list):
        raise ModelRegistryError(f"{path}: variants must be a list")
    result: dict[str, dict[str, Any]] = {}
    for value in values:
        if not isinstance(value, Mapping):
            raise ModelRegistryError(f"{path}: variant entry must be a mapping")
        variant_id = _required_string(value.get("id"), "variants[].id", path)
        if variant_id in result:
            raise ModelRegistryError(f"duplicate variant id: {variant_id}")
        if value.get("canonical") is not False:
            raise ModelRegistryError(f"{path}: variants must set canonical=false: {variant_id}")
        result[variant_id] = dict(value)
    return result


def validate_model_registry(repo_root: str | Path) -> list[str]:
    """Return human-readable structural issues without mutating any state."""

    try:
        specs = load_model_specs(repo_root)
    except ModelRegistryError as exc:
        return [str(exc)]
    issues: list[str] = []
    for model_id, spec in specs.items():
        if not model_id.startswith("ZIPANGU-"):
            issues.append(f"{model_id}: canonical id must start with ZIPANGU-")
        if spec.class_code == "Z" and spec.training_enabled:
            issues.append(f"{model_id}: Z training must remain disabled during migration")
        if spec.canonical_id or spec.variant_type:
            issues.append(f"{model_id}: canonical model contains variant-only fields")
    try:
        variants = load_variant_registry(repo_root)
    except ModelRegistryError as exc:
        return issues + [str(exc)]
    for variant_id, variant in variants.items():
        canonical_id = variant.get("canonical_id")
        if not isinstance(canonical_id, str) or canonical_id not in specs:
            issues.append(f"{variant_id}: canonical_id must reference a registered canonical model")
        if variant_id in specs:
            issues.append(f"{variant_id}: variant id is also registered as canonical")
        if not isinstance(variant.get("source"), str) or not variant["source"].strip():
            issues.append(f"{variant_id}: variant source must be a non-empty repository id")
        if not isinstance(variant.get("release_policy"), str) or not variant["release_policy"].strip():
            issues.append(f"{variant_id}: release_policy is required")
    return issues


def model_id_from_base(repo: str, repo_root: str | Path) -> str | None:
    """Resolve a known upstream repository to its canonical ZIPANGU id."""

    return next((spec.id for spec in load_model_specs(repo_root).values() if spec.base_repo == repo), None)
