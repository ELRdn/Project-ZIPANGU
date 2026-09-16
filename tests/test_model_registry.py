from pathlib import Path

from zipangu.models import (
    DEFAULT_MODEL_ID,
    load_model_specs,
    load_variant_registry,
    validate_model_registry,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_default_model_is_k_and_family_is_complete():
    specs = load_model_specs(REPO_ROOT)
    assert DEFAULT_MODEL_ID == "ZIPANGU-K-I-4B"
    assert set(specs) == {
        "ZIPANGU-K-I-4B",
        "ZIPANGU-C-I-9B",
        "ZIPANGU-Z-I-26B-A4B",
    }
    assert specs[DEFAULT_MODEL_ID].base_model_id == "empero-ai/Qwen3.8-4B-Distill"


def test_model_registry_keeps_z_future_only():
    issues = validate_model_registry(REPO_ROOT)
    assert not issues
    specs = load_model_specs(REPO_ROOT)
    assert specs["ZIPANGU-Z-I-26B-A4B"].training_enabled is False


def test_community_variants_never_become_canonical():
    variants = load_variant_registry(REPO_ROOT)
    assert variants
    assert all(item.get("canonical") is False for item in variants.values())
    assert all(item.get("canonical_id") for item in variants.values())
