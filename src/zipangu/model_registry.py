"""Backward-compatible import surface for the ZIPANGU model registry."""

from .models import (  # noqa: F401
    ALLOWED_CLASSES,
    DEFAULT_BASE_MODEL_ID,
    DEFAULT_MODEL_ID,
    ModelRegistryError,
    ModelSpec,
    load_model_spec,
    load_model_specs,
    load_variant_registry,
    model_id_from_base,
    validate_model_registry,
)
