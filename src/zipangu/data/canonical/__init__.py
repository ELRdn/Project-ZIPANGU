"""Compatibility facade adding parse-error provenance to canonical rows."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


_LEGACY_PATH = Path(__file__).resolve().parents[1] / "canonical.py"
_SPEC = importlib.util.spec_from_file_location("zipangu.data._legacy_canonical", _LEGACY_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - packaging failure
    raise ImportError(f"could not load canonical implementation: {_LEGACY_PATH}")
_legacy = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _legacy
_SPEC.loader.exec_module(_legacy)

CANONICAL_FIELDS = _legacy.CANONICAL_FIELDS
canonical_record_fingerprint = _legacy.canonical_record_fingerprint
simhash_text = _legacy.simhash_text


def canonicalize_source_row(source_row: Any, *, source_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    record = _legacy.canonicalize_source_row(source_row, source_metadata=source_metadata)
    if source_row.raw.get("_parse_error"):
        record["conversion_error"] = source_row.raw["_parse_error"]
    if source_row.raw.get("_invalid_row") is not None:
        record["conversion_error"] = "row is not a JSON object"
    return record
