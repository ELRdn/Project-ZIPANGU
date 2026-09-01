from __future__ import annotations

import hashlib
import json
from typing import Any


def normalized_text_hash(text: str) -> str:
    normalized = " ".join(text.strip().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def canonical_json_hash(record: dict[str, Any]) -> str:
    blob = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
