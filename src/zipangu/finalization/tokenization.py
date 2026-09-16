"""Actual-model-tokenizer statistics with an explicit mask-unavailable state."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import csv
import math
import os
from pathlib import Path
import re
from typing import Any

from .contamination import content_text
from .core import DEFAULT_BASE_MODEL_ID, atomic_write_text, sha256_text


class TokenizationUnavailable(RuntimeError):
    """The exact requested tokenizer could not be loaded or used."""


def _length(value: Any) -> int:
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if hasattr(value, "shape"):
        shape = getattr(value, "shape")
        if shape:
            return int(shape[-1]) if len(shape) > 1 else int(shape[0])
    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], (list, tuple)):
            return len(value[0])
        return len(value)
    return 0


def _mask_from(value: Any) -> list[int] | None:
    if not isinstance(value, Mapping):
        return None
    for key in ("assistant_masks", "assistant_tokens_mask", "assistant_token_mask"):
        mask = value.get(key)
        if mask is None:
            continue
        if hasattr(mask, "tolist"):
            mask = mask.tolist()
        if isinstance(mask, list) and mask and isinstance(mask[0], list):
            mask = mask[0]
        if isinstance(mask, (list, tuple)):
            return [int(bool(item)) for item in mask]
    return None


class TokenCounter:
    """Tokenize one canonical record without retaining a dataset in memory."""

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        chat_template = getattr(tokenizer, "chat_template", None)
        self._assistant_mask_supported = not isinstance(chat_template, str) or (
            "{% generation" in chat_template
        )

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        try:
            encoded = self.tokenizer(text, add_special_tokens=True)
        except Exception as exc:
            raise TokenizationUnavailable(f"base tokenizer failed: {type(exc).__name__}") from exc
        return _length(encoded)

    def count_record(self, record: Mapping[str, Any]) -> dict[str, Any]:
        messages = record.get("messages")
        if not isinstance(messages, list) or not messages:
            return {
                "raw_content_tokens": self.count_text(content_text(record)),
                "training_formatted_tokens": None,
                "supervised_assistant_tokens": None,
                "assistant_mask_status": "unavailable_no_messages",
            }
        raw_tokens = self.count_text(content_text(record))
        try:
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
            )
            formatted_tokens = _length(formatted)
            if not formatted_tokens and isinstance(formatted, str):
                formatted_tokens = self.count_text(formatted)
        except Exception:
            # A malformed conversation is a row-level data-quality failure.
            # Preserve its raw-token evidence but keep it out of candidate
            # selection by leaving formatted tokens unavailable.
            return {
                "raw_content_tokens": raw_tokens,
                "training_formatted_tokens": None,
                "supervised_assistant_tokens": None,
                "assistant_mask_status": "unavailable_chat_template",
            }

        supervised: int | None = None
        mask_status = "unavailable_mask_not_returned"
        if not self._assistant_mask_supported:
            return {
                "raw_content_tokens": raw_tokens,
                "training_formatted_tokens": formatted_tokens,
                "supervised_assistant_tokens": None,
                "assistant_mask_status": "unavailable_tokenizer_mask",
            }
        try:
            masked = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_assistant_tokens_mask=True,
            )
            mask = _mask_from(masked)
            if mask is not None and len(mask) == formatted_tokens:
                supervised = sum(mask)
                mask_status = "available"
            elif mask is not None:
                mask_status = "unavailable_mask_length_mismatch"
        except Exception:
            # A tokenizer without a generation-aware chat template is a real
            # limitation, not permission to estimate assistant tokens.
            mask_status = "unavailable_tokenizer_mask"
        return {
            "raw_content_tokens": raw_tokens,
            "training_formatted_tokens": formatted_tokens,
            "supervised_assistant_tokens": supervised,
            "assistant_mask_status": mask_status,
        }

    def count_records(self, records: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Count a batch with the fast tokenizer and fall back row-by-row."""

        if not records:
            return []
        try:
            raw_batch = self.tokenizer(
                [content_text(record) for record in records],
                add_special_tokens=True,
            )
            raw_ids = raw_batch.get("input_ids") if isinstance(raw_batch, Mapping) else None
            if not isinstance(raw_ids, list) or len(raw_ids) != len(records):
                raise ValueError("tokenizer did not return one raw sequence per record")
            raw_lengths = [len(item) for item in raw_ids]

            message_indices = [
                index
                for index, record in enumerate(records)
                if isinstance(record.get("messages"), list) and record.get("messages")
            ]
            results = [
                {
                    "raw_content_tokens": raw_lengths[index],
                    "training_formatted_tokens": None,
                    "supervised_assistant_tokens": None,
                    "assistant_mask_status": "unavailable_no_messages",
                }
                for index in range(len(records))
            ]
            if not message_indices:
                return results

            conversations = [records[index]["messages"] for index in message_indices]
            formatted_batch = self.tokenizer.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=False,
            )
            if not isinstance(formatted_batch, list) or len(formatted_batch) != len(message_indices):
                raise ValueError("chat template did not return one sequence per record")
            formatted_lengths = [len(item) for item in formatted_batch]
            for offset, index in enumerate(message_indices):
                results[index]["training_formatted_tokens"] = formatted_lengths[offset]
                results[index]["assistant_mask_status"] = "unavailable_tokenizer_mask"

            if not self._assistant_mask_supported:
                return results

            masked_batch = self.tokenizer.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_assistant_tokens_mask=True,
            )
            masks = None
            if isinstance(masked_batch, Mapping):
                masks = masked_batch.get("assistant_masks")
                if masks is None:
                    masks = masked_batch.get("assistant_tokens_mask")
                if masks is None:
                    masks = masked_batch.get("assistant_token_mask")
                if hasattr(masks, "tolist"):
                    masks = masks.tolist()
            if not isinstance(masks, list) or len(masks) != len(message_indices):
                return results
            for offset, index in enumerate(message_indices):
                mask = masks[offset]
                if hasattr(mask, "tolist"):
                    mask = mask.tolist()
                if isinstance(mask, (list, tuple)) and len(mask) == formatted_lengths[offset]:
                    results[index]["supervised_assistant_tokens"] = sum(int(bool(item)) for item in mask)
                    results[index]["assistant_mask_status"] = "available"
                else:
                    results[index]["assistant_mask_status"] = "unavailable_mask_length_mismatch"
            return results
        except Exception:
            return [self.count_record(record) for record in records]


def resolve_model_revision(model_id: str = DEFAULT_BASE_MODEL_ID, *, configured_revision: str | None = None) -> str:
    full_sha = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
    if configured_revision and full_sha.fullmatch(configured_revision.strip()):
        return configured_revision.strip()
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(model_id)
        revision = getattr(info, "sha", None)
    except Exception as exc:  # pragma: no cover - network dependent
        raise TokenizationUnavailable("could not resolve a full model revision") from exc
    if not isinstance(revision, str) or not full_sha.fullmatch(revision):
        raise TokenizationUnavailable("model revision was not a full 40-character SHA")
    return revision


def load_qwen_tokenizer(
    *,
    cache_root: str | Path,
    model_root: str | Path,
    model_id: str = DEFAULT_BASE_MODEL_ID,
    configured_revision: str | None = None,
) -> tuple[Any, str]:
    """Load only the requested model tokenizer, with cache roots on E:."""

    cache_path = Path(cache_root)
    model_path = Path(model_root)
    cache_path.mkdir(parents=True, exist_ok=True)
    model_path.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_path)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_path / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_path / "transformers")
    revision = resolve_model_revision(model_id, configured_revision=configured_revision)
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise TokenizationUnavailable("transformers is not installed") from exc
    local_candidates = [
        model_path / model_id.replace("/", "__"),
        model_path / model_id,
    ]
    sources: list[str] = [str(path) for path in local_candidates if path.is_dir()]
    sources.append(model_id)
    last_error: Exception | None = None
    for source in sources:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                source,
                revision=revision,
                cache_dir=str(cache_path / "hub"),
                local_files_only=source != model_id,
            )
            return tokenizer, revision
        except Exception as exc:  # pragma: no cover - environment dependent
            last_error = exc
    raise TokenizationUnavailable(f"could not load {model_id} at revision {revision}") from last_error


def _quantile(values: list[int], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower), 3)


class TokenStats:
    """Streaming aggregate with a deterministic bounded quantile reservoir."""

    QUANTILE_SAMPLE_LIMIT = 100_000

    def __init__(self) -> None:
        self.rows = 0
        self.raw_tokens = 0
        self.formatted_tokens = 0
        self.supervised_tokens = 0
        self.supervised_available = 0
        self._seen_for_quantiles = 0
        self.max_tokens = 0
        self.length_buckets: Counter[str] = Counter()
        self.values: list[int] = []

    def add(self, counts: Mapping[str, Any]) -> None:
        formatted = counts.get("training_formatted_tokens")
        if formatted is None:
            return
        formatted_int = int(formatted)
        self.rows += 1
        self.raw_tokens += int(counts.get("raw_content_tokens") or 0)
        self.formatted_tokens += formatted_int
        self.max_tokens = max(self.max_tokens, formatted_int)
        if formatted_int <= 1_000:
            self.length_buckets["<=1k"] += 1
        elif formatted_int <= 4_000:
            self.length_buckets["1k-4k"] += 1
        elif formatted_int <= 8_000:
            self.length_buckets["4k-8k"] += 1
        elif formatted_int <= 32_000:
            self.length_buckets["8k-32k"] += 1
        else:
            self.length_buckets[">32k"] += 1
        self._seen_for_quantiles += 1
        if len(self.values) < self.QUANTILE_SAMPLE_LIMIT:
            self.values.append(formatted_int)
        else:
            # Deterministic reservoir sampling keeps memory bounded while
            # preserving exact quantiles for the first 100k rows.
            slot = (
                self._seen_for_quantiles * 6_364_136_223_846_793_005
                + 1_442_695_040_888_963_407
            ) % self._seen_for_quantiles
            if slot < self.QUANTILE_SAMPLE_LIMIT:
                self.values[slot] = formatted_int
        supervised = counts.get("supervised_assistant_tokens")
        if supervised is not None:
            self.supervised_available += 1
            self.supervised_tokens += int(supervised)

    def as_dict(self) -> dict[str, Any]:
        values = self.values
        return {
            "rows": self.rows,
            "raw_content_tokens": self.raw_tokens,
            "training_formatted_tokens": self.formatted_tokens,
            "supervised_assistant_tokens": self.supervised_tokens if self.supervised_available else None,
            "supervised_rows": self.supervised_available,
            "quantile_sample_size": len(values),
            "quantile_method": "exact_or_deterministic_reservoir",
            "total_tokens": self.formatted_tokens,
            "median": _quantile(values, 0.50),
            "mean": round(self.formatted_tokens / self.rows, 3) if self.rows else None,
            "p75": _quantile(values, 0.75),
            "p90": _quantile(values, 0.90),
            "p95": _quantile(values, 0.95),
            "p99": _quantile(values, 0.99),
            "max": self.max_tokens if self.rows else None,
            "<=1k": self.length_buckets["<=1k"],
            "1k-4k": self.length_buckets["1k-4k"],
            "4k-8k": self.length_buckets["4k-8k"],
            "8k-32k": self.length_buckets["8k-32k"],
            ">32k": self.length_buckets[">32k"],
        }


def template_family_hash(record: Mapping[str, Any]) -> str:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return sha256_text("")[:16]
    roles = ",".join(str(item.get("role", "")) for item in messages if isinstance(item, Mapping))
    return sha256_text(roles)[:16]


def write_token_statistics(repo_root: str | Path, stats: Mapping[tuple[str, str], TokenStats]) -> tuple[Path, Path]:
    report_dir = Path(repo_root) / "reports" / "pretrain_finalization"
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "token_statistics.csv"
    fields = [
        "dataset",
        "category",
        "rows",
        "raw_content_tokens",
        "training_formatted_tokens",
        "supervised_assistant_tokens",
        "supervised_rows",
        "quantile_sample_size",
        "quantile_method",
        "total_tokens",
        "median",
        "mean",
        "p75",
        "p90",
        "p95",
        "p99",
        "max",
        "<=1k",
        "1k-4k",
        "4k-8k",
        "8k-32k",
        ">32k",
    ]
    rows = []
    for (dataset, category), accumulator in sorted(stats.items()):
        row = {"dataset": dataset, "category": category}
        row.update(accumulator.as_dict())
        rows.append(row)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# ZIPANGU Token Statistics",
        "",
        "Counts use the requested base tokenizer and the dataset's chat template. Missing assistant masks are recorded as unavailable; no supervised-token estimate is substituted.",
        "",
        f"- tokenizer: `{DEFAULT_BASE_MODEL_ID}`",
        "- selection unit: formatted training tokens",
        "",
        "| Dataset | Category | Rows | Total formatted tokens | Median | Mean | P95 | Max | Supervised rows |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['dataset']}` | `{row['category']}` | {row['rows']:,} | {row['total_tokens']:,} | {row['median'] or 'n/a'} | {row['mean'] or 'n/a'} | {row['p95'] or 'n/a'} | {row['max'] or 'n/a'} | {row['supervised_rows']:,} |"
        )
    md_path = report_dir / "TOKEN_STATISTICS.md"
    atomic_write_text(md_path, "\n".join(lines) + "\n")
    return csv_path, md_path
