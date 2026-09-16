"""Scalable, deterministic contamination fingerprints.

Only hashes, signatures, and identifiers are retained by the index.  Source
text is consumed while building the index and is never written to a tracked
artifact.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from .core import atomic_write_json, sha256_text


MINHASH_PERMUTATIONS = 128
NGRAM_SIZE = 13
LONG_SUBSTRING_SIZE = 80
NEAR_MIN_NORMALIZED_LENGTH = 40
MINHASH_REVIEW_THRESHOLD = 0.35
MINHASH_MIN_LENGTH_RATIO = 0.50
SIMHASH_REVIEW_THRESHOLD = 0.90
SIMHASH_MIN_LENGTH_RATIO = 0.80
QUARANTINE_NEAR_THRESHOLD = 0.80
_UINT64_MAX = (1 << 64) - 1
_SPACE_RE = re.compile(r"\s+", re.UNICODE)
DEFAULT_BENCHMARK_SIGNALS = (
    "mt-bench",
    "mt bench",
    "answercarefully",
    "answer carefully",
    "llm-jp-instructions",
    "japanese mt-bench",
)


class EmptyFingerprintError(ValueError):
    """Raised when an empty normalized projection reaches a fingerprint index."""


def normalize_text(value: Any) -> str:
    """Normalize text for comparison without changing the source row."""

    if not isinstance(value, str):
        value = str(value) if value is not None else ""
    value = unicodedata.normalize("NFKC", value).casefold()
    return _SPACE_RE.sub(" ", value).strip()


def _ngrams(text: str, size: int = NGRAM_SIZE) -> set[str]:
    if len(text) < size:
        return {text} if text else set()
    return {text[index : index + size] for index in range(len(text) - size + 1)}


def _bounded_ngrams(
    text: str,
    size: int,
    *,
    max_grams: int = 64,
    _normalized: bool = False,
) -> set[str]:
    normalized = text if _normalized else normalize_text(text)
    if len(normalized) < size:
        return {normalized} if normalized else set()
    count = len(normalized) - size + 1
    stride = max(1, (count + max_grams - 1) // max_grams)
    return {normalized[index : index + size] for index in range(0, count, stride)}

def _hash64(value: str, seed: int = 0) -> int:
    payload = seed.to_bytes(8, "little", signed=False) + value.encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def minhash_signature(
    text: str,
    *,
    num_perm: int = MINHASH_PERMUTATIONS,
    _normalized: bool = False,
) -> tuple[int, ...]:
    """Return a fixed, reproducible MinHash signature."""

    grams = _bounded_ngrams(
        text,
        NGRAM_SIZE,
        max_grams=64,
        _normalized=_normalized,
    )
    if not grams:
        return tuple(_UINT64_MAX for _ in range(num_perm))
    return tuple(min(_hash64(gram, seed=index + 1) for gram in grams) for index in range(num_perm))


def minhash_similarity(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a == b for a, b in zip(left, right)) / len(left)


def simhash(text: str, *, bits: int = 64, _normalized: bool = False) -> int:
    """Compute a deterministic SimHash over normalized character n-grams."""

    grams = _bounded_ngrams(
        text,
        3,
        max_grams=256,
        _normalized=_normalized,
    )
    if not grams:
        return 0
    votes = [0] * bits
    for gram in grams:
        value = _hash64(gram)
        for bit in range(bits):
            votes[bit] += 1 if (value >> bit) & 1 else -1
    result = 0
    for bit, vote in enumerate(votes):
        if vote >= 0:
            result |= 1 << bit
    return result


def simhash_similarity(left: int, right: int, *, bits: int = 64) -> float:
    return 1.0 - ((int(left) ^ int(right)).bit_count() / bits)


def _long_windows(text: str, *, _normalized: bool = False) -> Iterator[str]:
    """Yield exhaustive long windows without retaining a text-sized set."""

    normalized = text if _normalized else normalize_text(text)
    if len(normalized) < LONG_SUBSTRING_SIZE:
        return
    step = max(1, LONG_SUBSTRING_SIZE // 2)
    positions = range(0, len(normalized) - LONG_SUBSTRING_SIZE + 1, step)
    for position in positions:
        yield normalized[position : position + LONG_SUBSTRING_SIZE]


def _messages_text(value: Any) -> str:
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if not isinstance(item, Mapping):
                continue
            content = item.get("content", item.get("text", ""))
            if isinstance(content, list):
                content = " ".join(
                    str(part.get("text", "")) if isinstance(part, Mapping) else str(part)
                    for part in content
                )
            if content:
                parts.append(str(content))
        return "\n".join(parts)
    return ""


def prompt_text(record: Mapping[str, Any]) -> str:
    messages = record.get("messages")
    if isinstance(messages, list):
        return "\n".join(
            str(item.get("content", ""))
            for item in messages
            if isinstance(item, Mapping) and str(item.get("role", "")).casefold() in {"system", "user"}
        )
    for key in ("prompt", "instruction", "question", "problem", "input", "user"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def content_text(record: Mapping[str, Any]) -> str:
    messages = record.get("messages")
    if messages is not None:
        value = _messages_text(messages)
        if value:
            return value
    parts: list[str] = []
    for key in (
        "system",
        "prompt",
        "instruction",
        "question",
        "problem",
        "input",
        "user",
        "reasoning",
        "assistant_final",
        "answer",
        "response",
        "output",
    ):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value)
    return "\n".join(parts)


def benchmark_signal(
    text: str,
    signals: Iterable[str] = DEFAULT_BENCHMARK_SIGNALS,
    *,
    _normalized: bool = False,
) -> bool:
    normalized = text if _normalized else normalize_text(text)
    return any(normalize_text(signal) in normalized for signal in signals)


@dataclass(frozen=True)
class Fingerprint:
    identifier: str
    content_hash: str
    prompt_hash: str
    signature: tuple[int, ...]
    simhash: int
    normalized_length: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.identifier,
            "content_hash": self.content_hash,
            "prompt_hash": self.prompt_hash,
            "minhash": list(self.signature),
            "simhash": f"{self.simhash:016x}",
            "normalized_length": self.normalized_length,
        }


class ContaminationIndex:
    """Inverted fingerprint index; candidate lookup is sub-quadratic."""

    def __init__(self, *, bands: int = 32, rows_per_band: int = 4):
        if bands * rows_per_band != MINHASH_PERMUTATIONS:
            raise ValueError("bands * rows_per_band must equal 128")
        self.bands = bands
        self.rows_per_band = rows_per_band
        self.fingerprints: dict[str, Fingerprint] = {}
        self.content_hashes: dict[str, set[str]] = defaultdict(set)
        self.prompt_hashes: dict[str, set[str]] = defaultdict(set)
        self.long_substrings: dict[str, set[str]] = defaultdict(set)
        self.ngram_index: dict[str, set[str]] = defaultdict(set)
        self.lsh_index: dict[tuple[int, tuple[int, ...]], set[str]] = defaultdict(set)

    @staticmethod
    def _band_keys(signature: tuple[int, ...], bands: int, rows_per_band: int) -> Iterable[tuple[int, tuple[int, ...]]]:
        for band in range(bands):
            start = band * rows_per_band
            yield band, signature[start : start + rows_per_band]

    def add(self, identifier: str, content: str, *, prompt: str = "", metadata: Mapping[str, Any] | None = None) -> Fingerprint:
        del metadata  # metadata signals are checked on the candidate side.
        normalized = normalize_text(content)
        if not normalized:
            raise EmptyFingerprintError(
                f"empty normalized content is forbidden for fingerprint: {identifier}"
            )
        normalized_prompt = normalize_text(prompt)
        fingerprint = Fingerprint(
            identifier=str(identifier),
            content_hash=sha256_text(normalized),
            prompt_hash=sha256_text(normalized_prompt) if normalized_prompt else "",
            signature=minhash_signature(normalized, _normalized=True),
            simhash=simhash(normalized, _normalized=True),
            normalized_length=len(normalized),
        )
        self.fingerprints[fingerprint.identifier] = fingerprint
        self.content_hashes[fingerprint.content_hash].add(fingerprint.identifier)
        if fingerprint.prompt_hash:
            self.prompt_hashes[fingerprint.prompt_hash].add(fingerprint.identifier)
        for window in _long_windows(normalized, _normalized=True):
            self.long_substrings[sha256_text(window)].add(fingerprint.identifier)
        for gram in _bounded_ngrams(
            normalized,
            NGRAM_SIZE,
            max_grams=512,
            _normalized=True,
        ):
            self.ngram_index[sha256_text(gram)].add(fingerprint.identifier)
        for key in self._band_keys(fingerprint.signature, self.bands, self.rows_per_band):
            self.lsh_index[key].add(fingerprint.identifier)
        return fingerprint

    def add_record(self, identifier: str, record: Mapping[str, Any]) -> Fingerprint:
        return self.add(identifier, content_text(record), prompt=prompt_text(record))

    def lookup(self, content: str, *, prompt: str = "", metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        normalized = normalize_text(content)
        if not normalized:
            raise EmptyFingerprintError("empty normalized candidate content is forbidden")
        candidate_hash = sha256_text(normalized)
        candidate_prompt = normalize_text(prompt)
        prompt_hash = sha256_text(candidate_prompt) if candidate_prompt else ""
        signature = minhash_signature(normalized, _normalized=True)
        candidate_simhash = simhash(normalized, _normalized=True)

        exact_ids = set(self.content_hashes.get(candidate_hash, set()))
        exact_prompt_ids = set(self.prompt_hashes.get(prompt_hash, set())) if prompt_hash else set()
        long_ids: set[str] = set()
        for window in _long_windows(normalized, _normalized=True):
            long_ids.update(self.long_substrings.get(sha256_text(window), set()))

        ngram_ids: set[str] = set()
        for gram in _bounded_ngrams(
            normalized,
            NGRAM_SIZE,
            max_grams=512,
            _normalized=True,
        ):
            ngram_ids.update(self.ngram_index.get(sha256_text(gram), set()))
        lsh_ids: set[str] = set()
        for key in self._band_keys(signature, self.bands, self.rows_per_band):
            lsh_ids.update(self.lsh_index.get(key, set()))

        near_candidates = (ngram_ids | lsh_ids) - exact_ids - long_ids
        # A bucket can be large for boilerplate.  The LSH/inverted index is
        # still the candidate generator; the cap prevents a pathological
        # source from turning the verification step into a quadratic scan.
        near_matches: list[tuple[str, float, tuple[str, ...]]] = []
        for identifier in sorted(near_candidates)[:2048]:
            fingerprint = self.fingerprints.get(identifier)
            if fingerprint is None:
                continue
            shorter = min(len(normalized), fingerprint.normalized_length)
            longer = max(len(normalized), fingerprint.normalized_length)
            length_ratio = shorter / longer if longer else 0.0
            minhash_score = minhash_similarity(signature, fingerprint.signature)
            simhash_score = simhash_similarity(candidate_simhash, fingerprint.simhash)
            evidence: list[str] = []
            if (
                shorter >= NEAR_MIN_NORMALIZED_LENGTH
                and length_ratio >= MINHASH_MIN_LENGTH_RATIO
                and minhash_score >= MINHASH_REVIEW_THRESHOLD
            ):
                evidence.append("minhash_similarity_ge_0_35")
            if (
                shorter >= NEAR_MIN_NORMALIZED_LENGTH
                and length_ratio >= SIMHASH_MIN_LENGTH_RATIO
                and simhash_score >= SIMHASH_REVIEW_THRESHOLD
            ):
                evidence.append("simhash_similarity_ge_0_90_length_ratio_ge_0_80")
            if not evidence:
                continue
            similarity = max(minhash_score, simhash_score)
            near_matches.append(
                (identifier, round(similarity, 6), tuple(evidence))
            )
        near_matches.sort(key=lambda item: (-item[1], item[0]))

        reasons: list[str] = []
        status = "clean"
        matched: list[dict[str, Any]] = []
        if exact_ids:
            status = "quarantine_exact"
            reasons.append("full_content_sha256")
            matched.extend({"id": identifier, "similarity": 1.0} for identifier in sorted(exact_ids))
        elif exact_prompt_ids:
            status = "quarantine_exact"
            reasons.append("prompt_sha256")
            matched.extend({"id": identifier, "similarity": 1.0} for identifier in sorted(exact_prompt_ids))
        elif long_ids:
            status = "quarantine_exact"
            reasons.append("long_substring_ge_80")
            matched.extend({"id": identifier, "similarity": 1.0} for identifier in sorted(long_ids))
        elif near_matches:
            best = near_matches[0][1]
            status = (
                "quarantine_near"
                if best >= QUARANTINE_NEAR_THRESHOLD
                else "requires_manual_review"
            )
            reasons.extend(
                sorted(
                    {
                        reason
                        for _, _, match_reasons in near_matches[:16]
                        for reason in match_reasons
                    }
                )
            )
            matched.extend(
                {"id": identifier, "similarity": similarity}
                for identifier, similarity, _ in near_matches[:16]
            )
        if benchmark_signal(
            normalized,
            (metadata or {}).get("benchmark_signals", DEFAULT_BENCHMARK_SIGNALS),
            _normalized=True,
        ):
            if status == "clean":
                status = "requires_manual_review"
            reasons.append("benchmark_or_source_metadata_signal")
        return {
            "status": status,
            "similarity": matched[0]["similarity"] if matched else 0.0,
            "matches": matched,
            "reasons": reasons,
            "content_hash": candidate_hash,
            "prompt_hash": prompt_hash,
            "simhash": f"{candidate_simhash:016x}",
            "minhash": list(signature),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "algorithm": "exact+substring+13gram+128perm_minhash_lsh+simhash",
            "bands": self.bands,
            "rows_per_band": self.rows_per_band,
            "near_thresholds": {
                "minimum_normalized_length": NEAR_MIN_NORMALIZED_LENGTH,
                "minhash_review": MINHASH_REVIEW_THRESHOLD,
                "minhash_min_length_ratio": MINHASH_MIN_LENGTH_RATIO,
                "simhash_review": SIMHASH_REVIEW_THRESHOLD,
                "simhash_min_length_ratio": SIMHASH_MIN_LENGTH_RATIO,
                "quarantine_near": QUARANTINE_NEAR_THRESHOLD,
            },
            "fingerprints": [item.as_dict() for item in sorted(self.fingerprints.values(), key=lambda item: item.identifier)],
            "content_hashes": {key: sorted(value) for key, value in sorted(self.content_hashes.items())},
            "prompt_hashes": {key: sorted(value) for key, value in sorted(self.prompt_hashes.items())},
            "long_substrings": {key: sorted(value) for key, value in sorted(self.long_substrings.items())},
            "ngram_index": {key: sorted(value) for key, value in sorted(self.ngram_index.items())},
            "lsh_index": {f"{band}:{','.join(str(value) for value in signature)}": sorted(ids) for (band, signature), ids in sorted(self.lsh_index.items(), key=lambda item: str(item[0]))},
        }

    def write(self, path: str | Path) -> Path:
        return atomic_write_json(path, self.as_dict())


def contamination_status(
    candidate: Mapping[str, Any],
    index: ContaminationIndex | None,
    *,
    missing_required_eval: bool,
) -> dict[str, Any]:
    """Scan one canonical row and apply the fail-closed missing-eval gate."""

    candidate_content = content_text(candidate)
    normalized_content = normalize_text(candidate_content)
    if not normalized_content:
        result = {
            "status": "excluded_empty_projection",
            "similarity": 0.0,
            "matches": [],
            "reasons": ["empty_candidate_projection"],
            "content_hash": "",
            "prompt_hash": "",
        }
    elif index is None:
        result = {
            "status": "not_checked_missing_eval_source",
            "similarity": 0.0,
            "matches": [],
            "reasons": ["evaluation_index_unavailable"],
            "content_hash": sha256_text(normalize_text(content_text(candidate))),
            "prompt_hash": (
                sha256_text(normalize_text(prompt_text(candidate)))
                if normalize_text(prompt_text(candidate))
                else ""
            ),
        }
        if missing_required_eval:
            result["reasons"].append("required_eval_source_missing")
    else:
        result = index.lookup(candidate_content, prompt=prompt_text(candidate), metadata=candidate)
    if missing_required_eval and result.get("status") == "clean":
        result["status"] = "not_checked_missing_eval_source"
        result.setdefault("reasons", []).append("required_eval_source_missing")
    return result


def write_fingerprint(path: str | Path, fingerprint: Mapping[str, Any]) -> None:
    """Write a single metadata-only row fingerprint on the external drive."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(fingerprint), ensure_ascii=False, sort_keys=True) + "\n")
