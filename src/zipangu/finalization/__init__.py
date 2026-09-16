"""Safety-first local finalization helpers for the ZIPANGU experiment.

The finalization package is deliberately separate from the existing dataset
pipeline.  It may create derived artifacts in the caller-provided external
directories, but it never changes raw datasets, registry status, or approval
state.
"""

from .core import (
    DEFAULT_PATHS,
    FIRST_WAVE_DATASETS,
    MIN_RETENTION,
    QUALITY_THRESHOLDS,
    FinalizationBlocked,
    RuntimeStore,
    config_hash,
    input_fingerprint,
)
from .contamination import (
    ContaminationIndex,
    contamination_status,
    minhash_signature,
    simhash,
)
from .tokenization import TokenCounter, TokenizationUnavailable

__all__ = [
    "ContaminationIndex",
    "DEFAULT_PATHS",
    "FIRST_WAVE_DATASETS",
    "FinalizationBlocked",
    "MIN_RETENTION",
    "QUALITY_THRESHOLDS",
    "RuntimeStore",
    "TokenCounter",
    "TokenizationUnavailable",
    "config_hash",
    "contamination_status",
    "input_fingerprint",
    "minhash_signature",
    "simhash",
]
