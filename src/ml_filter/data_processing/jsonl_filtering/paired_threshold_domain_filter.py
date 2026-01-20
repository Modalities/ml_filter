"""Deprecated: use PairedThresholdFilter instead.

This module remains for backward compatibility. The unified reader in
`paired_threshold_filter.py` now handles both score and domain filtering.
"""

from ml_filter.data_processing.jsonl_filtering.paired_threshold_filter import (  # noqa: F401
    PairedThresholdFilter as PairedThresholdAndDomainFilter,
)

__all__ = ["PairedThresholdAndDomainFilter"]
