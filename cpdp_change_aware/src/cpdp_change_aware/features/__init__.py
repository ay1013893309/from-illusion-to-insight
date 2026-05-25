"""Feature extraction helpers."""

from .ast_features import build_structural_delta_summary
from .context import build_extended_context, build_retrieval_document
from .diffing import compute_diff_stats, unified_diff

__all__ = [
    "build_extended_context",
    "build_retrieval_document",
    "build_structural_delta_summary",
    "compute_diff_stats",
    "unified_diff",
]

