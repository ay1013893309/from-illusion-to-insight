"""
Analysis helpers for change-aware defect prediction.
"""

from .diff import compute_diff, compute_diffs, format_diffs, unified_diff
from .java_parse import (
    find_java_classes,
    find_java_methods,
    extract_relevant_context,
)
from .method_of_difference import (
    defect_examples,
    TIPS,
    METHOD_INSTRUCTIONS,
    METHODS_WITH_DIFFS,
    METHODS_WITH_UNIFIED_DIFF,
    extract_local_context,
    generate_prompt,
    generate_prompt_v2,
    partition_dataset_versions,
)
from .file_matching import (
    exact_path_matches,
    remove_exact_matches,
    match_ast_files,
    match_files_full,
)

__all__ = [
    "compute_diff",
    "compute_diffs",
    "format_diffs",
    "unified_diff",
    "find_java_classes",
    "find_java_methods",
    "extract_relevant_context",
    "defect_examples",
    "TIPS",
    "extract_local_context",
    "generate_prompt",
    "generate_prompt_v2",
    "partition_dataset_versions",
    "exact_path_matches",
    "remove_exact_matches",
    "match_ast_files",
    "match_files_full",
    "match_files_and_create_hunks",
]


def match_files_and_create_hunks(*args, **kwargs):
    """
    Compatibility wrapper kept for notebook-derived imports.
    """
    from ..data.loader import match_files_and_create_hunks as _match_files_and_create_hunks

    return _match_files_and_create_hunks(*args, **kwargs)
