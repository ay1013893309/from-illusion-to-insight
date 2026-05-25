"""Build extended contexts for retrieval and debate."""

from __future__ import annotations

from ..schemas import ChangePair, RetrievedExample, StaticFileSample
from .ast_features import build_structural_delta_summary, build_structure_snapshot_summary
from .diffing import compute_diff_stats, unified_diff


def _trim_code_block(code: str, max_lines: int) -> str:
    lines = code.splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[:max_lines])


def build_extended_context(change_pair: ChangePair, max_code_lines: int = 120) -> str:
    diff_text = unified_diff(
        change_pair.old_code,
        change_pair.new_code,
        fromfile=f"{change_pair.project}:old",
        tofile=f"{change_pair.project}:new",
    )
    stats = compute_diff_stats(change_pair.old_code, change_pair.new_code)
    structural = build_structural_delta_summary(
        change_pair.old_code,
        change_pair.new_code,
        change_pair.language,
    )
    old_excerpt = _trim_code_block(change_pair.old_code, max_code_lines)
    new_excerpt = _trim_code_block(change_pair.new_code, max_code_lines)

    return "\n\n".join(
        [
            f"project={change_pair.project}",
            f"file_path={change_pair.file_path}",
            f"language={change_pair.language}",
            f"diff_stats={stats}",
            structural,
            "unified_diff:\n" + diff_text,
            "old_code_excerpt:\n" + old_excerpt,
            "new_code_excerpt:\n" + new_excerpt,
        ]
    )


def build_retrieval_document(change_pair: ChangePair) -> str:
    stats = compute_diff_stats(change_pair.old_code, change_pair.new_code)
    structural = build_structural_delta_summary(
        change_pair.old_code,
        change_pair.new_code,
        change_pair.language,
    )
    diff_text = unified_diff(change_pair.old_code, change_pair.new_code)
    return "\n".join(
        [
            f"project={change_pair.project}",
            f"file={change_pair.file_path}",
            f"language={change_pair.language}",
            f"stats={stats}",
            structural,
            diff_text,
        ]
    )


def build_post_change_document(change_pair: ChangePair) -> str:
    stats = compute_diff_stats(change_pair.old_code, change_pair.new_code)
    structural = build_structure_snapshot_summary(change_pair.new_code, change_pair.language)
    return "\n".join(
        [
            f"project={change_pair.project}",
            f"file={change_pair.file_path}",
            f"language={change_pair.language}",
            f"post_change_stats={stats}",
            structural,
            change_pair.new_code,
        ]
    )


def build_static_file_document(sample: StaticFileSample) -> str:
    structural = build_structure_snapshot_summary(sample.code, sample.language)
    return "\n".join(
        [
            f"project={sample.project}",
            f"file={sample.file_path}",
            f"language={sample.language}",
            structural,
            sample.code,
        ]
    )


def format_retrieved_example(example: RetrievedExample) -> str:
    return (
        f"[{example.pair_id}] project={example.project} "
        f"label={example.new_label} similarity={example.similarity:.3f}\n"
        f"{example.summary}"
    )
