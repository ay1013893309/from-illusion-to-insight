"""Diff helpers."""

from __future__ import annotations

import difflib
from typing import Dict


def unified_diff(old_code: str, new_code: str, fromfile: str = "old", tofile: str = "new") -> str:
    return "\n".join(
        difflib.unified_diff(
            old_code.splitlines(),
            new_code.splitlines(),
            fromfile=fromfile,
            tofile=tofile,
            lineterm="",
        )
    )


def compute_diff_stats(old_code: str, new_code: str) -> Dict[str, int]:
    stats = {"added_lines": 0, "removed_lines": 0, "changed_blocks": 0}
    matcher = difflib.SequenceMatcher(a=old_code.splitlines(), b=new_code.splitlines())
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        stats["changed_blocks"] += 1
        if tag in {"replace", "delete"}:
            stats["removed_lines"] += i2 - i1
        if tag in {"replace", "insert"}:
            stats["added_lines"] += j2 - j1
    return stats

