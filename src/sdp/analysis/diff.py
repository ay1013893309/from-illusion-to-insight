import difflib
from typing import Dict, List, Tuple


def compute_diff(src1: str, src2: str) -> Dict[str, List[Tuple]]:
    a = src1.splitlines()
    b = src2.splitlines()
    sm = difflib.SequenceMatcher(a=a, b=b)

    removed = []
    added = []
    changed = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag == "delete":
            for idx in range(i1, i2):
                removed.append((idx + 1, a[idx]))
        elif tag == "insert":
            for j in range(j1, j2):
                added.append((j + 1, b[j]))
        elif tag == "replace":
            min_len = min(i2 - i1, j2 - j1)
            for k in range(min_len):
                changed.append((i1 + k + 1, a[i1 + k], j1 + k + 1, b[j1 + k]))
            for idx in range(i1 + min_len, i2):
                removed.append((idx + 1, a[idx]))
            for j in range(j1 + min_len, j2):
                added.append((j + 1, b[j]))
        else:
            raise RuntimeError(f"Unexpected diff tag: {tag}")

    return {"removed": removed, "added": added, "changed": changed}


def format_diffs(diffs: Dict[str, List[Tuple]], max_items: int = None) -> str:
    parts = []
    removed = diffs.get("removed", [])
    added = diffs.get("added", [])
    changed = diffs.get("changed", [])

    def take_items(items):
        if max_items is None or len(items) <= max_items:
            return items, False
        return items[:max_items], True

    r_show, r_trunc = take_items(removed)
    a_show, a_trunc = take_items(added)
    c_show, c_trunc = take_items(changed)

    if removed:
        parts.append("Removed lines:")
        for ln, txt in r_show:
            parts.append(f"  - (SRC1:{ln}) {txt}")
        if r_trunc:
            parts.append(f"  ... and {len(removed) - len(r_show)} more removed lines")
    else:
        parts.append("Removed lines: (none)")

    if added:
        parts.append("Added lines:")
        for ln, txt in a_show:
            parts.append(f"  + (SRC2:{ln}) {txt}")
        if a_trunc:
            parts.append(f"  ... and {len(added) - len(a_show)} more added lines")
    else:
        parts.append("Added lines: (none)")

    if changed:
        parts.append("Changed lines:")
        for old_ln, old_txt, new_ln, new_txt in c_show:
            parts.append(f"  * (SRC1:{old_ln}) {old_txt}")
            parts.append(f"    (SRC2:{new_ln}) {new_txt}")
        if c_trunc:
            parts.append(f"  ... and {len(changed) - len(c_show)} more changed blocks")
    else:
        parts.append("Changed lines: (none)")

    return "\n".join(parts)


def unified_diff(src1: str, src2: str, fromfile: str = "SRC1", tofile: str = "SRC2") -> str:
    a = src1.splitlines(keepends=False)
    b = src2.splitlines(keepends=False)
    ud = difflib.unified_diff(a, b, fromfile=fromfile, tofile=tofile, lineterm="")
    return "\n".join(list(ud))


def compute_diffs(src1: str, src2: str) -> Dict[str, List[Tuple]]:
    """
    Backward-compatible diff helper used by older notebook-derived code.
    """
    diffs = compute_diff(src1, src2)
    diffs["unified"] = unified_diff(src1, src2).splitlines()
    return diffs
