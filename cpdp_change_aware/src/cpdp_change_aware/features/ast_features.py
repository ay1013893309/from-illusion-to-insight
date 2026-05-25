"""Structure-aware summaries for cross-project transfer."""

from __future__ import annotations

import ast
import re
from collections import Counter
from typing import Dict, Iterable, List, Tuple


PYTHON_NODES = (
    ast.ClassDef,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.If,
    ast.For,
    ast.While,
    ast.Try,
    ast.Call,
    ast.Return,
    ast.Assign,
)

JAVA_METHOD_RE = re.compile(
    r"(?:public|protected|private)?\s*(?:static\s+)?[\w<>\[\]]+\s+([A-Za-z_]\w*)\s*\([^;{}]*\)\s*\{"
)
JAVA_CLASS_RE = re.compile(r"\b(class|interface|enum)\s+([A-Za-z_]\w*)")


def _summarize_python(code: str) -> Tuple[Counter, List[str]]:
    counts: Counter = Counter()
    names: List[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return counts, names

    for node in ast.walk(tree):
        for expected in PYTHON_NODES:
            if isinstance(node, expected):
                counts[expected.__name__.lower()] += 1
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
    return counts, sorted(set(names))


def _summarize_java_like(code: str) -> Tuple[Counter, List[str]]:
    counts: Counter = Counter()
    names: List[str] = []
    lines = code.splitlines()
    keywords = {
        "if": r"\bif\s*\(",
        "for": r"\bfor\s*\(",
        "while": r"\bwhile\s*\(",
        "switch": r"\bswitch\s*\(",
        "try": r"\btry\b",
        "catch": r"\bcatch\s*\(",
        "return": r"\breturn\b",
        "throw": r"\bthrow\b",
    }
    for line in lines:
        for label, pattern in keywords.items():
            counts[label] += len(re.findall(pattern, line))
    for match in JAVA_METHOD_RE.finditer(code):
        counts["methoddef"] += 1
        names.append(match.group(1))
    for match in JAVA_CLASS_RE.finditer(code):
        counts[match.group(1)] += 1
        names.append(match.group(2))
    counts["lines"] = len(lines)
    return counts, sorted(set(names))


def summarize_structure(code: str, language: str) -> Tuple[Counter, List[str]]:
    lowered = (language or "").lower()
    if lowered == "python":
        return _summarize_python(code)
    return _summarize_java_like(code)


def _format_counter(counter: Dict[str, int]) -> str:
    items = [f"{key}={value}" for key, value in sorted(counter.items()) if value]
    return ", ".join(items) if items else "none"


def _delta_items(old_counts: Counter, new_counts: Counter) -> List[str]:
    keys = sorted(set(old_counts) | set(new_counts))
    deltas: List[str] = []
    for key in keys:
        delta = new_counts.get(key, 0) - old_counts.get(key, 0)
        if delta:
            sign = "+" if delta > 0 else ""
            deltas.append(f"{key}:{sign}{delta}")
    return deltas


def build_structural_delta_summary(old_code: str, new_code: str, language: str) -> str:
    old_counts, old_names = summarize_structure(old_code, language)
    new_counts, new_names = summarize_structure(new_code, language)

    added_names = sorted(set(new_names) - set(old_names))
    removed_names = sorted(set(old_names) - set(new_names))
    deltas = _delta_items(old_counts, new_counts)

    lines = [
        f"old_structure: {_format_counter(old_counts)}",
        f"new_structure: {_format_counter(new_counts)}",
        f"structural_delta: {', '.join(deltas) if deltas else 'none'}",
        f"added_identifiers: {', '.join(added_names[:12]) if added_names else 'none'}",
        f"removed_identifiers: {', '.join(removed_names[:12]) if removed_names else 'none'}",
    ]
    return "\n".join(lines)


def build_structure_snapshot_summary(code: str, language: str) -> str:
    counts, names = summarize_structure(code, language)
    lines = [
        f"structure: {_format_counter(counts)}",
        f"identifiers: {', '.join(names[:20]) if names else 'none'}",
    ]
    return "\n".join(lines)
