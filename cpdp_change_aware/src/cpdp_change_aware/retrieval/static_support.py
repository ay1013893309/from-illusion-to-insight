"""Helpers for source-change to static-target retrieval without TF-IDF."""

from __future__ import annotations

import math
from collections import Counter
from typing import List, Sequence

from ..features.ast_features import summarize_structure
from ..features.context import build_post_change_document
from ..schemas import ChangePair, StaticFileSample

STRUCTURE_KEYS = [
    "class",
    "interface",
    "enum",
    "methoddef",
    "if",
    "for",
    "while",
    "switch",
    "try",
    "catch",
    "return",
    "throw",
    "lines",
]


def vectorize_counts(counter: Counter) -> List[float]:
    return [float(counter.get(key, 0)) for key in STRUCTURE_KEYS]


def cosine_similarity(left: List[float], right: List[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def identifier_overlap(left: Sequence[str], right: Sequence[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


class StaticStructureRetriever:
    """Retrieve source change examples for a static target sample."""

    def __init__(self, *, top_k: int = 3):
        self.top_k = top_k
        self.source_pairs: List[ChangePair] = []
        self.source_vectors: List[List[float]] = []
        self.source_identifiers: List[List[str]] = []

    def fit(self, source_pairs: Sequence[ChangePair]) -> "StaticStructureRetriever":
        self.source_pairs = list(source_pairs)
        self.source_vectors = []
        self.source_identifiers = []
        for pair in self.source_pairs:
            counts, names = summarize_structure(pair.new_code, pair.language)
            self.source_vectors.append(vectorize_counts(counts))
            self.source_identifiers.append(names)
        return self

    def retrieve(self, sample: StaticFileSample) -> List[dict]:
        counts, names = summarize_structure(sample.code, sample.language)
        target_vector = vectorize_counts(counts)

        scored = []
        for pair, source_vector, source_names in zip(
            self.source_pairs,
            self.source_vectors,
            self.source_identifiers,
        ):
            structure_sim = cosine_similarity(target_vector, source_vector)
            name_sim = identifier_overlap(names, source_names)
            score = 0.85 * structure_sim + 0.15 * name_sim
            scored.append((score, pair))

        scored.sort(key=lambda item: item[0], reverse=True)
        results = []
        for score, pair in scored[: self.top_k]:
            results.append(
                {
                    "pair_id": pair.pair_id,
                    "project": pair.project,
                    "label": pair.new_label,
                    "similarity": round(score, 4),
                    "summary": build_post_change_document(pair),
                    "file_path": pair.file_path,
                }
            )
        return results
