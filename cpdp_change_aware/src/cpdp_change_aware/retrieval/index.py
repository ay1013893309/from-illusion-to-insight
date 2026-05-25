"""Similarity retrieval over source change pairs."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..features.context import build_extended_context, build_retrieval_document
from ..schemas import ChangePair, RetrievedExample


class RetrievalIndex:
    """A lightweight retrieval index for source-project change pairs."""

    def __init__(self, vectorizer: TfidfVectorizer | None = None):
        self.vectorizer = vectorizer or TfidfVectorizer(max_features=12000, ngram_range=(1, 2))
        self.change_pairs: List[ChangePair] = []
        self.documents: List[str] = []
        self.matrix = None

    def fit(self, change_pairs: List[ChangePair]) -> "RetrievalIndex":
        self.change_pairs = list(change_pairs)
        self.documents = [build_retrieval_document(pair) for pair in self.change_pairs]
        self.matrix = self.vectorizer.fit_transform(self.documents)
        return self

    def query(self, change_pair: ChangePair, top_k: int = 5) -> List[RetrievedExample]:
        if self.matrix is None:
            raise RuntimeError("Retrieval index is not fitted.")
        query_text = build_retrieval_document(change_pair)
        query_vector = self.vectorizer.transform([query_text])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        top_indices = np.argsort(scores)[::-1][:top_k]

        examples: List[RetrievedExample] = []
        for idx in top_indices:
            source_pair = self.change_pairs[int(idx)]
            examples.append(
                RetrievedExample(
                    pair_id=source_pair.pair_id,
                    project=source_pair.project,
                    similarity=float(scores[int(idx)]),
                    new_label=source_pair.new_label,
                    summary=build_extended_context(source_pair, max_code_lines=40),
                    metadata={
                        "file_path": source_pair.file_path,
                        "language": source_pair.language,
                    },
                )
            )
        return examples

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: str | Path) -> "RetrievalIndex":
        with Path(path).open("rb") as handle:
            obj = pickle.load(handle)
        if not isinstance(obj, cls):
            raise TypeError(f"Artifact is not a {cls.__name__} instance.")
        return obj

