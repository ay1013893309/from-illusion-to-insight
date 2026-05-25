"""Source-change-guided baseline for static target-project prediction."""

from __future__ import annotations

from typing import List, Sequence

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from ..features.context import build_post_change_document, build_static_file_document
from ..schemas import ChangePair, StaticFileSample
from .direct_transfer import EvaluationSummary


class SourceChangeToStaticBaseline:
    """Train on source change pairs and predict a single-version target project."""

    def __init__(self, *, max_features: int = 20000, ngram_range: tuple[int, int] = (1, 2)):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        self.model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")

    @staticmethod
    def _source_documents(change_pairs: Sequence[ChangePair]) -> List[str]:
        return [build_post_change_document(pair) for pair in change_pairs]

    @staticmethod
    def _target_documents(samples: Sequence[StaticFileSample]) -> List[str]:
        return [build_static_file_document(sample) for sample in samples]

    @staticmethod
    def _source_labels(change_pairs: Sequence[ChangePair]) -> List[int]:
        labels = [pair.new_label for pair in change_pairs]
        if any(label is None for label in labels):
            raise ValueError("All source change pairs must have new_label.")
        return [int(label) for label in labels]

    @staticmethod
    def _target_labels(samples: Sequence[StaticFileSample]) -> List[int]:
        labels = [sample.label for sample in samples]
        if any(label is None for label in labels):
            raise ValueError("All target static samples must have label for evaluation.")
        return [int(label) for label in labels]

    def fit(self, source_pairs: Sequence[ChangePair]) -> "SourceChangeToStaticBaseline":
        x_train = self.vectorizer.fit_transform(self._source_documents(source_pairs))
        y_train = self._source_labels(source_pairs)
        self.model.fit(x_train, y_train)
        return self

    def predict_frame(self, target_samples: Sequence[StaticFileSample]) -> pd.DataFrame:
        x_test = self.vectorizer.transform(self._target_documents(target_samples))
        y_score = self.model.predict_proba(x_test)[:, 1]
        y_pred = (y_score >= 0.5).astype(int)
        y_true = self._target_labels(target_samples)

        rows = []
        for sample, gold, pred, score in zip(target_samples, y_true, y_pred, y_score):
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "project": sample.project,
                    "version": sample.version,
                    "file_path": sample.file_path,
                    "gold_label": int(gold),
                    "predicted_label": int(pred),
                    "predicted_score": float(score),
                }
            )
        return pd.DataFrame(rows)

    def run(
        self,
        *,
        source_pairs: Sequence[ChangePair],
        target_samples: Sequence[StaticFileSample],
        source_name: str,
        target_name: str,
    ) -> tuple[pd.DataFrame, EvaluationSummary]:
        from .direct_transfer import DirectTransferBaseline

        self.fit(source_pairs)
        prediction_frame = self.predict_frame(target_samples)
        summary = DirectTransferBaseline.evaluate_frame(
            prediction_frame,
            source_name=source_name,
            target_name=target_name,
        )
        return prediction_frame, summary
