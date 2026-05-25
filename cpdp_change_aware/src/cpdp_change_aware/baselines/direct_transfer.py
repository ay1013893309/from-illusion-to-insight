"""Direct source-to-target CPDP baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ..features.context import build_retrieval_document
from ..schemas import ChangePair


@dataclass
class EvaluationSummary:
    source_name: str
    target_name: str
    rows: int
    positives: int
    negatives: int
    auc: float
    average_precision: float
    precision: float
    recall: float
    f1: float
    mcc: float
    accuracy: float
    tn: int
    fp: int
    fn: int
    tp: int

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(self)])


class DirectTransferBaseline:
    """A simple TF-IDF + logistic regression baseline."""

    def __init__(self, *, max_features: int = 20000, ngram_range: tuple[int, int] = (1, 2)):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        self.model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")

    @staticmethod
    def _documents(change_pairs: Sequence[ChangePair]) -> List[str]:
        return [build_retrieval_document(pair) for pair in change_pairs]

    @staticmethod
    def _labels(change_pairs: Sequence[ChangePair]) -> List[int]:
        labels = [pair.new_label for pair in change_pairs]
        if any(label is None for label in labels):
            raise ValueError("All change pairs must have new_label for baseline evaluation.")
        return [int(label) for label in labels]

    def fit(self, source_pairs: Sequence[ChangePair]) -> "DirectTransferBaseline":
        x_train = self.vectorizer.fit_transform(self._documents(source_pairs))
        y_train = self._labels(source_pairs)
        self.model.fit(x_train, y_train)
        return self

    def predict_frame(self, target_pairs: Sequence[ChangePair]) -> pd.DataFrame:
        x_test = self.vectorizer.transform(self._documents(target_pairs))
        y_score = self.model.predict_proba(x_test)[:, 1]
        y_pred = (y_score >= 0.5).astype(int)
        y_true = self._labels(target_pairs)

        rows = []
        for pair, gold, pred, score in zip(target_pairs, y_true, y_pred, y_score):
            rows.append(
                {
                    "pair_id": pair.pair_id,
                    "project": pair.project,
                    "file_path": pair.file_path,
                    "gold_label": int(gold),
                    "predicted_label": int(pred),
                    "predicted_score": float(score),
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def evaluate_frame(
        prediction_frame: pd.DataFrame,
        *,
        source_name: str,
        target_name: str,
    ) -> EvaluationSummary:
        y_true = prediction_frame["gold_label"].astype(int)
        y_pred = prediction_frame["predicted_label"].astype(int)
        y_score = prediction_frame["predicted_score"].astype(float)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        return EvaluationSummary(
            source_name=source_name,
            target_name=target_name,
            rows=int(len(prediction_frame)),
            positives=int(y_true.sum()),
            negatives=int((1 - y_true).sum()),
            auc=float(roc_auc_score(y_true, y_score)),
            average_precision=float(average_precision_score(y_true, y_score)),
            precision=float(precision_score(y_true, y_pred, zero_division=0)),
            recall=float(recall_score(y_true, y_pred, zero_division=0)),
            f1=float(f1_score(y_true, y_pred, zero_division=0)),
            mcc=float(matthews_corrcoef(y_true, y_pred)),
            accuracy=float(accuracy_score(y_true, y_pred)),
            tn=int(tn),
            fp=int(fp),
            fn=int(fn),
            tp=int(tp),
        )

    def run(
        self,
        *,
        source_pairs: Sequence[ChangePair],
        target_pairs: Sequence[ChangePair],
        source_name: str,
        target_name: str,
    ) -> tuple[pd.DataFrame, EvaluationSummary]:
        self.fit(source_pairs)
        prediction_frame = self.predict_frame(target_pairs)
        summary = self.evaluate_frame(
            prediction_frame,
            source_name=source_name,
            target_name=target_name,
        )
        return prediction_frame, summary


def save_baseline_outputs(
    prediction_frame: pd.DataFrame,
    summary: EvaluationSummary,
    *,
    prediction_path: str | Path,
    summary_path: str | Path,
) -> None:
    prediction_path = Path(prediction_path)
    summary_path = Path(summary_path)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(prediction_path, index=False)
    summary.to_frame().to_csv(summary_path, index=False)
