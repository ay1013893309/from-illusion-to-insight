"""Training and prediction workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from .config import SETTINGS
from .debate.system import CrossProjectDebateSystem
from .llm.client import OpenAICompatibleClient
from .retrieval.index import RetrievalIndex
from .schemas import ChangePair, PredictionResult


class CrossProjectChangePredictor:
    """Train a source-project retrieval model and predict target-project changes."""

    def __init__(
        self,
        *,
        backend: Optional[str] = None,
        model: Optional[str] = None,
        top_k: int = 5,
    ):
        self.backend = backend or SETTINGS.default_backend
        self.model = model or SETTINGS.default_model
        self.top_k = top_k
        self.index = RetrievalIndex()

        llm_client = None
        if self.backend == "llm":
            if not SETTINGS.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY is required when backend=llm.")
            llm_client = OpenAICompatibleClient(
                api_key=SETTINGS.openai_api_key,
                base_url=SETTINGS.openai_base_url,
            )

        self.debate = CrossProjectDebateSystem(
            backend=self.backend,
            model=self.model,
            llm_client=llm_client,
        )

    def fit(self, source_pairs: List[ChangePair]) -> "CrossProjectChangePredictor":
        labeled = [pair for pair in source_pairs if pair.new_label is not None]
        if not labeled:
            raise ValueError("At least one labeled source change pair is required.")
        self.index.fit(labeled)
        return self

    def save(self, artifact_path: str | Path) -> None:
        self.index.save(artifact_path)

    def load(self, artifact_path: str | Path) -> "CrossProjectChangePredictor":
        self.index = RetrievalIndex.load(artifact_path)
        return self

    def predict_one(self, target_pair: ChangePair) -> PredictionResult:
        retrieved = self.index.query(target_pair, top_k=self.top_k)
        return self.debate.predict(target_pair, retrieved)

    def predict_many(self, target_pairs: Iterable[ChangePair]) -> List[PredictionResult]:
        return [self.predict_one(pair) for pair in target_pairs]

    @staticmethod
    def to_frame(results: Iterable[PredictionResult]) -> pd.DataFrame:
        rows = []
        for result in results:
            row = result.to_dict()
            row["retrieved_pair_ids"] = "|".join(item["pair_id"] for item in row["retrieved_examples"])
            row["retrieved_labels"] = "|".join(str(item["new_label"]) for item in row["retrieved_examples"])
            row["retrieved_examples"] = str(row["retrieved_examples"])
            rows.append(row)
        return pd.DataFrame(rows)

