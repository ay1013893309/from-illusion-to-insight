"""Multi-agent debate for static target-project prediction."""

from __future__ import annotations

import json
from typing import List, Sequence

import pandas as pd

from ..config import SETTINGS
from ..llm.client import OpenAICompatibleClient
from ..prompts.static_templates import (
    build_static_analyzer_prompt,
    build_static_judge_prompt,
    build_static_proposer_prompt,
    build_static_skeptic_prompt,
)
from ..retrieval.static_support import StaticStructureRetriever
from ..schemas import ChangePair, StaticFileSample
from ..baselines.direct_transfer import EvaluationSummary, save_baseline_outputs


class StaticTargetDebateSystem:
    """Debate-based predictor for static target files."""

    def __init__(self, *, model: str | None = None, top_k: int = 3):
        if not SETTINGS.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for static debate prediction.")
        self.model = model or SETTINGS.default_model
        self.llm_client = OpenAICompatibleClient(
            api_key=SETTINGS.openai_api_key,
            base_url=SETTINGS.openai_base_url,
        )
        self.retriever = StaticStructureRetriever(top_k=top_k)

    def fit(self, source_pairs: Sequence[ChangePair]) -> "StaticTargetDebateSystem":
        self.retriever.fit(source_pairs)
        return self

    def _complete(self, system_prompt: str, user_prompt: str) -> str:
        return self.llm_client.complete_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=self.model,
            temperature=0.1,
        )

    @staticmethod
    def _parse_judge_response(response: str) -> dict:
        start = response.find("{")
        end = response.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {"label": 0, "defect_probability": 0.5, "confidence": 50.0, "reason": response.strip()}
        try:
            payload = json.loads(response[start : end + 1])
        except json.JSONDecodeError:
            return {"label": 0, "defect_probability": 0.5, "confidence": 50.0, "reason": response.strip()}

        label = int(payload.get("label", 0))
        defect_probability = max(0.0, min(1.0, float(payload.get("defect_probability", 0.5))))
        confidence = max(0.0, min(100.0, float(payload.get("confidence", 50.0))))
        return {
            "label": label,
            "defect_probability": defect_probability,
            "confidence": confidence,
            "reason": str(payload.get("reason", response.strip())),
        }

    def predict_frame(self, target_samples: Sequence[StaticFileSample]) -> pd.DataFrame:
        rows = []
        for sample in target_samples:
            retrieved = self.retriever.retrieve(sample)

            analyzer_system, analyzer_user = build_static_analyzer_prompt(sample, retrieved)
            analyzer = self._complete(analyzer_system, analyzer_user)

            proposer_system, proposer_user = build_static_proposer_prompt(sample, analyzer, retrieved)
            proposer = self._complete(proposer_system, proposer_user)

            skeptic_system, skeptic_user = build_static_skeptic_prompt(sample, analyzer, proposer, retrieved)
            skeptic = self._complete(skeptic_system, skeptic_user)

            judge_system, judge_user = build_static_judge_prompt(
                sample,
                analyzer,
                proposer,
                skeptic,
                retrieved,
            )
            judge_raw = self._complete(judge_system, judge_user)
            decision = self._parse_judge_response(judge_raw)

            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "project": sample.project,
                    "version": sample.version,
                    "file_path": sample.file_path,
                    "gold_label": int(sample.label or 0),
                    "predicted_label": int(decision["label"]),
                    "predicted_score": float(decision["defect_probability"]),
                    "confidence": float(decision["confidence"]),
                    "analyzer_response": analyzer,
                    "proposer_response": proposer,
                    "skeptic_response": skeptic,
                    "judge_response": decision["reason"],
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
        from ..baselines.direct_transfer import DirectTransferBaseline

        self.fit(source_pairs)
        prediction_frame = self.predict_frame(target_samples)
        summary = DirectTransferBaseline.evaluate_frame(
            prediction_frame,
            source_name=source_name,
            target_name=target_name,
        )
        return prediction_frame, summary
