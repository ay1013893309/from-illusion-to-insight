"""LLM baseline for static target-project prediction without TF-IDF."""

from __future__ import annotations

import json
from typing import List, Sequence

import pandas as pd

from ..config import SETTINGS
from ..features.context import build_static_file_document
from ..llm.client import OpenAICompatibleClient
from ..retrieval.static_support import StaticStructureRetriever
from ..schemas import ChangePair, StaticFileSample
from .direct_transfer import EvaluationSummary


class StaticTargetLLMBaseline:
    """Source change pairs guide static target prediction with an LLM."""

    def __init__(self, *, model: str | None = None, top_k: int = 3):
        if not SETTINGS.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for the LLM static baseline.")
        self.model = model or SETTINGS.default_model
        self.top_k = top_k
        self.llm_client = OpenAICompatibleClient(
            api_key=SETTINGS.openai_api_key,
            base_url=SETTINGS.openai_base_url,
        )
        self.retriever = StaticStructureRetriever(top_k=top_k)

    def fit(self, source_pairs: Sequence[ChangePair]) -> "StaticTargetLLMBaseline":
        self.retriever.fit(source_pairs)
        return self

    def _retrieve_examples(self, sample: StaticFileSample) -> List[dict]:
        return self.retriever.retrieve(sample)

    def _build_prompt(self, sample: StaticFileSample, retrieved: List[dict]) -> tuple[str, str]:
        examples_block = "\n\n".join(
            [
                f"[{item['pair_id']}] project={item['project']} label={item['label']} similarity={item['similarity']}\n{item['summary']}"
                for item in retrieved
            ]
        )
        if not examples_block:
            examples_block = "No retrieved source change examples."

        system_prompt = (
            "You are an expert in cross-project software defect prediction. "
            "You will predict whether a target-project file is defective using only its current code "
            "plus a few retrieved source-project change examples. "
            "Focus on project-agnostic defect patterns, code structure, exception handling, state updates, "
            "validation logic, and risky control flow. Return valid JSON only."
        )

        user_prompt = (
            "Task:\n"
            "Predict whether the target static file is defective.\n\n"
            "Decision guidance:\n"
            "- Defect evidence: risky exception handling, missing validation, complex branching, unsafe state changes, suspicious resource handling.\n"
            "- Benign evidence: simple DTO/config/test files, naming-only patterns, straightforward accessors, trivial glue code.\n"
            "- Use retrieved examples as analogies, not as majority voting.\n"
            "- If uncertain, keep defect_probability near 0.50.\n\n"
            f"Retrieved source change examples:\n{examples_block}\n\n"
            f"Target static file:\n{build_static_file_document(sample)}\n\n"
            "Return JSON only with schema:\n"
            "{\"label\": 0 or 1, \"defect_probability\": 0.0-1.0, \"confidence\": 0-100, \"reason\": \"short reason\"}"
        )
        return system_prompt, user_prompt

    @staticmethod
    def _parse_response(response: str) -> dict:
        start = response.find("{")
        end = response.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {"label": 0, "defect_probability": 0.5, "confidence": 50.0, "reason": response.strip()}
        try:
            payload = json.loads(response[start : end + 1])
        except json.JSONDecodeError:
            return {"label": 0, "defect_probability": 0.5, "confidence": 50.0, "reason": response.strip()}

        label = int(payload.get("label", 0))
        defect_probability = float(payload.get("defect_probability", 0.5))
        confidence = float(payload.get("confidence", 50.0))
        defect_probability = max(0.0, min(1.0, defect_probability))
        confidence = max(0.0, min(100.0, confidence))
        return {
            "label": label,
            "defect_probability": defect_probability,
            "confidence": confidence,
            "reason": str(payload.get("reason", response.strip())),
        }

    def predict_frame(self, target_samples: Sequence[StaticFileSample]) -> pd.DataFrame:
        rows = []
        for sample in target_samples:
            retrieved = self._retrieve_examples(sample)
            system_prompt, user_prompt = self._build_prompt(sample, retrieved)
            response = self.llm_client.complete_text(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=self.model,
                temperature=0.1,
            )
            decision = self._parse_response(response)
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
                    "reason": decision["reason"],
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
