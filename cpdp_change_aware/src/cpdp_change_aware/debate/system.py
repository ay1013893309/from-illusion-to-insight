"""Multi-agent debate for cross-project change-aware prediction."""

from __future__ import annotations

import json
from statistics import mean
from typing import List, Optional

from ..features.ast_features import summarize_structure
from ..features.context import build_extended_context
from ..features.diffing import compute_diff_stats
from ..llm.client import OpenAICompatibleClient
from ..prompts.templates import (
    build_analyzer_prompt,
    build_judge_prompt,
    build_proposer_prompt,
    build_skeptic_prompt,
)
from ..schemas import ChangePair, PredictionResult, RetrievedExample


class CrossProjectDebateSystem:
    """Run either a heuristic or LLM-based cross-project debate."""

    def __init__(
        self,
        *,
        backend: str = "heuristic",
        model: str = "gpt-5-mini",
        llm_client: Optional[OpenAICompatibleClient] = None,
    ):
        self.backend = backend
        self.model = model
        self.llm_client = llm_client

    def predict(self, change_pair: ChangePair, retrieved_examples: List[RetrievedExample]) -> PredictionResult:
        extended_context = build_extended_context(change_pair)
        if self.backend == "llm":
            return self._predict_with_llm(change_pair, extended_context, retrieved_examples)
        return self._predict_heuristically(change_pair, extended_context, retrieved_examples)

    def _predict_heuristically(
        self,
        change_pair: ChangePair,
        extended_context: str,
        retrieved_examples: List[RetrievedExample],
    ) -> PredictionResult:
        stats = compute_diff_stats(change_pair.old_code, change_pair.new_code)
        old_counts, _ = summarize_structure(change_pair.old_code, change_pair.language)
        new_counts, _ = summarize_structure(change_pair.new_code, change_pair.language)

        defect_examples = [item for item in retrieved_examples if item.new_label == 1]
        benign_examples = [item for item in retrieved_examples if item.new_label == 0]
        defect_rate = mean([item.new_label for item in retrieved_examples if item.new_label is not None]) if retrieved_examples else 0.5

        branch_delta = (
            new_counts.get("if", 0)
            + new_counts.get("for", 0)
            + new_counts.get("while", 0)
            + new_counts.get("try", 0)
            - old_counts.get("if", 0)
            - old_counts.get("for", 0)
            - old_counts.get("while", 0)
            - old_counts.get("try", 0)
        )
        size_delta = stats["added_lines"] + stats["removed_lines"]

        risk_score = 0.0
        risk_score += 0.55 * defect_rate
        risk_score += 0.20 if branch_delta > 0 else 0.0
        risk_score += 0.10 if stats["changed_blocks"] >= 3 else 0.0
        risk_score += 0.10 if size_delta >= 12 else 0.0
        risk_score += 0.05 if new_counts.get("catch", 0) > old_counts.get("catch", 0) else 0.0
        risk_score = max(0.0, min(1.0, risk_score))

        predicted_label = int(risk_score >= 0.5)
        confidence = 50.0 + abs(risk_score - 0.5) * 100.0

        analyzer = (
            "Analyzer: the target change was compared against source-project change pairs with similar "
            f"structural patterns. diff_stats={stats}, branch_delta={branch_delta}, "
            f"retrieved_defect_rate={defect_rate:.2f}."
        )
        proposer = (
            "Proposer: the change resembles bug-prone source changes"
            if defect_examples
            else "Proposer: the change contains risky structural growth without many benign analogues"
        )
        skeptic = (
            "Skeptic: several retrieved examples are benign, so the change may be refactoring or harmless maintenance"
            if benign_examples
            else "Skeptic: there is limited benign evidence, but some edits may still be non-functional"
        )
        judge = (
            f"Judge: predicted_label={predicted_label}, confidence={confidence:.1f}, "
            f"decision_basis=risk_score({risk_score:.2f}) with cross-project few-shot evidence."
        )

        return PredictionResult(
            pair_id=change_pair.pair_id,
            project=change_pair.project,
            file_path=change_pair.file_path,
            predicted_label=predicted_label,
            predicted_score=round(risk_score, 4),
            confidence=round(confidence, 1),
            backend=self.backend,
            analyzer_response=analyzer,
            proposer_response=proposer,
            skeptic_response=skeptic,
            judge_response=judge,
            retrieved_examples=retrieved_examples,
            gold_label=change_pair.new_label,
        )

    def _predict_with_llm(
        self,
        change_pair: ChangePair,
        extended_context: str,
        retrieved_examples: List[RetrievedExample],
    ) -> PredictionResult:
        if self.llm_client is None:
            raise RuntimeError("LLM backend selected but no llm_client was provided.")

        analyzer_system, analyzer_user = build_analyzer_prompt(change_pair, extended_context, retrieved_examples)
        analyzer = self.llm_client.complete_text(
            system_prompt=analyzer_system,
            user_prompt=analyzer_user,
            model=self.model,
        )

        proposer_system, proposer_user = build_proposer_prompt(
            change_pair,
            extended_context,
            analyzer,
            retrieved_examples,
        )
        proposer = self.llm_client.complete_text(
            system_prompt=proposer_system,
            user_prompt=proposer_user,
            model=self.model,
        )

        skeptic_system, skeptic_user = build_skeptic_prompt(
            change_pair,
            extended_context,
            proposer,
            retrieved_examples,
        )
        skeptic = self.llm_client.complete_text(
            system_prompt=skeptic_system,
            user_prompt=skeptic_user,
            model=self.model,
        )

        judge_system, judge_user = build_judge_prompt(
            change_pair,
            extended_context,
            analyzer,
            proposer,
            skeptic,
            retrieved_examples,
        )
        judge = self.llm_client.complete_text(
            system_prompt=judge_system,
            user_prompt=judge_user,
            model=self.model,
        )
        decision = self._parse_judge_decision(judge)

        return PredictionResult(
            pair_id=change_pair.pair_id,
            project=change_pair.project,
            file_path=change_pair.file_path,
            predicted_label=int(decision["label"]),
            predicted_score=float(decision["defect_probability"]),
            confidence=float(decision["confidence"]),
            backend=self.backend,
            analyzer_response=analyzer,
            proposer_response=proposer,
            skeptic_response=skeptic,
            judge_response=decision["reason"],
            retrieved_examples=retrieved_examples,
            gold_label=change_pair.new_label,
        )

    @staticmethod
    def _parse_judge_decision(response: str) -> dict:
        start = response.find("{")
        end = response.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {
                "label": 0,
                "defect_probability": 0.5,
                "confidence": 50,
                "reason": response.strip(),
            }
        try:
            payload = json.loads(response[start : end + 1])
        except json.JSONDecodeError:
            return {
                "label": 0,
                "defect_probability": 0.5,
                "confidence": 50,
                "reason": response.strip(),
            }
        label = int(payload.get("label", 0))
        confidence = float(payload.get("confidence", 50))
        defect_probability = payload.get("defect_probability")
        if defect_probability is None:
            defect_probability = confidence / 100.0 if label == 1 else 1.0 - (confidence / 100.0)
        defect_probability = max(0.0, min(1.0, float(defect_probability)))
        return {
            "label": label,
            "defect_probability": defect_probability,
            "confidence": confidence,
            "reason": str(payload.get("reason", response.strip())),
        }
