"""Prompt templates for cross-project debate."""

from __future__ import annotations

from typing import Iterable

from ..features.context import format_retrieved_example
from ..schemas import ChangePair, RetrievedExample


RISK_RUBRIC = """Risk rubric:
- Strong defect evidence: changed control flow, changed state update logic, weakened validation, silent exception swallowing, changed resource handling, new null/edge-case path, inconsistent API setup order.
- Benign evidence: pure rename, formatting, comments, logging-only edits, test-only edits, behavior-preserving extraction/move, dependency injection wiring with no logic change.
- Cross-project transfer rule: trust retrieved examples only when the semantic pattern matches, not just surface tokens or filenames.
- If evidence is mixed, lower confidence and keep defect_probability near 0.50."""


def _few_shot_block(retrieved_examples: Iterable[RetrievedExample]) -> str:
    items = [format_retrieved_example(item) for item in retrieved_examples]
    return "\n\n".join(items) if items else "No retrieved examples."


def _retrieval_summary(retrieved_examples: list[RetrievedExample]) -> str:
    if not retrieved_examples:
        return "retrieved_examples=0, defect_examples=0, benign_examples=0"
    defect_count = sum(1 for item in retrieved_examples if item.new_label == 1)
    benign_count = sum(1 for item in retrieved_examples if item.new_label == 0)
    avg_similarity = sum(item.similarity for item in retrieved_examples) / len(retrieved_examples)
    return (
        f"retrieved_examples={len(retrieved_examples)}, "
        f"defect_examples={defect_count}, benign_examples={benign_count}, "
        f"avg_similarity={avg_similarity:.3f}"
    )


def build_analyzer_prompt(change_pair: ChangePair, extended_context: str, retrieved_examples: list[RetrievedExample]) -> tuple[str, str]:
    system = (
        "You are Analyzer, a cross-project defect analysis expert. "
        "Focus on project-agnostic bug patterns, structural change signals, and semantic risk. "
        "Be concise and evidence-driven."
    )
    user = (
        "Analyze the following target-project code change. "
        "Do not rely on project-specific names. Focus on generic bug-inducing patterns.\n\n"
        f"{RISK_RUBRIC}\n\n"
        f"Retrieval summary: {_retrieval_summary(retrieved_examples)}\n\n"
        f"Retrieved source-project examples:\n{_few_shot_block(retrieved_examples)}\n\n"
        f"Target change context:\n{extended_context}\n\n"
        "Return exactly these four lines:\n"
        "1. semantic_change: ...\n"
        "2. risk_signals: ...\n"
        "3. benign_signals: ...\n"
        "4. transfer_match: ...\n"
        "Keep each line short."
    )
    return system, user


def build_proposer_prompt(
    change_pair: ChangePair,
    extended_context: str,
    analyzer_response: str,
    retrieved_examples: list[RetrievedExample],
) -> tuple[str, str]:
    system = (
        "You are Proposer. Your role is to argue that the change likely introduces or preserves a defect. "
        "Use retrieved source-project evidence and the analyzer's findings. "
        "Argue only from semantic evidence, not from filenames."
    )
    user = (
        f"{RISK_RUBRIC}\n\n"
        f"Retrieval summary: {_retrieval_summary(retrieved_examples)}\n\n"
        f"Retrieved examples:\n{_few_shot_block(retrieved_examples)}\n\n"
        f"Analyzer findings:\n{analyzer_response}\n\n"
        f"Target change:\n{extended_context}\n\n"
        "Return exactly three bullet points.\n"
        "Each bullet must cite one concrete defect-oriented reason tied to the target change."
    )
    return system, user


def build_skeptic_prompt(
    change_pair: ChangePair,
    extended_context: str,
    proposer_response: str,
    retrieved_examples: list[RetrievedExample],
) -> tuple[str, str]:
    system = (
        "You are Skeptic. Your role is to challenge the bug hypothesis. "
        "Consider refactoring, harmless edits, formatting, dead-code removal, and behavior-preserving changes. "
        "Push back only when there is credible benign evidence."
    )
    user = (
        f"{RISK_RUBRIC}\n\n"
        f"Retrieval summary: {_retrieval_summary(retrieved_examples)}\n\n"
        f"Retrieved examples:\n{_few_shot_block(retrieved_examples)}\n\n"
        f"Proposer argument:\n{proposer_response}\n\n"
        f"Target change:\n{extended_context}\n\n"
        "Return exactly three bullet points.\n"
        "Each bullet must give one benign or uncertainty-based counterargument."
    )
    return system, user


def build_judge_prompt(
    change_pair: ChangePair,
    extended_context: str,
    analyzer_response: str,
    proposer_response: str,
    skeptic_response: str,
    retrieved_examples: list[RetrievedExample],
) -> tuple[str, str]:
    system = (
        "You are Judge. Decide whether the changed file in the target project is defective in the new version. "
        "Base your decision on cross-project evidence and the debate. "
        "You are a calibrated classifier, not a storyteller. "
        "Return valid JSON only."
    )
    user = (
        f"{RISK_RUBRIC}\n\n"
        f"Retrieval summary: {_retrieval_summary(retrieved_examples)}\n\n"
        f"Retrieved examples:\n{_few_shot_block(retrieved_examples)}\n\n"
        f"Analyzer:\n{analyzer_response}\n\n"
        f"Proposer:\n{proposer_response}\n\n"
        f"Skeptic:\n{skeptic_response}\n\n"
        f"Target change:\n{extended_context}\n\n"
        "Decision policy:\n"
        "- Predict 1 only when defect evidence outweighs benign evidence.\n"
        "- Predict 0 for clearly benign edits such as rename/logging/test-only/formatting changes.\n"
        "- Use retrieved examples as supporting evidence, not as a majority vote.\n"
        "- If uncertain, keep defect_probability near 0.50.\n\n"
        "Return JSON only with this schema:\n"
        "{\"label\": 0 or 1, \"defect_probability\": 0.0-1.0, \"confidence\": 0-100, "
        "\"reason\": \"short reason\", \"top_risk\": \"...\", \"top_benign\": \"...\"}"
    )
    return system, user
