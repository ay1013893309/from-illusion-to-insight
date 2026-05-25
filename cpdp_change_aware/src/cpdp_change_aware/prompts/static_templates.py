"""Prompt templates for static-target multi-agent debate."""

from __future__ import annotations

from typing import Iterable

from ..features.context import build_static_file_document
from ..schemas import StaticFileSample

STATIC_RISK_RUBRIC = """Static-file risk rubric:
- Strong defect evidence: unsafe mutable state, missing input validation, complex nested control flow, suspicious null handling, silent exception swallowing, complex resource lifecycle, dense low-level indexing or buffer logic.
- Benign evidence: test-only code, DTO/config holder, simple delegator/getter/setter, constants-only utility, straightforward analyzer/token filter wiring with little custom logic.
- Retrieval rule: use source change examples as analogies for semantic risk patterns, not as a vote based on filenames.
- Calibration rule: if the file contains mixed evidence, keep defect_probability close to 0.50."""


def _examples_block(retrieved_examples: Iterable[dict]) -> str:
    parts = []
    for item in retrieved_examples:
        parts.append(
            f"[{item['pair_id']}] project={item['project']} label={item['label']} "
            f"similarity={item['similarity']}\n{item['summary']}"
        )
    return "\n\n".join(parts) if parts else "No retrieved source examples."


def _retrieval_summary(retrieved_examples: list[dict]) -> str:
    if not retrieved_examples:
        return "retrieved_examples=0, defect_examples=0, benign_examples=0"
    defect = sum(1 for item in retrieved_examples if item.get("label") == 1)
    benign = sum(1 for item in retrieved_examples if item.get("label") == 0)
    avg_similarity = sum(float(item["similarity"]) for item in retrieved_examples) / len(retrieved_examples)
    return (
        f"retrieved_examples={len(retrieved_examples)}, "
        f"defect_examples={defect}, benign_examples={benign}, "
        f"avg_similarity={avg_similarity:.3f}"
    )


def build_static_analyzer_prompt(sample: StaticFileSample, retrieved_examples: list[dict]) -> tuple[str, str]:
    system = (
        "You are Analyzer in a static-target CPDP debate. "
        "You only see the current file and retrieved source-project change examples. "
        "Focus on project-agnostic defect patterns and be concise."
    )
    user = (
        f"{STATIC_RISK_RUBRIC}\n\n"
        f"Retrieval summary: {_retrieval_summary(retrieved_examples)}\n\n"
        f"Retrieved source examples:\n{_examples_block(retrieved_examples)}\n\n"
        f"Target static file:\n{build_static_file_document(sample)}\n\n"
        "Return exactly four lines:\n"
        "1. file_role: ...\n"
        "2. risk_signals: ...\n"
        "3. benign_signals: ...\n"
        "4. closest_transfer_pattern: ...\n"
        "Keep each line short."
    )
    return system, user


def build_static_proposer_prompt(
    sample: StaticFileSample,
    analyzer_response: str,
    retrieved_examples: list[dict],
) -> tuple[str, str]:
    system = (
        "You are Proposer in a static-target CPDP debate. "
        "Argue that the current file is defective using structural and semantic evidence."
    )
    user = (
        f"{STATIC_RISK_RUBRIC}\n\n"
        f"Retrieved source examples:\n{_examples_block(retrieved_examples)}\n\n"
        f"Analyzer findings:\n{analyzer_response}\n\n"
        f"Target static file:\n{build_static_file_document(sample)}\n\n"
        "Return exactly three bullet points. "
        "Each bullet must state one defect-oriented argument grounded in the target file."
    )
    return system, user


def build_static_skeptic_prompt(
    sample: StaticFileSample,
    analyzer_response: str,
    proposer_response: str,
    retrieved_examples: list[dict],
) -> tuple[str, str]:
    system = (
        "You are Skeptic in a static-target CPDP debate. "
        "Challenge the defect claim by looking for benign roles, routine scaffolding, or weak transfer matches."
    )
    user = (
        f"{STATIC_RISK_RUBRIC}\n\n"
        f"Retrieved source examples:\n{_examples_block(retrieved_examples)}\n\n"
        f"Analyzer findings:\n{analyzer_response}\n\n"
        f"Proposer arguments:\n{proposer_response}\n\n"
        f"Target static file:\n{build_static_file_document(sample)}\n\n"
        "Return exactly three bullet points. "
        "Each bullet must give one benign or uncertainty-based counterargument."
    )
    return system, user


def build_static_judge_prompt(
    sample: StaticFileSample,
    analyzer_response: str,
    proposer_response: str,
    skeptic_response: str,
    retrieved_examples: list[dict],
) -> tuple[str, str]:
    system = (
        "You are Judge in a static-target CPDP debate. "
        "Decide whether the current target-project file is defective. "
        "You are a calibrated classifier. Return valid JSON only."
    )
    user = (
        f"{STATIC_RISK_RUBRIC}\n\n"
        f"Retrieval summary: {_retrieval_summary(retrieved_examples)}\n\n"
        f"Retrieved source examples:\n{_examples_block(retrieved_examples)}\n\n"
        f"Analyzer:\n{analyzer_response}\n\n"
        f"Proposer:\n{proposer_response}\n\n"
        f"Skeptic:\n{skeptic_response}\n\n"
        f"Target static file:\n{build_static_file_document(sample)}\n\n"
        "Decision policy:\n"
        "- Predict 1 only when file-internal defect evidence clearly outweighs benign evidence.\n"
        "- Predict 0 for clearly routine, test, configuration, or glue files.\n"
        "- Use retrieved examples only as analogies for risk patterns.\n"
        "- If uncertain, keep defect_probability near 0.50.\n\n"
        "Return JSON only:\n"
        "{\"label\": 0 or 1, \"defect_probability\": 0.0-1.0, \"confidence\": 0-100, "
        "\"reason\": \"short reason\", \"top_risk\": \"...\", \"top_benign\": \"...\"}"
    )
    return system, user
