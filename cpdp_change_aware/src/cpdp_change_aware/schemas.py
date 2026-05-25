"""Shared dataclasses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ChangePair:
    pair_id: str
    project: str
    file_path: str
    old_code: str
    new_code: str
    language: str = "java"
    old_version: Optional[str] = None
    new_version: Optional[str] = None
    old_label: Optional[int] = None
    new_label: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StaticFileSample:
    sample_id: str
    project: str
    file_path: str
    code: str
    language: str = "java"
    version: Optional[str] = None
    label: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievedExample:
    pair_id: str
    project: str
    similarity: float
    new_label: Optional[int]
    summary: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionResult:
    pair_id: str
    project: str
    file_path: str
    predicted_label: int
    predicted_score: float
    confidence: float
    backend: str
    analyzer_response: str
    proposer_response: str
    skeptic_response: str
    judge_response: str
    retrieved_examples: List[RetrievedExample] = field(default_factory=list)
    gold_label: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["retrieved_examples"] = [asdict(item) for item in self.retrieved_examples]
        return payload
