"""Cross-project, change-aware defect prediction package."""

from .pipeline import CrossProjectChangePredictor
from .schemas import ChangePair, PredictionResult, RetrievedExample, StaticFileSample

__all__ = [
    "ChangePair",
    "CrossProjectChangePredictor",
    "PredictionResult",
    "RetrievedExample",
    "StaticFileSample",
]
