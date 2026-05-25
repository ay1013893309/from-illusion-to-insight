"""
Verdict parsing utilities for extracting judge predictions and confidence.
"""
import re
from typing import Optional, Tuple


VERDICT_PATTERNS = [
    r"final\s*prediction\s*[:\-]?\s*\**\s*(benign|defective)\s*\**",
    r"final\s*decision\s*[:\-]?\s*\**\s*(benign|defective)\s*\**",
    r"prediction\s*[:\-]?\s*\**\s*(benign|defective)\s*\**",
]


def parse_judge_verdict(text: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Parse the Judge's decision text -> ('BENIGN' or 'DEFECTIVE', 0 or 1).

    Handles variations like:
      - ### Final Prediction: BENIGN
      - Final Prediction: **Defective**
      - final-prediction: benign
      - final prediction - defective
    
    Args:
        text: Judge's response text
    
    Returns:
        Tuple of (verdict_str, verdict_int) where:
        - verdict_str: 'BENIGN' or 'DEFECTIVE'
        - verdict_int: 0 for BENIGN, 1 for DEFECTIVE
        - Returns (None, None) if unparseable
    """
    if not text:
        return None, None

    text_normalized = text.lower()

    for pattern in VERDICT_PATTERNS:
        match = re.search(pattern, text_normalized, flags=re.IGNORECASE)
        if match:
            verdict_str = match.group(1).upper()
            verdict_int = 0 if verdict_str == "BENIGN" else 1
            return verdict_str, verdict_int

    # Fallback: inspect the tail of the response where the mandated final lines should live.
    tail = "\n".join(text_normalized.splitlines()[-6:])
    benign_pos = tail.rfind("benign")
    defective_pos = tail.rfind("defective")
    if benign_pos == -1 and defective_pos == -1:
        return None, None
    if benign_pos > defective_pos:
        return "BENIGN", 0
    return "DEFECTIVE", 1

    return None, None


def parse_confidence(text: str) -> Optional[int]:
    """
    Extract confidence percentage from Judge response.
    
    Handles variations like:
      - ### Confidence: 85
      - confidence: 0.85 (converts to 85)
      - Confidence (85%)
    
    Args:
        text: Judge's response text
    
    Returns:
        Confidence percentage (0-100) or None if not found
    """
    if not text:
        return None

    patterns = [
        r"confidence\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*%?",
        r"confidence\s*\((\d+(?:\.\d+)?)\s*%?\)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue

        conf_str = match.group(1)
        conf_val = float(conf_str)
        if conf_val <= 1.0:
            conf_val = conf_val * 100
        return max(0, min(100, int(conf_val)))

    return None
