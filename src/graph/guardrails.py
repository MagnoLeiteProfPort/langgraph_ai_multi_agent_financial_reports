"""
guardrails.py
-------------
Simple guardrails: prompt-time instructions and output-time validator.
In production, extend with policy engines or LLM-based classifiers.
"""
from __future__ import annotations
from typing import List, Tuple

DISALLOWED = ["guaranteed returns", "inside information"]

DISCLAIMER = (
    "This content is for informational purposes only and is not financial advice. "
    "Do your own research and consider consulting a licensed professional."
)

def validate_output(text: str) -> Tuple[str, List[str]]:
    flags: List[str] = []
    for bad in DISALLOWED:
        if bad.lower() in text.lower():
            flags.append(f"disallowed_phrase:{bad}")
    if "not financial advice" not in text.lower():
        text = text.rstrip() + "\n\n" + DISCLAIMER
    return text, flags
