"""
tools/critique.py
-----------------
Self-critique helper utilities for the Reviewer agent.
"""
from __future__ import annotations
from typing import List

def checklist(text: str) -> List[str]:
    """
    Returns a list of concrete critique notes to improve the draft.
    """
    notes: List[str] = []
    if "risks" not in text.lower():
        notes.append("Add a 'Risks' section with 2-4 bullets.")
    if "sources" not in text.lower():
        notes.append("Add a 'Sources' line with at least 2 citations or 'N/A'.")
    if "catalyst" not in text.lower():
        notes.append("Mention near-term catalysts (earnings, product launches).")
    return notes
