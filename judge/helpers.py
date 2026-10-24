from __future__ import annotations

import re
from typing import Dict, List, Optional


def validate_pairwise_inputs(
    prompts: List[str],
    references: Optional[List[str]],
    completions_a: List[str],
    completions_b: List[str],
) -> List[str]:
    """Validate and return references for a Phase 1 pairwise comparison."""
    lengths = {len(prompts), len(completions_a), len(completions_b)}
    if references is not None:
        lengths.add(len(references))
    if len(lengths) != 1:
        raise ValueError(
            "prompts, references (when provided), completions_a, and completions_b must have "
            "the same length."
        )
    return references or [""] * len(prompts)


def parse_judgement(text: str) -> int:
    """Parse A/B/C into 0/1/-1, matching the Phase 1 judge contract."""
    clean = text.strip()
    if not clean:
        return -1

    matches = re.findall(
        r"(?:Judg(?:e)?ment|判定)\s*[:：]\s*([ABC])", clean, re.IGNORECASE
    )
    if matches:
        return {"A": 0, "B": 1, "C": -1}[matches[-1].upper()]

    # Conservative fallback: only accept a standalone final A/B/C token.
    fallback = re.findall(r"\b([ABC])\b", clean.upper())
    if not fallback:
        return -1
    return {"A": 0, "B": 1, "C": -1}[fallback[-1]]


def invert_pairwise_decisions(decisions: List[int]) -> List[int]:
    """Invert decisions from a swapped B/A comparison."""
    return [{0: 1, 1: 0}.get(decision, -1) for decision in decisions]


def resolve_symmetric_decisions(
    decisions_ab: List[int],
    decisions_ba: List[int],
) -> List[int]:
    """Keep only decisions that agree between AB and the inverted BA run."""
    if len(decisions_ab) != len(decisions_ba):
        raise ValueError("AB and BA decisions must have the same length.")
    inverted_ba = invert_pairwise_decisions(decisions_ba)
    return [
        decision_ab if decision_ab == decision_ba else -1
        for decision_ab, decision_ba in zip(decisions_ab, inverted_ba)
    ]
