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
    if references is None:
        raise ValueError(
            "Phase 1 Judge requires references (the master continuations y_hat)."
        )
    lengths = {len(prompts), len(references), len(completions_a), len(completions_b)}
    if len(lengths) != 1:
        raise ValueError(
            "prompts, references, completions_a, and completions_b must have "
            "the same length."
        )
    return references


def parse_judgement(text: str) -> int:
    """Parse A/B/C into 0/1/-1, matching the Phase 1 judge contract."""
    clean = text.strip()
    if not clean:
        return -1

    # The decision marker must be the final labelled decision, but it may be
    # on the same line as the final rationale sentence.  Some API judges emit
    # ``...。判定：A`` instead of putting ``判定：A`` on a new line.
    match = re.search(
        r"(?:Judg(?:e)?ment|判定)[ \t]*[:：][ \t]*([ABC])"
        r"[ \t]*(?:[.!。])?[ \t]*$",
        clean,
        re.IGNORECASE,
    )
    if match is None:
        return -1
    return {"A": 0, "B": 1, "C": -1}[match.group(1).upper()]


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
