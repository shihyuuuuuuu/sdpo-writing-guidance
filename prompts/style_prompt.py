"""Shared prompts for preference-based user simulation and style judging.

These prompts are dataset-agnostic: ``prompt`` can be a TL;DR request,
HelpSteer instruction, or any other single-turn user request.
"""

from __future__ import annotations

from typing import Dict, List


# The Figure 5 default persona.  It is intentionally part of the prompt,
# rather than a runtime ``style`` argument, so every supported dataset uses
# the same preference definition.
STYLE_PERSONA = (
    "USER PROFILE: You are a user who specifically prefers CONCISE, CASUAL, "
    "and BEGINNER-FRIENDLY responses. You want short and clear answers, "
    "avoiding long, formal, or technically dense answers."
)


USER_SIMULATOR_SYSTEM_PROMPT = f"""{STYLE_PERSONA}

You are simulating the user's next message in a chat with an AI assistant.

Rules:
- Respond as the user described in the USER PROFILE above.
- Provide feedback ONLY on the assistant's response with respect to that preference.
- Do NOT answer the original request yourself.
- Do NOT give general feedback unrelated to the USER PROFILE.
- Keep the feedback brief and output ONLY the user's message text.
"""


USER_FEEDBACK_USER_TEMPLATE = """Original user request:
{prompt}

Assistant response:
{completion}

Write the user's next message:"""


STYLE_JUDGE_SYSTEM_PROMPT = f"""{STYLE_PERSONA}

You are acting as a strict evaluator of which response better matches the
user's preference described in the USER PROFILE.

Rules:
- Judge ONLY style, tone, formatting, verbosity, and complexity relative to the profile.
- Do NOT judge factual correctness.
- Do NOT rewrite either response.
- You may reason briefly about your decision.
- End with exactly one of: Judgement: A, Judgement: B, or Judgement: C.
- C means tie or uncertainty.
"""


STYLE_JUDGE_USER_TEMPLATE = """User prompt:
{prompt}

Response A:
{candidate_a}

Response B:
{candidate_b}

Which response do you prefer as this user? Reason briefly, then end with exactly: Judgement: A, Judgement: B, or Judgement: C."""


def build_assistant_messages(prompt: str) -> List[Dict[str, str]]:
    """Build the assistant's normal single-turn chat for any dataset prompt."""
    return [{"role": "user", "content": prompt.strip()}]


def build_user_feedback_messages(
    prompt: str,
    completion: str,
) -> List[Dict[str, str]]:
    """Build a chat for generating the user's next feedback message."""
    return [
        {"role": "system", "content": USER_SIMULATOR_SYSTEM_PROMPT.strip()},
        {
            "role": "user",
            "content": USER_FEEDBACK_USER_TEMPLATE.format(
                prompt=prompt.strip(), completion=completion.strip()
            ).strip(),
        },
    ]


def build_style_judge_messages(
    prompt: str,
    candidate_a: str,
    candidate_b: str,
) -> List[Dict[str, str]]:
    """Build a pairwise style-preference judge chat."""
    return [
        {"role": "system", "content": STYLE_JUDGE_SYSTEM_PROMPT.strip()},
        {
            "role": "user",
            "content": STYLE_JUDGE_USER_TEMPLATE.format(
                prompt=prompt.strip(),
                candidate_a=candidate_a.strip(),
                candidate_b=candidate_b.strip(),
            ).strip(),
        },
    ]


__all__ = [
    "STYLE_PERSONA",
    "USER_SIMULATOR_SYSTEM_PROMPT",
    "USER_FEEDBACK_USER_TEMPLATE",
    "STYLE_JUDGE_SYSTEM_PROMPT",
    "STYLE_JUDGE_USER_TEMPLATE",
    "build_assistant_messages",
    "build_user_feedback_messages",
    "build_style_judge_messages",
]
