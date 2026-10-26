from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Optional

from openai import OpenAI

from .judge_base import JudgeBase
from .helpers import (
    parse_judgement,
    resolve_symmetric_decisions,
    validate_pairwise_inputs,
)
from prompts.prompt import build_phase1_judge_messages


class OpenAIJudge(JudgeBase):
    """OpenAI backend for the Phase 1 reference-based pairwise judge.

    This class follows the same public contract as ``judge.Judge``:
      0  -> candidate A wins (the SDPO output, if used that way)
      1  -> candidate B wins (the baseline output, if used that way)
     -1  -> tie or uncertain

    The only backend-specific difference is that judgements are generated via
    the OpenAI API instead of a local Transformers model.
    """

    DEFAULT_MODEL = "gpt-4.1-mini"
    # GPT-5 reasoning tokens share this budget with the visible rationale and
    # judgement. Keep enough room for both the reasoning and final label.

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_retries: int = 8,
        base_backoff_s: float = 0.75,
        max_workers: int = 16,
        api_key_env: str = "OPENAI_API_KEY",
        author: str = "Unknown author",
        language: str = "en",
    ):
        super().__init__()
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"{api_key_env} is not set")

        self.client = OpenAI(api_key=api_key)
        self.model = model or self.DEFAULT_MODEL
        self.temperature = float(temperature)
        self.max_retries = int(max_retries)
        self.base_backoff_s = float(base_backoff_s)
        self.max_workers = int(max_workers)
        self.author = author
        self.language = language
        self.last_raw_outputs: List[str] = []
        # Raw generated rationales from the most recent generated-judging run.
        self.last_raw_outputs_ab: List[str] = []
        self.last_raw_outputs_ba: List[str] = []
        self.last_decisions_ab: List[int] = []
        self.last_decisions_ba: List[int] = []

    def _call_once(
        self,
        context: str,
        reference: str,
        candidate_a: str,
        candidate_b: str,
    ) -> tuple[int, str]:
        request = {
            "model": self.model,
            "messages": build_phase1_judge_messages(
                context, reference, candidate_a, candidate_b,
                author=self.author, language=self.language,
            ),
            "max_completion_tokens": self.max_output_tokens,
        }
        is_gpt5 = self.model.lower().startswith("gpt-5")
        # GPT-5-family reasoning models do not accept arbitrary sampling
        # temperature values. Keep the API default and constrain reasoning;
        # GPT-4.x continues to use the requested temperature.
        if is_gpt5:
            request["reasoning_effort"] = "low"
        else:
            request["temperature"] = self.temperature

        response = self.client.chat.completions.create(**request)
        text = (response.choices[0].message.content or "").strip()
        if not text and getattr(response.choices[0], "finish_reason", None) == "length":
            raise RuntimeError(
                f"{self.model} exhausted max_completion_tokens="
                f"{self.max_output_tokens} before producing a judgement"
            )
        return parse_judgement(text), text

    def _call_with_retries(
        self,
        context: str,
        reference: str,
        candidate_a: str,
        candidate_b: str,
    ) -> tuple[int, str]:
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                return self._call_once(
                    context, reference, candidate_a, candidate_b
                )
            except Exception as error:
                last_error = error
                if attempt + 1 < self.max_retries:
                    time.sleep(self.base_backoff_s * (2**attempt))
        raise RuntimeError(
            f"OpenAI judge failed after {self.max_retries} retries: {last_error}"
        )

    def choose_batch_generated(
        self,
        prompts: List[str],
        completions_a: List[str],
        completions_b: List[str],
        references: Optional[List[str]] = None,
        batch_size: int = 16,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> List[int]:
        """Compare A/B against the same reference using AB/BA symmetry."""
        references = validate_pairwise_inputs(
            prompts, references, completions_a, completions_b
        )
        final_decisions: List[int] = []
        self.last_raw_outputs_ab = []
        self.last_raw_outputs_ba = []
        self.last_decisions_ab = []
        self.last_decisions_ba = []

        for start in range(0, len(prompts), batch_size):
            end = min(len(prompts), start + batch_size)
            batch_size_actual = end - start

            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futures_ab = {
                    pool.submit(
                        self._call_with_retries,
                        prompts[i],
                        references[i],
                        completions_a[i],
                        completions_b[i],
                    ): i - start
                    for i in range(start, end)
                }
                futures_ba = {
                    pool.submit(
                        self._call_with_retries,
                        prompts[i],
                        references[i],
                        completions_b[i],
                        completions_a[i],
                    ): i - start
                    for i in range(start, end)
                }

                decisions_ab = {}
                decisions_ba = {}
                raw_outputs_ab = {}
                raw_outputs_ba = {}

                for future in as_completed([*futures_ab, *futures_ba]):
                    if future in futures_ab:
                        index = futures_ab[future]
                        decision, raw = future.result()
                        decisions_ab[index] = decision
                        raw_outputs_ab[index] = raw
                    else:
                        index = futures_ba[future]
                        decision, raw = future.result()
                        decisions_ba[index] = decision
                        raw_outputs_ba[index] = raw
                    if progress_callback is not None:
                        progress_callback(1)

            ordered_ab = [decisions_ab[i] for i in range(batch_size_actual)]
            ordered_ba = [decisions_ba[i] for i in range(batch_size_actual)]
            final_decisions.extend(
                resolve_symmetric_decisions(ordered_ab, ordered_ba)
            )
            raw_ab = [raw_outputs_ab[i] for i in range(batch_size_actual)]
            raw_ba = [raw_outputs_ba[i] for i in range(batch_size_actual)]
            self.last_raw_outputs_ab.extend(raw_ab)
            self.last_raw_outputs_ba.extend(raw_ba)
            self.last_decisions_ab.extend(ordered_ab)
            self.last_decisions_ba.extend(ordered_ba)

        self.last_raw_outputs = self.last_raw_outputs_ab + self.last_raw_outputs_ba

        return final_decisions
