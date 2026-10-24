# user_simulator.py
from abc import ABC, abstractmethod
import os
import time
from typing import List, Optional, Union

from openai import OpenAI
import torch
from transformers import PreTrainedModel, GenerationConfig

from online_sdpo.helpers.model import generate_chat_batch
from prompts.prompt import (
    build_user_feedback_messages,
    get_phase1_feedback_fallback,
    get_phase1_feedback_user_template,
    get_phase1_user_prompt,
)
from prompts.style_prompt import USER_SIMULATOR_SYSTEM_PROMPT


class UserSimulator(ABC):

    DEFAULT_TEMPERATURE = 0.7

    def __init__(
        self,
        max_new_tokens: int,
        language: str,
    ) -> None:
        self.max_output_tokens = max_new_tokens
        self.temperature = self.DEFAULT_TEMPERATURE
        self.language = language

    @abstractmethod
    def generate_feedbacks(
        self,
        paragraphs: List[str],
        completions: List[str],
        p_next_golds: List[str],
    ) -> List[str]:
        """
        Returns one user feedback string per (prompt, completion, p_next_gold).
        """
        raise NotImplementedError

    def _build_prompt_text(self, p_context: str, completion: str, p_next_gold: str) -> str:
        """Return the fully formatted string the model sees for one example."""

        user_msg = get_phase1_feedback_user_template(self.language).format(
            context=p_context,
            completion=completion,
            reference=p_next_gold,
        )
        chat = [
            {"role": "system", "content": get_phase1_user_prompt(self.language).strip()},
            {"role": "user", "content": user_msg},
        ]

        return chat


class LocalUserSimulator(UserSimulator):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer,
        device: torch.device,
        max_input_tokens: int,
        max_new_tokens: int,
        language: str,
    ):
        super().__init__(
            max_new_tokens=max_new_tokens,
            language=language,
        )
        self.model = model
        self.tok = tokenizer
        self.device = device
        self.max_input_tokens = max_input_tokens

    @torch.no_grad()
    def generate_feedbacks(
        self,
        paragraphs: List[str],
        completions: List[str],
        p_next_golds: List[str],
    ) -> List[str]:
        
        self.model.eval()
        chats = [
            self._build_prompt_text(p, c, p_next)
            for p, c, p_next in zip(paragraphs, completions, p_next_golds)
        ]

        return generate_chat_batch(
            self.model,
            self.tok,
            self.device,
            chats,
            self.max_input_tokens,
            self.max_output_tokens,
            self.temperature,
        )


class OpenAIUserSimulator(UserSimulator):
    """OpenAI-backed user simulator for online feedback generation."""

    # GPT-5 reasoning tokens also consume the completion budget. 250 is often
    # enough for GPT-4.1-mini, but can leave GPT-5 with no visible feedback.

    def __init__(
        self,
        model: str,
        max_new_tokens: int,
        language: str,
        max_retries: int = 8,
        base_backoff_s: float = 0.75,
        api_key_env: str = "OPENAI_API_KEY",
    ):
        super().__init__(
            max_new_tokens=max_new_tokens,
            language=language,
        )
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"{api_key_env} is not set")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_retries = int(max_retries)
        self.base_backoff_s = float(base_backoff_s)

    def _one_call(self, chat: List[dict]) -> str:
        request = {
            "model": self.model,
            "messages": chat,
            "max_completion_tokens": self.max_output_tokens,
        }
        is_gpt5 = self.model.lower().startswith("gpt-5")
        # GPT-5-family reasoning models do not accept arbitrary sampling
        # temperature values. Keep the API default and constrain reasoning
        # instead; GPT-4.x continues to use the requested temperature.
        if is_gpt5:
            request["reasoning_effort"] = "low"
        else:
            request["temperature"] = self.temperature

        response = self.client.chat.completions.create(**request)
        text = (response.choices[0].message.content or "").strip()
        if not text and getattr(response.choices[0], "finish_reason", None) == "length":
            raise RuntimeError(
                f"{self.model} exhausted max_completion_tokens="
                f"{self.max_output_tokens} before producing visible feedback"
            )
        return text if text else get_phase1_feedback_fallback(self.language)

    def generate_feedbacks(
        self,
        paragraphs: List[str],
        completions: List[str],
        p_next_golds: List[str],
    ) -> List[str]:
        if len(paragraphs) != len(completions):
            raise ValueError("paragraphs and completions must have the same length")
        if len(p_next_golds) != len(paragraphs):
            raise ValueError("paragraphs and p_next_golds must have the same length")

        outputs: List[str] = []
        chats = [
            self._build_prompt_text(p, c, p_next)
            for p, c, p_next in zip(paragraphs, completions, p_next_golds)
        ]
        for chat in chats:
            last_error: Optional[Exception] = None
            for attempt in range(self.max_retries):
                try:
                    outputs.append(self._one_call(chat))
                    break
                except Exception as error:
                    last_error = error
                    time.sleep(self.base_backoff_s * (2**attempt))
            else:
                raise RuntimeError(
                    f"OpenAI user simulation failed after {self.max_retries} retries: {last_error}"
                )
        return outputs


class StyleUserSimulator(ABC):
    """Dataset-agnostic style-feedback simulator used by Figure 5."""

    def __init__(self, max_new_tokens: int):
        self.max_output_tokens = max_new_tokens
        self.temperature = 0.7

    def _build_prompt_text(self, prompt: str, completion: str):
        return build_user_feedback_messages(prompt, completion)


class LocalStyleUserSimulator(StyleUserSimulator):
    def __init__(self, model, tokenizer, device, max_input_tokens, max_new_tokens):
        super().__init__(max_new_tokens)
        self.model = model
        self.tok = tokenizer
        self.device = device
        self.max_input_tokens = max_input_tokens

    @torch.no_grad()
    def generate_feedbacks(self, prompts: List[str], completions: List[str]) -> List[str]:
        if len(prompts) != len(completions):
            raise ValueError("prompts and completions must have the same length")
        self.model.eval()
        chats = [self._build_prompt_text(p, c) for p, c in zip(prompts, completions)]
        return generate_chat_batch(
            self.model, self.tok, self.device, chats,
            self.max_input_tokens, self.max_output_tokens, self.temperature,
        )


class OpenAIStyleUserSimulator(StyleUserSimulator):
    def __init__(self, model: str, max_new_tokens: int, api_key_env: str = "OPENAI_API_KEY"):
        super().__init__(max_new_tokens)
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"{api_key_env} is not set")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate_feedbacks(self, prompts: List[str], completions: List[str]) -> List[str]:
        if len(prompts) != len(completions):
            raise ValueError("prompts and completions must have the same length")
        outputs = []
        for prompt, completion in zip(prompts, completions):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self._build_prompt_text(prompt, completion),
                max_completion_tokens=self.max_output_tokens,
                **({"reasoning_effort": "low"} if self.model.lower().startswith("gpt-5") else {"temperature": self.temperature}),
            )
            outputs.append((response.choices[0].message.content or "").strip())
        return outputs
