from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, List, Optional


class JudgeBase(ABC):
    """Common public interface for local and API-backed Phase 1 judges."""

    DEFAULT_MAX_OUTPUT_TOKENS = 512 # 參考：平均 200 左右

    def __init__(self) -> None:
        self.max_output_tokens = self.DEFAULT_MAX_OUTPUT_TOKENS

    @abstractmethod
    def choose_batch_generated(
        self,
        prompts: List[str],
        completions_a: List[str],
        completions_b: List[str],
        references: Optional[List[str]] = None,
        batch_size: int = 16,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> List[int]:
        """Compare candidates by generating and parsing a judgement."""
        raise NotImplementedError
