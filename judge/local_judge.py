from __future__ import annotations

from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel

from .judge_base import JudgeBase
from .helpers import (
    parse_judgement,
    resolve_symmetric_decisions,
    validate_pairwise_inputs,
)
from online_sdpo.helpers.model import generate_chat_batch
from prompts.prompt import build_phase1_judge_messages


class LocalJudge(JudgeBase):
    """Reference-based pairwise judge for Phase 1.

    Decision meanings are always relative to the original argument order:
    0 = candidate A wins, 1 = candidate B wins, -1 = tie/uncertain.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer,
        device: torch.device,
        max_input_tokens: int = 2500, # 參考：平均 2000 左右
        tie_margin: float = 0.05,
        author: str = "Unknown author",
        language: str = "en",
    ):
        super().__init__()
        self.model = model.eval()
        self.tok = tokenizer
        self.device = device
        self.max_input_tokens = max_input_tokens
        self.tie_margin = tie_margin
        self.author = author
        self.language = language
        self.last_raw_outputs: List[str] = []
        # Raw generated rationales from the most recent generated-judging run.
        # Keep AB and BA separately so callers can inspect the symmetry check.
        self.last_raw_outputs_ab: List[str] = []
        self.last_raw_outputs_ba: List[str] = []
        self.last_decisions_ab: List[int] = []
        self.last_decisions_ba: List[int] = []

        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

    def _label_variants(self) -> Dict[str, List[List[int]]]:
        variants = {
            "A": ["A", " A", "\nA", "\n A"],
            "B": ["B", " B", "\nB", "\n B"],
            "C": ["C", " C", "\nC", "\n C"],
        }
        output: Dict[str, List[List[int]]] = {}
        for label, strings in variants.items():
            token_variants = [
                ids
                for value in strings
                if (ids := self.tok.encode(value, add_special_tokens=False))
            ]
            if not token_variants:
                raise RuntimeError(f"Tokenizer produced no ids for label {label}.")
            output[label] = token_variants
        return output

    @torch.no_grad()
    def _score_label_variant_batch(
        self,
        prompt_ids: torch.Tensor,
        prompt_attn: torch.Tensor,
        prompt_lens: torch.Tensor,
        label_ids: List[int],
    ) -> torch.Tensor:
        batch_size = prompt_ids.size(0)
        label = torch.tensor(label_ids, dtype=torch.long, device=prompt_ids.device)
        label_len = label.numel()

        # Select active tokens explicitly. This also works with left padding.
        sequences = [
            torch.cat([prompt_ids[i][prompt_attn[i].bool()], label], dim=0)
            for i in range(batch_size)
        ]
        max_len = max(sequence.numel() for sequence in sequences)
        pad_id = (
            self.tok.pad_token_id
            if self.tok.pad_token_id is not None
            else self.tok.eos_token_id
        )
        input_ids = torch.full(
            (batch_size, max_len), pad_id, dtype=torch.long, device=prompt_ids.device
        )
        attention_mask = torch.zeros(
            (batch_size, max_len), dtype=prompt_attn.dtype, device=prompt_attn.device
        )
        for i, sequence in enumerate(sequences):
            input_ids[i, : sequence.numel()] = sequence
            attention_mask[i, : sequence.numel()] = 1

        logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits

        positions = (
            prompt_lens.to(prompt_ids.device).view(batch_size, 1) - 1
            + torch.arange(label_len, device=prompt_ids.device).view(1, label_len)
        )
        batch_indices = torch.arange(batch_size, device=prompt_ids.device).view(
            batch_size, 1
        )
        label_logits = logits[batch_indices, positions, :]
        label_logprobs = F.log_softmax(label_logits.float(), dim=-1)
        labels = label.view(1, label_len).expand(batch_size, label_len)
        return label_logprobs.gather(-1, labels.unsqueeze(-1)).squeeze(-1).sum(dim=1)

    @torch.no_grad()
    def _score_abc_batch(self, inputs_text: List[str]) -> Dict[str, torch.Tensor]:
        encoded = self.tok(
            inputs_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_tokens,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        prompt_ids = encoded["input_ids"]
        prompt_attn = encoded["attention_mask"]
        prompt_lens = prompt_attn.sum(dim=1)

        scores: Dict[str, torch.Tensor] = {}
        for label, variants in self._label_variants().items():
            best: Optional[torch.Tensor] = None
            for variant in variants:
                score = self._score_label_variant_batch(
                    prompt_ids, prompt_attn, prompt_lens, variant
                )
                best = score if best is None else torch.maximum(best, score)
            scores[label] = best
        return scores

    def _decide_from_scores(self, scores_abc: Dict[str, torch.Tensor]) -> List[int]:
        score_a, score_b, score_c = (
            scores_abc["A"],
            scores_abc["B"],
            scores_abc["C"],
        )
        stacked = torch.stack([score_a, score_b, score_c], dim=1)
        top2 = torch.topk(stacked, k=2, dim=1).values
        confident = (top2[:, 0] - top2[:, 1]) >= self.tie_margin
        best = torch.argmax(stacked, dim=1)
        clear_ab = torch.abs(score_a - score_b) >= self.tie_margin

        decisions = torch.full(
            (stacked.size(0),), -1, dtype=torch.long, device=stacked.device
        )
        decisions[confident & clear_ab & (best == 0)] = 0
        decisions[confident & clear_ab & (best == 1)] = 1
        return decisions.tolist()

    @torch.no_grad()
    def choose_batch_generated(
        self,
        prompts: List[str],
        completions_a: List[str],
        completions_b: List[str],
        references: Optional[List[str]] = None,
        batch_size: int = 16,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> List[int]:
        """Generate textual judgements for A/B, with AB/BA symmetry checking."""
        references = validate_pairwise_inputs(
            prompts, references, completions_a, completions_b
        )
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        decisions_final: List[int] = []
        self.last_raw_outputs_ab = []
        self.last_raw_outputs_ba = []
        self.last_decisions_ab = []
        self.last_decisions_ba = []
        for start in range(0, len(prompts), batch_size):
            end = min(len(prompts), start + batch_size)
            batch_contexts = prompts[start:end]
            batch_references = references[start:end]
            batch_a = completions_a[start:end]
            batch_b = completions_b[start:end]

            d_ab = self._get_generation_decisions(
                batch_contexts, batch_references, batch_a, batch_b
            )
            raw_ab = list(self.last_raw_outputs)
            if progress_callback is not None:
                progress_callback(end - start)
            d_ba = self._get_generation_decisions(
                batch_contexts, batch_references, batch_b, batch_a
            )
            raw_ba = list(self.last_raw_outputs)
            if progress_callback is not None:
                progress_callback(end - start)

            decisions_final.extend(resolve_symmetric_decisions(d_ab, d_ba))
            self.last_raw_outputs_ab.extend(raw_ab)
            self.last_raw_outputs_ba.extend(raw_ba)
            self.last_decisions_ab.extend(d_ab)
            self.last_decisions_ba.extend(d_ba)
        self.last_raw_outputs = self.last_raw_outputs_ab + self.last_raw_outputs_ba
        return decisions_final

    def _get_generation_decisions(
        self,
        contexts: List[str],
        references: List[str],
        candidates_a: List[str],
        candidates_b: List[str],
    ) -> List[int]:
        messages_list = [
            build_phase1_judge_messages(
                context, reference, candidate_a, candidate_b,
                author=self.author, language=self.language,
            ) for context, reference, candidate_a, candidate_b in zip(
                contexts, references, candidates_a, candidates_b
            )
        ]

        generated = generate_chat_batch(
            self.model,
            self.tok,
            self.device,
            messages_list,
            self.max_input_tokens,
            self.max_output_tokens,
            temperature=0,
        )
        self.last_raw_outputs = generated

        return [parse_judgement(text) for text in generated]
