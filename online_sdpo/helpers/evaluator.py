import os
import time
from functools import cached_property
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import GenerationConfig
from tqdm import tqdm

from .logging import print_eval_result, print_first_judge
from .record import append_jsonl
from ..online_sdpo_updater import OnlineSDPOUpdater
from prompts.prompt import build_phase1_judge_messages, build_phase1_messages


class Evaluator:
    p_contexts: List[str]
    p_nexts: List[str]
    chats: List[List[Dict[str, str]]]
    
    max_new_tokens: int
    temperature: float
    eval_dir: str

    def __init__(self, eval_ds, lang: str, max_new_tokens: int, temperature: float, eval_dir: str):
        self.p_contexts = [ex["p_context"].strip() for ex in eval_ds]
        self.p_nexts = [ex["p_next"].strip() for ex in eval_ds]
        self.chats = [build_phase1_messages(c, lang) for c in self.p_contexts]
        self._base_completions = []
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.eval_dir = eval_dir

    @property
    def base_completions(self) -> List[str]:
        return self._base_completions
    
    @base_completions.setter
    def base_completions(self, new_value):
        self._base_completions = new_value
        self.__dict__.pop("baseline_avg_len", None)

    @cached_property
    def baseline_avg_len(self):
        return sum(len(c) for c in self._base_completions) / max(len(self._base_completions), 1)

    def generate_completions(self, updater: OnlineSDPOUpdater, batch_size=16) -> List[str]:
        """Generate completions for all eval prompts using the current model."""
        completions = []

        for start in range(0, len(self.chats), batch_size):
            batch = self.chats[start:start + batch_size]
            completions.extend(
                updater.generate_responses_batch(
                    batch,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                )
            )

        return completions

    @staticmethod
    def _avg_completion_tokens(completions: List[str], tokenizer) -> float:
        if not completions:
            return 0.0
        encoded = tokenizer(
            completions,
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )["input_ids"]
        return sum(len(ids) for ids in encoded) / len(encoded)

    # 評估這個 updater (和 baseline 比較)
    def evaluate(
        self,
        step: int,
        interactions_seen: int,
        updater: OnlineSDPOUpdater,
        seed,
        judge,
    ):
        # 先檢查是否有 baseline completions
        eval_metrics_path = os.path.join(self.eval_dir, "eval_metrics.jsonl")
        eval_completions_path = os.path.join(self.eval_dir, "eval_completions.jsonl")

        if updater.config.model_vllm:
            self._check_vllm_adapter(step, updater)

        torch.cuda.empty_cache()
        print(f"\n[EVAL @ step {step}] Generating eval completions...", flush=True)
        t_eval = time.time()
        current_completions = self.generate_completions(updater)
        eval_gen_time = time.time() - t_eval
        current_avg_len = sum(len(c) for c in current_completions) / max(len(current_completions), 1)
        current_avg_tokens = self._avg_completion_tokens(
            current_completions, updater.tokenizer,
        )
        baseline_avg_tokens = self._avg_completion_tokens(
            self.base_completions, updater.tokenizer,
        )
        print(f"[EVAL @ step {step}] avg_len={current_avg_len:.0f} chars (baseline={self.baseline_avg_len:.0f})", flush=True)
        
        print(f"[EVAL @ step {step}] Running judge ({len(self.p_contexts)} comparisons)...", flush=True)
        t_judge = time.time()
        eval_metrics = self._run_evaluation(
            judge, current_completions, bootstrap_seed=seed,
            progress_desc=f"Judge step {step}",
        )
        judge_time = time.time() - t_judge

        eval_metrics["step"] = step
        eval_metrics["interactions_seen"] = interactions_seen
        eval_metrics["eval_gen_time_s"] = round(eval_gen_time, 1)
        eval_metrics["judge_time_s"] = round(judge_time, 1)
        eval_metrics["current_avg_tokens"] = round(current_avg_tokens, 2)
        eval_metrics["baseline_avg_tokens"] = round(baseline_avg_tokens, 2)
        if step == 1:
            messages = build_phase1_judge_messages(
                self.p_contexts[0], self.p_nexts[0], current_completions[0],
                self.base_completions[0], author=judge.author, language=judge.language,
            )
            decisions = eval_metrics.get("decisions", [])
            print_first_judge(messages, decisions, judge)
        
        append_jsonl(eval_metrics_path, eval_metrics)
        print_eval_result(eval_metrics, eval_gen_time, judge_time)
        eval_completion_record = self._get_eval_record(
            eval_metrics, judge, current_completions,
        )
        append_jsonl(eval_completions_path, eval_completion_record)

    def _check_vllm_adapter(self, step: int, updater: OnlineSDPOUpdater):
        adapter_path = updater._latest_lora_path
        request_id = updater._lora_request_id
        print(
            f"[EVAL @ step {step}] vLLM LoRA: path={adapter_path!r}, request_id={request_id}",
            flush=True,
        )
        adapter_saved = adapter_path is not None and all(
            os.path.isfile(os.path.join(adapter_path, name))
            for name in ("adapter_config.json", "adapter_model.safetensors")
        )
        if not adapter_saved or request_id != step:
            raise RuntimeError(
                f"[EVAL @ step {step}] vLLM LoRA adapter is missing or stale "
                f"(path={adapter_path!r}, request_id={request_id}). "
                "Refusing to evaluate the base model as current."
            )
        
    def _get_eval_record(self, eval_metrics, judge, current_completions):
        debug_n = min(5, len(self.p_contexts))
        decisions = eval_metrics.get("decisions", [])
        raw_ab = getattr(judge, "last_raw_outputs_ab", [])
        raw_ba = getattr(judge, "last_raw_outputs_ba", [])
        decisions_ab = getattr(judge, "last_decisions_ab", [])
        decisions_ba = getattr(judge, "last_decisions_ba", [])

        eval_completion_record = {
            "step": eval_metrics["step"],
            "completions": [
                {
                    "prompt": p,
                    "current": c,
                    "baseline": b,
                    "judge": {
                        # ab_raw：A=current、B=baseline
                        # ba_raw：A=baseline、B=current
                        # ab_decision / ba_decision：各自解析出的結果
                        # final_decision：
                        # - 0：current 勝
                        # - 1：baseline 勝
                        # - -1：tie 或兩次判定不一致
                        "ab_raw": raw_ab[i] if i < len(raw_ab) else "",
                        "ba_raw": raw_ba[i] if i < len(raw_ba) else "",
                        "ab_decision": decisions_ab[i] if i < len(decisions_ab) else None,
                        "ba_decision": decisions_ba[i] if i < len(decisions_ba) else None,
                        "final_decision": decisions[i] if i < len(decisions) else None,
                    },
                } for i, (p, c, b) in enumerate(
                    zip(self.p_contexts[:debug_n], current_completions[:debug_n], self.base_completions[:debug_n])
                )
            ]
        }

        return eval_completion_record

    def _run_evaluation(
        self,
        judge,
        current_completions: List[str],
        bootstrap_seed: Optional[int] = None,
        progress_desc: Optional[str] = None,
    ) -> Dict:
        """Run pairwise judge comparison, return metrics dict with per-example decisions."""
        total_calls = 2 * len(self.p_contexts)
        progress = tqdm(
            total=total_calls,
            desc=progress_desc or "Judging",
            unit="judgement",
            dynamic_ncols=True,
            leave=False,
        )
        try:
            decisions = judge.choose_batch_generated(
                prompts=self.p_contexts,
                references=self.p_nexts,
                completions_a=current_completions,
                completions_b=self.base_completions,
                progress_callback=progress.update,
            )
        finally:
            progress.close()
        metrics = compute_metrics(decisions, bootstrap_seed=bootstrap_seed)
        metrics["decisions"] = decisions  # per-example: 0=current wins, 1=baseline wins, -1=tie
        return metrics


def _non_tie_outcomes(decisions: List[int]) -> np.ndarray:
    return np.array([1 if d == 0 else 0 for d in decisions if d != -1], dtype=np.int8)


def bootstrap_prop_se(y: np.ndarray, B: int = 10_000, seed: Optional[int] = None) -> float:
    n = int(y.size)
    if n == 0:
        return float("nan")
    if n == 1:
        return 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    boot_means = y[idx].mean(axis=1)
    return float(boot_means.std(ddof=1))


def compute_metrics(decisions: List[int], bootstrap_seed: Optional[int] = None) -> Dict:
    n = len(decisions)
    if n == 0:
        return {"n": 0, "coverage": 0.0}

    ties = sum(d == -1 for d in decisions)
    wins_a = sum(d == 0 for d in decisions)
    wins_b = sum(d == 1 for d in decisions)
    n_eff = wins_a + wins_b
    coverage = 1.0 - (ties / n)

    if n_eff == 0:
        return {"n": n, "wins_a": 0, "wins_b": 0, "ties": ties,
                "coverage": coverage, "winrate_a": float("nan"),
                "se": float("nan"), "n_effective": 0}

    p_hat = wins_a / n_eff
    se_analytic = float(np.sqrt(p_hat * (1.0 - p_hat) / n_eff) * 100.0) if n_eff > 1 else 0.0
    se_boot = float(bootstrap_prop_se(_non_tie_outcomes(decisions), seed=bootstrap_seed) * 100.0)

    return {
        "n": n,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "coverage": float(coverage),
        "winrate_a": float(p_hat * 100.0),
        "se": se_analytic,
        "se_bootstrap": se_boot,
        "n_effective": n_eff,
    }
