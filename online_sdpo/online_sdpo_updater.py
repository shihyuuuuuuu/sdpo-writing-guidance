"""
Core SDPO update logic for live, single-user chat training.

Owns the model, tokenizer, optimizer, and handles both
generation (eval mode) and per-turn gradient updates (train mode).
"""
import os
import shutil
import threading
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    TextIteratorStreamer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from .helpers.loss import full_distillation_loss, simple_signal_loss
from .helpers.model import generate_chat_batch, load_model_and_tokenizer
from .online_sdpo_updater_config import OnlineSDPOConfig
from prompts.prompt import build_hindsight_messages


class OnlineSDPOUpdater:

    def __init__(self, config: OnlineSDPOConfig):
        self.config = config
        self.step = 0

        # Async training state
        self._train_thread: Optional[threading.Thread] = None
        self._train_lock = threading.Lock()
        self._last_async_metrics: Optional[Dict] = None
        self._async_error: Optional[str] = None

        # vLLM state
        self.llm = None  # vLLM LLM instance
        self._latest_lora_path: Optional[str] = None
        self._lora_request_id: int = 0

        # Force LoRA + async when using vLLM
        if config.model_vllm:
            if not config.use_lora:
                print("[LIVE SDPO] model_vllm=True forces use_lora=True", flush=True)
                config.use_lora = True

        print("[LIVE SDPO] Loading model and tokenizer...", flush=True)
        self._load_model_and_tokenizer()
        print("[LIVE SDPO] Setting up optimizer...", flush=True)
        self._setup_optimizer()
        print("[LIVE SDPO] Updater ready.", flush=True)

    # ------------------------------------------------------------------ #
    #  Initialization
    # ------------------------------------------------------------------ #

    def _load_model_and_tokenizer(self):
        self.model, self.tokenizer = load_model_and_tokenizer(
            self.config.model, self.config.model_gpu, self.config.attn_implementation
        )

        if self.config.use_lora:
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
            self.model.config.use_cache = False

            lora_cfg = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.lora_target_modules,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_cfg)
            self.model.print_trainable_parameters()

        self.device = next(self.model.parameters()).device
        print(f"[LIVE SDPO] Training model loaded on {self.device}", flush=True)

        # vLLM inference engine
        if self.config.model_vllm:
            self._init_vllm()

    def _setup_optimizer(self):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            eps=self.config.adam_epsilon,
        )

    # ------------------------------------------------------------------ #
    #  vLLM initialization
    # ------------------------------------------------------------------ #

    def _init_vllm(self):
        vllm_gpu = str(self.config.vllm_gpu)
        original_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")

        print(f"[LIVE SDPO] Initializing vLLM on GPU {vllm_gpu}...", flush=True)
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = vllm_gpu

            from vllm import LLM

            self.llm = LLM(
                model=self.config.model,
                # Quantize the original checkpoint in-flight for vLLM
                # inference. This keeps the vLLM copy consistent with the
                # QLoRA training setup without requiring a separate
                # pre-quantized checkpoint.
                quantization="bitsandbytes",
                load_format="bitsandbytes",
                dtype="bfloat16",
                max_model_len=4096,
                tensor_parallel_size=1,
                gpu_memory_utilization=self.config.vllm_gpu_mem,
                enable_lora=True,
                max_lora_rank=self.config.lora_r,
            )
        finally:
            if original_visible_devices is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_visible_devices

        print(
            f"[LIVE SDPO] vLLM ready on GPU {vllm_gpu}; "
            f"CUDA_VISIBLE_DEVICES restored to {original_visible_devices!r}.",
            flush=True,
        )

    def generate_responses_batch(
        self,
        messages_list: List[List[Dict[str, str]]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> List[str]:
        """Generate responses for multiple prompts in a single batched call.
        """
        if self.config.model_vllm:
            return self._generate_vllm_batch(messages_list, max_new_tokens, temperature)

        return self._generate_hf_batch(messages_list, max_new_tokens, temperature)

    def _generate_hf_batch(
        self,
        messages_list: List[List[Dict[str, str]]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> List[str]:
        """Generate completions for multiple prompts on one HF model/device."""
        self._enter_inference_mode()
        return generate_chat_batch(
            self.model,
            self.tokenizer,
            self.device,
            messages_list,
            self.config.max_input_tokens,
            max_new_tokens or self.config.max_new_tokens,
            temperature if temperature is not None else self.config.temperature,
            self.config.top_p,
        )

    def _generate_vllm_batch(
        self,
        messages_list: List[List[Dict[str, str]]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> List[str]:
        """Batched generation using vLLM — all prompts in one call."""
        from vllm import SamplingParams

        prompt_texts = [
            self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            for msgs in messages_list
        ]

        sampling_params = SamplingParams(
            temperature=temperature if temperature is not None else self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=max_new_tokens if max_new_tokens is not None else self.config.max_new_tokens,
        )

        outputs = self.llm.generate(
            prompt_texts,
            sampling_params=sampling_params,
            lora_request=self._get_vllm_lora_request(),
        )
        return [out.outputs[0].text for out in outputs]
    
    def _get_vllm_lora_request(self):
        from vllm.lora.request import LoRARequest
        if self._latest_lora_path is not None:
            return LoRARequest(
                f"live_adapter_{self._lora_request_id}",
                self._lora_request_id,
                self._latest_lora_path,
            )
        return None

    # ------------------------------------------------------------------ #
    #  Generation (inference mode)
    # ------------------------------------------------------------------ #

    def _enter_inference_mode(self):
        self.model.eval()

    def _enter_training_mode(self):
        self.model.train()

    # ------------------------------------------------------------------ #
    #  Training step
    # ------------------------------------------------------------------ #

    def train_batch(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Update once from the mean gradient of an interaction batch.

        Parameters
        ----------
        interactions
            Each item contains ``messages_before_response``,
            ``assistant_response``, and ``user_follow_up``.
        """
        if not interactions:
            raise ValueError("train_batch requires at least one interaction")

        self._enter_training_mode()
        t0 = time.time()

        prepared = []
        for interaction in interactions:
            messages = interaction["messages_before_response"]
            completion_text = interaction["assistant_response"].rstrip()
            if self.tokenizer.eos_token is not None:
                completion_text += self.tokenizer.eos_token
            completion_ids = self.tokenizer(
                [completion_text],
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"]

            x_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            xo_messages = build_hindsight_messages(
                messages,
                interaction["user_follow_up"],
                language=self.config.language,
            )
            xo_text = self.tokenizer.apply_chat_template(
                xo_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            # 沒 o / 有 o / 目前回覆
            prepared.append((x_text, xo_text, completion_ids))

        # Backpropagating each scaled loss immediately gives the gradient of
        # the mean loss without retaining every interaction's computation graph.
        batch_size = len(prepared)
        K = self.config.train_steps_per_batch
        for k in range(K):
            self.optimizer.zero_grad(set_to_none=True)
            losses = []
            batch_metrics = []
            for x_text, xo_text, completion_ids in prepared:
                if self.config.loss_mode == "simple_signal":
                    loss, current_metrics = simple_signal_loss(
                        self, x_text, xo_text, completion_ids, self.config.signal_clip,
                    )
                elif self.config.loss_mode == "full_distillation":
                    loss, current_metrics = full_distillation_loss(
                        self, x_text, xo_text, completion_ids,
                    )
                else:
                    raise ValueError(f"Unknown loss_mode: {self.config.loss_mode}")

                (loss / batch_size).backward()
                losses.append(loss.detach().item())
                batch_metrics.append(current_metrics)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.config.max_grad_norm,
            )
            self.optimizer.step()

        metrics = {}
        for key in batch_metrics[0]:
            if key == "token_debug":
                continue
            values = [ex[key] for ex in batch_metrics]
            if key == "completion_tokens":
                metrics[key] = sum(values) # 整個 batch 的 total tokens
            elif all(isinstance(value, (int, float)) for value in values):
                metrics[key] = sum(values) / batch_size
            else:
                metrics[key] = values[0]

        if "token_debug" in batch_metrics[0]:
            metrics["token_debug"] = [
                example_metrics["token_debug"]
                for example_metrics in batch_metrics
            ]

        self.step += 1
        metrics["step"] = self.step
        metrics["grad_norm"] = grad_norm.item()
        metrics["loss"] = sum(losses) / batch_size
        metrics["batch_size"] = batch_size
        metrics["train_steps_per_batch"] = K
        metrics["train_time_s"] = round(time.time() - t0, 2)

        # ---- sync LoRA adapter for vLLM ----
        if self.config.model_vllm:
            adapter_path = self.config.live_adapter_dir
            os.makedirs(adapter_path, exist_ok=True)
            self.model.save_pretrained(adapter_path)
            self.tokenizer.save_pretrained(adapter_path)
            self._latest_lora_path = adapter_path
            self._lora_request_id += 1

        # ---- auto-checkpoint ----
        if (
            self.config.save_checkpoints
            and self.step % self.config.checkpoint_every_n_steps == 0
        ):
            self.save_checkpoint()

        self._enter_inference_mode()
        return metrics

    # ------------------------------------------------------------------ #
    #  Checkpointing
    # ------------------------------------------------------------------ #

    def save_checkpoint(self, tag: Optional[str] = None) -> str:
        tag = tag or f"step_{self.step}"
        ckpt_path = os.path.join(self.config.checkpoint_dir, tag)
        os.makedirs(ckpt_path, exist_ok=True)

        # Merge LoRA into base on CPU so the checkpoint is a standalone HF model
        # that can be loaded directly via AutoModelForCausalLM.from_pretrained().
        # Deep-copy to CPU to avoid disrupting the live training state or GPU memory.
        if self.config.use_lora:
            merged = copy.deepcopy(self.model).cpu().merge_and_unload()
            merged.save_pretrained(ckpt_path)
            del merged
            torch.cuda.empty_cache()
        else:
            self.model.save_pretrained(ckpt_path)
        self.tokenizer.save_pretrained(ckpt_path)

        torch.save(
            {"optimizer_state_dict": self.optimizer.state_dict(), "step": self.step},
            os.path.join(ckpt_path, "training_state.pt"),
        )

        self._cleanup_old_checkpoints()
        print(f"[LIVE SDPO] Checkpoint saved → {ckpt_path}")
        return ckpt_path

    def _cleanup_old_checkpoints(self):
        if not os.path.exists(self.config.checkpoint_dir):
            return
        ckpts = sorted(
            [
                d for d in os.listdir(self.config.checkpoint_dir)
                if os.path.isdir(os.path.join(self.config.checkpoint_dir, d))
            ],
            key=lambda d: os.path.getmtime(
                os.path.join(self.config.checkpoint_dir, d)
            ),
        )
        while len(ckpts) > self.config.max_checkpoints:
            oldest = ckpts.pop(0)
            shutil.rmtree(os.path.join(self.config.checkpoint_dir, oldest))
