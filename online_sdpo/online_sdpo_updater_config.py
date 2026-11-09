from dataclasses import dataclass, field
from typing import Optional


@dataclass(kw_only=True)
class OnlineSDPOConfig:
    """ Model """
    model: str
    attn_implementation: str = "sdpa"

    """ LoRA """
    use_lora: bool
    lora_r: int = 64
    lora_alpha: int = 128
    lora_target_modules: list[str] = field(
        default_factory=lambda:
        [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    lora_dropout: float = 0.0

    """ SDPO signal """
    loss_mode: str  # "simple_signal" | "full_distillation" | "opsd_forward_kl_clip"
    signal_clip: float = 0.0  # 0 = no clipping
    pointwise_clip: float = 0.05  # OPSD per-vocabulary forward-KL clip
    ignore_first_k: int = 0
    distillation_topk: int = 20  # top-k logits for full_distillation
    distillation_add_tail: bool = True  # True = add tail bucket, False = renormalize top-k
    fixed_teacher: bool = True  # use the initial/base policy for teacher scoring

    """ GPU / VLLM """
    model_vllm: bool
    model_gpu: str
    vllm_gpu: str
    vllm_gpu_mem: float

    """ Training """
    batch_size: int  # Interactions per optimizer update
    train_steps_per_batch: int = 1 # K optimizer updates per interaction batch

    """  Optimizer  """
    learning_rate: float
    max_decay_steps: int
    scheduler_type: str = "cosine"  # "linear" or "cosine"
    warmup_steps: int = 5
    min_lr_ratio: float = 0.1
    adam_epsilon: float = 1e-6
    max_grad_norm: float = 0.3
    weight_decay: float = 0.0

    """ Generation """
    max_input_tokens: int
    max_new_tokens: int
    temperature: float
    top_p: float = 0.8
    language: str = "en"

    """ Checkpointing """
    checkpoint_dir: str
    live_adapter_dir: Optional[str] = None
    save_checkpoints: bool = False
    checkpoint_every_n_steps: int = 1
    max_checkpoints: int = 3
