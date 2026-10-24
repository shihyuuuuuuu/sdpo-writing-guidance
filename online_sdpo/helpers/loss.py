from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def simple_signal_loss(
    updater,
    x_text: str,
    xo_text: str,
    completion_ids: torch.Tensor,
    signal_clip: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    loss = -(signal.detach() * log pi(y|x)),  length-normalised.
    Ported from offline_trainer.py:256-294.
    """
    # log pi(y | x, o) — detached, no grad.
    #
    # train_batch() puts the model in train mode before calling this loss.
    # Keep dropout disabled for the teacher pass; otherwise the difference
    # between teacher and student also contains two independent dropout
    # masks. Restore the previous mode before the student forward pass.
    was_training = updater.model.training
    updater.model.eval()
    try:
        with torch.no_grad():
            logps_xo, mask_xo, _ = _compute_token_logprobs(
                updater, xo_text, completion_ids, need_grad=False,
            )
    finally:
        updater.model.train(was_training)

    # log pi(y | x) — with grad
    logps_x, mask_x, _ = _compute_token_logprobs(
        updater, x_text, completion_ids, need_grad=True,
    )

    mask_f = mask_x.float()
    length = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)

    per_token_diff = (logps_xo - logps_x).detach()
    if signal_clip > 0:
        per_token_diff = per_token_diff.clamp(
            -signal_clip, signal_clip,
        )
    per_token_loss = -(per_token_diff * logps_x) * mask_f
    loss = (per_token_loss.sum(dim=1, keepdim=True) / length).mean()

    # Keep token ids as the canonical alignment key.  Decoded pieces are for
    # human-readable visualization only and may be byte/subword fragments.
    token_ids = completion_ids[0].detach().cpu().tolist()
    if updater.config.ignore_first_k > 0:
        token_ids = token_ids[updater.config.ignore_first_k:]
    token_strings = [
        updater.tokenizer.decode(
            [int(token_id)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        for token_id in token_ids
    ]
    token_debug = {
        "token_ids": token_ids,
        "token_text": token_strings,
        "completion_text": updater.tokenizer.decode(
            completion_ids[0].detach().cpu().tolist(),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
        "logp_x": logps_x[0].detach().cpu().tolist(),
        "logp_xo": logps_xo[0].detach().cpu().tolist(),
        "signal": per_token_diff[0].detach().cpu().tolist(),
    }

    total_tokens = mask_f.sum().clamp(min=1.0)
    metrics = {
        "signal_mean": (per_token_diff * mask_f).sum().item() / total_tokens.item(),
        "signal_std": (
            per_token_diff[mask_x.bool()].std().item()
            if mask_x.any() else 0.0
        ),
        "policy_logp_mean": (logps_x.detach() * mask_f).sum().item() / total_tokens.item(),
        "critic_logp_mean": (logps_xo * mask_f).sum().item() / total_tokens.item(),
        "completion_tokens": int(total_tokens.item()),
        "token_debug": token_debug,
    }

    return loss, metrics

def full_distillation_loss(
    updater,
    x_text: str,
    xo_text: str,
    completion_ids: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Reverse KL top-k distillation: KL(student || teacher).

    Student = pi(·|x), Teacher = pi(·|x,o).
    Top-k indices are taken from the student (on-policy), and the
    teacher is evaluated on those same indices.
    """
    # Teacher: pi(·|x,o) — no grad and deterministic dropout behavior.
    was_training = updater.model.training
    updater.model.eval()
    try:
        with torch.no_grad():
            _, _, logits_xo = _compute_token_logprobs(
                updater, xo_text, completion_ids,
                need_grad=False, need_logits=True,
            )
    finally:
        updater.model.train(was_training)

    # Student: pi(·|x) — with grad
    logps_x, mask_x, logits_x = _compute_token_logprobs(
        updater, x_text, completion_ids, need_grad=True, need_logits=True,
    )

    mask_f = mask_x.float()
    total_tokens = mask_f.sum().clamp(min=1.0)
    K = updater.config.distillation_topk

    # Full log-softmax over vocab
    student_log_probs = F.log_softmax(logits_x, dim=-1)   # (1, C', V)
    with torch.no_grad():
        teacher_log_probs = F.log_softmax(logits_xo, dim=-1)  # (1, C', V)

    # Student's top-k indices (on-policy support)
    student_topk, topk_idx = torch.topk(student_log_probs, k=K, dim=-1)  # (1, C', K)
    teacher_topk = torch.gather(teacher_log_probs, dim=-1, index=topk_idx)  # (1, C', K)

    # Tail handling
    if updater.config.distillation_add_tail:
        student_topk = _add_tail(student_topk)   # (1, C', K+1)
        teacher_topk = _add_tail(teacher_topk)   # (1, C', K+1)
    else:
        student_topk = _renorm_topk(student_topk)
        teacher_topk = _renorm_topk(teacher_topk)

    # Reverse KL: KL(student || teacher) = sum_x student(x) * (log student(x) - log teacher(x))
    # PyTorch kl_div(input, target) = target * (log target - input) when log_target=True
    # So KL(student || teacher) = kl_div(input=teacher, target=student, log_target=True)
    kl_per_token_per_k = F.kl_div(
        teacher_topk, student_topk, reduction="none", log_target=True,
    )  # (1, C', K+1) or (1, C', K)
    per_token_kl = kl_per_token_per_k.sum(dim=-1)  # (1, C')

    # Masked mean over tokens
    loss = (per_token_kl * mask_f).sum() / total_tokens

    metrics = {
        "kl_mean": (per_token_kl.detach() * mask_f).sum().item() / total_tokens.item(),
        "policy_logp_mean": (logps_x.detach() * mask_f).sum().item() / total_tokens.item(),
        "completion_tokens": int(total_tokens.item()),
        "loss_mode": "full_distillation",
    }
    return loss, metrics

def _compute_token_logprobs(
    updater,
    context_text: str,
    completion_ids: torch.Tensor,
    need_grad: bool = False,
    need_logits: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Per-token log-probs of *completion_ids* given a rendered context string.

    Returns
    -------
    per_token_logps : (1, C')  after ignore_first_k
    token_mask      : (1, C')  1 = real, 0 = pad
    logits_y        : (1, C', V) or None  (only when need_logits)
    """
    tokenizer = updater.tokenizer

    old_pad_side = tokenizer.padding_side
    old_trunc_side = tokenizer.truncation_side
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    enc = tokenizer(
        [context_text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=updater.config.max_input_tokens,
        add_special_tokens=False,
    ).to(updater.device)

    tokenizer.padding_side = old_pad_side
    tokenizer.truncation_side = old_trunc_side

    y_ids = completion_ids.to(updater.device)          # (1, C)
    # completion_ids is a single example (no padding), so all tokens
    # including the trailing EOS are real.  The old mask
    # (y_ids != pad_id) incorrectly zeroed EOS when pad_id == eos_id.
    y_mask = torch.ones_like(y_ids)                  # (1, C)

    input_ids = torch.cat([enc["input_ids"], y_ids], dim=1)
    attention_mask = torch.cat([enc["attention_mask"], y_mask], dim=1)

    ctx = torch.enable_grad() if need_grad else torch.no_grad()
    with ctx:
        outputs = updater.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits                     # (1, S, V)

    # Shift: logits[t] predicts token[t+1]
    logits_shifted = logits[:, :-1, :]
    labels = input_ids[:, 1:]

    seq_len_y = y_ids.size(1)
    logits_y = logits_shifted[:, -seq_len_y:, :]   # (1, C, V)
    labels_y = labels[:, -seq_len_y:]               # (1, C)
    labels_y = labels_y.masked_fill(y_mask == 0, -100)

    B, C, V = logits_y.shape
    nll = F.cross_entropy(
        logits_y.reshape(B * C, V),
        labels_y.reshape(B * C),
        reduction="none",
        ignore_index=-100,
    ).reshape(B, C)

    per_token_logps = -nll                          # (1, C)

    # ignore_first_k
    k = updater.config.ignore_first_k
    if k > 0 and seq_len_y > k:
        per_token_logps = per_token_logps[:, k:]
        y_mask = y_mask[:, k:]
        if need_logits:
            logits_y = logits_y[:, k:, :]

    return per_token_logps, y_mask, (logits_y if need_logits else None)

def _add_tail(log_probs: torch.Tensor) -> torch.Tensor:
    """Append log-prob of the tail (1 - sum(top_k_probs)) as a K+1th entry."""
    log_s = torch.logsumexp(log_probs, dim=-1, keepdim=True)
    log_s = torch.clamp(log_s, max=-1e-7)
    tail_log = torch.log(-torch.expm1(log_s))
    return torch.cat([log_probs, tail_log], dim=-1)

def _renorm_topk(log_probs: torch.Tensor) -> torch.Tensor:
    """Renormalize top-k log-probs to form a valid distribution."""
    logZ = torch.logsumexp(log_probs, dim=-1, keepdim=True)
    return log_probs - logZ
