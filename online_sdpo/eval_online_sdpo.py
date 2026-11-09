"""
Automated evaluation of OnlineSDPOUpdater on dataset prompts.

For each training prompt, generate a response and use the dataset's actual
next paragraph as the teacher-side reference context. The collected
interactions are averaged into batched SDPO updates. Periodically evaluates
the current model against the pre-training baseline using a judge.

Usage:
    python -m online_sdpo.eval_online_sdpo \
        --train_jsonl data/helpsteer_prompts/train.jsonl \
        --val_jsonl data/helpsteer_prompts/validation.jsonl \
        --style concise_casual_beginner
"""
import os
import time

from datasets import load_dataset

from judge import local_judge, openai_judge
from .helpers.evaluator import Evaluator
from .helpers.logging import log_token_debug, print_first_example, print_train_step
from .helpers.model import load_model_and_tokenizer
from .helpers.parse_args import parse_args
from .helpers.record import append_jsonl, save_summary
from .online_sdpo_updater_config import OnlineSDPOConfig
from .online_sdpo_updater import OnlineSDPOUpdater
from prompts.prompt import build_phase1_messages, build_phase1_teacher_messages


def prepare_ds(ds, args, n):
    return ds.shuffle(seed=args.seed).select(range(min(n, len(ds))))


def main():
    args = parse_args()

    train_steps_path = os.path.join(args.eval_dir, "train_steps.jsonl")
    token_debug_path = os.path.join(args.eval_dir, "token_debug.jsonl")

    # ── 1. Build config ──
    args_dict = vars(args)
    keys = {
        "model", "use_lora", "loss_mode", "model_vllm", "model_gpu",
        "vllm_gpu", "vllm_gpu_mem", "max_input_tokens", "max_new_tokens",
        "temperature", "batch_size", "save_checkpoints", "language",
        "max_grad_norm",
        "max_decay_steps", "scheduler_type", "warmup_steps", "min_lr_ratio",
    }
    config = OnlineSDPOConfig(
        **{k: v for k, v in args_dict.items() if k in keys},
        learning_rate=args.lr,
        checkpoint_dir=os.path.join(args.checkpt_dir, "checkpoints"),
        live_adapter_dir=os.path.join(args.eval_dir, "live_adapter"),
    )
    print("[EVAL] Initializing OnlineSDPOUpdater...", flush=True)

    # ── 2. Load datasets ──
    dsd = load_dataset("json", data_files={"train": args.train_jsonl, "validation": args.val_jsonl})
    train_ds = dsd["train"]
    eval_ds = dsd["validation"]
    train_ds = prepare_ds(train_ds, args, args.train_n)
    eval_ds = prepare_ds(eval_ds, args, args.eval_n)
    author = train_ds[0].get("metadata", {}).get("author", "unknown")

    # ── 3. Initiate updater and evaluator ──
    updater = OnlineSDPOUpdater(config)
    evaluator = Evaluator(
        eval_ds=eval_ds,
        lang=args.language,
        max_new_tokens=args.max_new_tokens,
        temperature=0,
        eval_dir=args.eval_dir,
    )

    save_summary(args, config, train_ds, eval_ds)

    # ── 4. Generate baseline from main model (before training) ──
    print(f"[EVAL] Generating baseline completions ({len(evaluator.chats)} prompts)...", flush=True)
    t0 = time.time()
    evaluator.base_completions = evaluator.generate_completions(updater)
    print(f"[EVAL] Baseline generation took {time.time() - t0:.1f}s  avg_len={evaluator.baseline_avg_len:.0f} chars", flush=True)

    # ── 5. Initialize judge only ──
    if args.judge_api_provider == "local":
        print(f"[EVAL] Loading local judge: {args.judge_model}", flush=True)
        judge_hf, judge_tok = load_model_and_tokenizer(args.judge_model, args.judge_gpu)
        judge_hf.eval()
        judge = local_judge.LocalJudge(
            model=judge_hf,
            tokenizer=judge_tok,
            device=next(judge_hf.parameters()).device,
            author=author,
            language=args.language,
        )
    elif args.judge_api_provider == "openai":
        judge = openai_judge.OpenAIJudge(
            model=args.judge_model,
            author=author,
            language=args.language,
        )
    else:
        raise ValueError(f"Invalid judge API provider: {args.judge_api_provider!r}")

    # ── 6. Training loop ──
    print(
        f"\n[EVAL] Starting training loop ({len(train_ds)} interactions, "
        f"batch_size={args.batch_size})...\n",
        flush=True,
    )

    for batch_start in range(0, len(train_ds), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(train_ds))
        batch_examples = [train_ds[i] for i in range(batch_start, batch_end)]
        batch_indices = list(range(batch_start, batch_end))

        p_contexts = [ex["p_context"].strip() for ex in batch_examples]
        p_nexts = [ex["p_next"].strip() for ex in batch_examples]
        u_messages_list = [build_phase1_messages(p_context, args.language) for p_context in p_contexts]

        # Generate assistant responses for the whole batch.
        print("Generating assistant responses......")
        t0 = time.time()
        responses = updater.generate_responses_batch(u_messages_list)
        gen_time = time.time() - t0

        if batch_start == 0 and args.debug_first_example:
            teacher_messages = build_phase1_teacher_messages(
                u_messages_list[0], p_nexts[0], args.language,
            )
            print_first_example(teacher_messages, p_nexts[0])

        batch_interactions = [
            {
                "messages_before_response": messages,
                "assistant_response": response,
                "p_next": p_next,
                "p_context": p_context,
                "dataset_index": dataset_index,
            }
            for messages, response, p_context, p_next, dataset_index in zip(
                u_messages_list, responses, p_contexts, p_nexts, batch_indices,
            )
        ]

        metrics = updater.train_batch(batch_interactions)
        token_debug = metrics.pop("token_debug", None)
        for i in batch_interactions:
            i.pop("messages_before_response", None)

        metrics["gen_time_s"] = round(gen_time, 2)
        metrics["interactions_seen"] = batch_end
        metrics["interactions"] = batch_interactions

        if token_debug is not None:
            log_token_debug(token_debug_path, metrics["step"], batch_end, batch_interactions, token_debug)
        
        append_jsonl(train_steps_path, metrics)
        print_train_step(metrics)

        # Periodic evaluation, counted in optimizer updates.
        if args.eval_every > 0 and metrics["step"] % args.eval_every == 0:
            evaluator.evaluate(
                metrics["step"], batch_end, updater, args.seed, judge,
            )

    # ── 7. Final evaluation ──
    final_step = updater.step
    if args.eval_every > 0 and final_step % args.eval_every != 0:
        evaluator.evaluate(
            final_step, len(train_ds), updater, args.seed, judge,
        )

    # ── 8. Save checkpoint ──
    if args.save_checkpoints:
        updater.save_checkpoint()

if __name__ == "__main__":
    main()
