"""
Automated evaluation of OnlineSDPOUpdater on dataset prompts.

Simulates the live chat flow: for each training prompt, generate a response
and get simulated user feedback. The collected interactions are averaged into
batched SDPO updates. Periodically evaluates the current model against the
pre-training baseline using a judge.

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
from .helpers.logging import print_first_example, print_train_step
from .helpers.model import load_model_and_tokenizer
from .helpers.parse_args import parse_args
from .helpers.record import append_jsonl, save_summary
from .online_sdpo_updater_config import OnlineSDPOConfig
from .online_sdpo_updater import OnlineSDPOUpdater
from prompts.prompt import build_assistant_messages, build_phase1_messages
from user_simulator.user_simulator import (
    LocalStyleUserSimulator,
    LocalUserSimulator,
    OpenAIStyleUserSimulator,
    OpenAIUserSimulator,
)


def prepare_ds(ds, args, n):
    return ds.shuffle(seed=args.seed).select(range(min(n, len(ds))))


def main():
    args = parse_args()

    train_steps_path = os.path.join(args.eval_dir, "train_steps.jsonl")

    # ── 1. Build config ──
    args_dict = vars(args)
    keys = {
        "model", "use_lora", "loss_mode", "model_vllm", "model_gpu",
        "vllm_gpu", "vllm_gpu_mem", "max_input_tokens", "max_new_tokens",
        "temperature", "batch_size", "save_checkpoints",
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
        style_mode=args.style_mode,
    )

    save_summary(args, config, train_ds, eval_ds)

    # ── 4. Generate baseline from main model (before training) ──
    print(f"[EVAL] Generating baseline completions ({len(evaluator.chats)} prompts)...", flush=True)
    t0 = time.time()
    evaluator.base_completions = evaluator.generate_completions(updater)
    print(f"[EVAL] Baseline generation took {time.time() - t0:.1f}s  avg_len={evaluator.baseline_avg_len:.0f} chars", flush=True)

    # ── 5. Initialize user simulator and judge ──
    user_llm = None  # shared vLLM instance for user sim + judge
    # Judge model 目前都是跟著 user model 走，會是一樣的模型
    if args.user_api_provider == "local":
        if args.user_vllm:
            ...
        else:
            print(f"[EVAL] Loading local user simulator: {args.user_model}", flush=True)
            user_hf, user_tok = load_model_and_tokenizer(args.user_model, args.user_gpu)
            user_hf.eval()
            simulator = LocalUserSimulator(
                model=user_hf, tokenizer=user_tok,
                device=next(user_hf.parameters()).device,
                max_input_tokens=args.user_max_input_tokens,
                max_new_tokens=args.user_max_new_tokens,
                language=args.language,
            ) if not args.style_mode else LocalStyleUserSimulator(
                model=user_hf, tokenizer=user_tok,
                device=next(user_hf.parameters()).device,
                max_input_tokens=args.user_max_input_tokens,
                max_new_tokens=args.user_max_new_tokens,
            )

            print(f"[EVAL] Using LocalJudge (shared with user sim)", flush=True)
            judge = local_judge.LocalJudge(
                model=user_hf,
                tokenizer=user_tok,
                device=next(user_hf.parameters()).device,
                author=author,
                language=args.language,
                style_mode=args.style_mode,
            )
    elif args.user_api_provider == "openai":
        print("[EVAL] Using OpenAI user simulator.", flush=True)
        simulator = OpenAIUserSimulator(
            model=args.user_model,
            max_new_tokens=args.user_max_new_tokens,
            language=args.language,
        ) if not args.style_mode else OpenAIStyleUserSimulator(
            model=args.user_model,
            max_new_tokens=args.user_max_new_tokens,
        )

        print(f"[EVAL] Using OpenAIJudge ({args.judge_model})", flush=True)
        judge = openai_judge.OpenAIJudge(
            model=args.judge_model,
            author=author,
            language=args.language,
            style_mode=args.style_mode,
        )
    else:
        raise ValueError(f"Invalid user API provider: {args.user_api_provider!r}")

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

        if args.style_mode:
            p_contexts = [ex["prompt"].strip() for ex in batch_examples]
            p_nexts = [None for _ in batch_examples]
            messages_list = [build_assistant_messages(prompt) for prompt in p_contexts]
        else:
            p_contexts = [ex["p_context"].strip() for ex in batch_examples]
            p_nexts = [ex["p_next"].strip() for ex in batch_examples]
            messages_list = [build_phase1_messages(p_context, args.language) for p_context in p_contexts]

        # Generate assistant responses for the whole batch.
        t0 = time.time()
        responses = updater.generate_responses_batch(messages_list)
        gen_time = time.time() - t0

        # Get simulated user feedback for the whole batch.
        t1 = time.time()
        if args.style_mode:
            feedbacks = simulator.generate_feedbacks(p_contexts, responses)
        else:
            feedbacks = simulator.generate_feedbacks(p_contexts, responses, p_nexts)
        sim_time = time.time() - t1

        if batch_start == 0 and args.debug_first_example:
            chat = (
                simulator._build_prompt_text(p_contexts[0], responses[0])
                if args.style_mode else
                simulator._build_prompt_text(p_contexts[0], responses[0], p_nexts[0])
            )
            print_first_example(chat, feedbacks[0])

        for feedback in feedbacks:
            print(f"feedback: {feedback[:]!r}", flush=True)

        batch_interactions = [
            {
                "messages_before_response": messages,
                "assistant_response": response,
                "user_follow_up": feedback,
                "p_context": p_context,
                "p_next": p_next,
                "dataset_index": dataset_index,
            }
            for messages, response, feedback, p_context, p_next, dataset_index in zip(
                messages_list, responses, feedbacks, p_contexts, p_nexts, batch_indices,
            )
        ]

        metrics = updater.train_batch(batch_interactions)
        for i in batch_interactions:
            i.pop("messages_before_response", None)

        metrics["gen_time_s"] = round(gen_time, 2)
        metrics["sim_time_s"] = round(sim_time, 2)
        metrics["interactions_seen"] = batch_end
        metrics["interactions"] = batch_interactions
        
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
