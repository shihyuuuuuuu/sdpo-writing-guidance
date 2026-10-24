import argparse


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate OnlineSDPOUpdater on dataset prompts")

    # Model
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--user_model", type=str, required=True)
    p.add_argument("--user_api_provider", type=str, required=True)
    p.add_argument("--judge_model", type=str, required=True)

    # Training
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--loss_mode", type=str, default="full_distillation",
                    choices=["simple_signal", "full_distillation"])
    p.add_argument("--temperature", type=float, required=True)
    p.add_argument("--train_n", type=int, required=True)
    p.add_argument("--eval_n", type=int, required=True)
    p.add_argument("--eval_every", type=int, required=True)
    p.add_argument("--batch_size", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--use_lora", action="store_true")

    p.add_argument("--max_input_tokens", type=int, required=True)
    p.add_argument("--max_new_tokens", type=int, required=True)
    p.add_argument("--user_max_input_tokens", type=int, required=True)
    p.add_argument("--user_max_new_tokens", type=int, required=True)

    # Data
    p.add_argument("--train_jsonl", type=str, required=True)
    p.add_argument("--val_jsonl", type=str, required=True)
    p.add_argument("--language", type=str, required=True)
    
    # GPU
    p.add_argument("--model_gpu", type=int, required=True)
    p.add_argument("--user_gpu", type=int, required=True)
    
    # VLLM
    p.add_argument("--model_vllm", action="store_true")
    p.add_argument("--user_vllm", action="store_true")
    p.add_argument("--vllm_gpu", type=int, required=True)
    p.add_argument("--vllm_gpu_mem", type=float, required=True)
    # deprecated
    p.add_argument("--user_vllm_tp", type=int, default=1,
                    help="Tensor parallel size for the user sim / judge vLLM instance (default: 1).")

    # Output
    p.add_argument("--save_checkpoints", action="store_true")
    p.add_argument("--checkpt_dir", type=str, required=True)
    p.add_argument("--eval_dir", type=str, required=True)
    p.add_argument("--debug_first_example", action="store_true",
                    help="Print full formatted inputs for the first simulator and judge call")

    return p.parse_args()
