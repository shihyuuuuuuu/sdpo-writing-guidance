import json
import os
from typing import Dict

def append_jsonl(path: str, record: Dict) -> None:
    """Append one JSON record immediately so progress survives interruption."""
    with open(path, "a", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, allow_nan=True)
        f.write("\n")
        f.flush()

def save_summary(args, config, train_ds, eval_ds):
    context_n = train_ds[0].get("context_n") if len(train_ds) else None
    args_dict = vars(args)
    args_dict["train_n"] = len(train_ds)
    args_dict["eval_n"] = len(eval_ds)
    args_dict["context_n"] = context_n
    args_dict["lora_r"] = config.lora_r
    args_dict["distillation_topk"] = config.distillation_topk

    out_path = os.path.join(args.eval_dir, "manifest.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(args_dict, f, ensure_ascii=False, indent=2)
    print(f"Summary saved: {out_path}", flush=True)
