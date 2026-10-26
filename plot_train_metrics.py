#!/usr/bin/env python3
"""Plot training diagnostics from train_steps.jsonl.

The three panels are:
  1. loss and gradient norm;
  2. hindsight signal statistics and policy/critic log-probability;
  3. average completion tokens and evaluation win rate, when eval metrics
     are available.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_path(repo_root: Path, explicit: Path | None, filename: str) -> Path:
    if explicit is not None:
        path = explicit.expanduser()
    else:
        candidates = list((repo_root / "eval").glob(f"*/{filename}"))
        if not candidates:
            raise FileNotFoundError(f"No {filename} found under {repo_root / 'eval'}")
        path = max(candidates, key=lambda item: item.stat().st_mtime)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    return path


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Plot train_steps.jsonl diagnostics")
    parser.add_argument("--train_steps", type=Path, default=None)
    parser.add_argument("--eval_metrics", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    train_path = resolve_path(repo_root, args.train_steps, "train_steps.jsonl")
    records = load_jsonl(train_path)
    if not records:
        raise SystemExit(f"No records found in {train_path}")
    eval_path = None
    if args.eval_metrics is not None:
        eval_path = resolve_path(repo_root, args.eval_metrics, "eval_metrics.jsonl")
    elif train_path.parent.joinpath("eval_metrics.jsonl").is_file():
        eval_path = train_path.parent / "eval_metrics.jsonl"
    eval_records = load_jsonl(eval_path) if eval_path else []

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("error: matplotlib is required") from exc

    steps = [float(row["step"]) for row in records]
    batch_sizes = [max(float(row.get("batch_size", 1)), 1.0) for row in records]
    avg_tokens = [float(row.get("completion_tokens", 0.0)) / size
                  for row, size in zip(records, batch_sizes)]

    fig, axes = plt.subplots(3, 1, figsize=(12, 13), sharex=True)

    ax = axes[0]
    ax.plot(steps, [float(row.get("loss", 0.0)) for row in records], "-o", label="Loss")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    grad_ax = ax.twinx()
    grad_ax.plot(steps, [float(row.get("grad_norm", 0.0)) for row in records],
                 "--o", color="tab:orange", label="Grad norm")
    grad_ax.set_ylabel("Grad norm")
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = grad_ax.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, loc="best")
    ax.set_title("Loss and gradient norm")

    ax = axes[1]
    for key, label, style in (
        ("signal_mean", "Signal mean", "-o"),
        ("signal_std", "Signal std", "--o"),
        ("policy_logp_mean", "Policy logp", "-o"),
        ("critic_logp_mean", "Critic logp", ":o"),
    ):
        if any(key in row for row in records):
            ax.plot(steps, [float(row.get(key, 0.0)) for row in records],
                    style, label=label)
    ax.set_ylabel("Value")
    ax.set_title("Signal statistics and policy/critic log-probability")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(steps, avg_tokens, "-o", color="tab:green", label="Avg completion tokens")
    ax.set_ylabel("Tokens", color="tab:green")
    ax.tick_params(axis="y", labelcolor="tab:green")
    ax.set_xlabel("Training step")
    ax.set_title("Completion length and evaluation win rate")
    ax.grid(alpha=0.3)
    if eval_records:
        eval_x = [float(row.get("step", 0.0)) for row in eval_records]
        # winrate_a is already stored as a percentage by compute_metrics().
        eval_y = [float(row.get("winrate_a", 0.0)) for row in eval_records]
        win_ax = ax.twinx()
        win_ax.plot(eval_x, eval_y, "-o", color="tab:purple", label="Eval win rate")
        win_ax.set_ylabel("Win rate (%)", color="tab:purple")
        win_ax.tick_params(axis="y", labelcolor="tab:purple")
        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = win_ax.get_legend_handles_labels()
        ax.legend(handles + handles2, labels + labels2, loc="best")
    else:
        ax.legend(loc="best")

    output = args.output or train_path.with_name("train_metrics.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle(train_path.parent.name, y=0.995, fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    print(f"Read {len(records)} training records from {train_path}")
    print(f"Saved plot to {output}")


if __name__ == "__main__":
    main()
