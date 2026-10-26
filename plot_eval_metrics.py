#!/usr/bin/env python3
"""Plot online-SDPO evaluation metrics written to ``eval_metrics.jsonl``.

Examples:
    # From the sdpo_feedback directory: plot the newest evaluation run.
    python plot_eval_metrics.py

    # Plot one run directory or a specific JSONL file.
    python plot_eval_metrics.py --input eval/eval-online-sdpo-20261011-025202
    python plot_eval_metrics.py --input eval/.../eval_metrics.jsonl --output /tmp/metrics.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Plot online-SDPO eval_metrics.jsonl results."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=repo_root / "eval",
        help=(
            "An eval_metrics.jsonl file, an evaluation-run directory, or a directory "
            "containing runs. Defaults to sdpo_feedback/eval."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to eval_metrics.png beside the selected JSONL.",
    )
    return parser.parse_args()


def resolve_metrics_path(input_path: Path) -> Path:
    """Resolve a metrics file, choosing the newest one when given a directory."""
    input_path = input_path.expanduser()
    if input_path.is_file():
        if input_path.name != "eval_metrics.jsonl":
            raise ValueError(f"Expected an eval_metrics.jsonl file, got: {input_path}")
        return input_path
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {input_path}")

    direct_file = input_path / "eval_metrics.jsonl"
    if direct_file.is_file():
        return direct_file

    candidates = list(input_path.glob("*/eval_metrics.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"No eval_metrics.jsonl found under: {input_path}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_metrics(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc.msg}") from exc
            missing = {"step", "wins_a", "wins_b", "ties"} - row.keys()
            if missing:
                raise ValueError(
                    f"Missing {', '.join(sorted(missing))} at {path}:{line_number}"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"No metric records found in: {path}")
    return sorted(
        rows,
        key=lambda row: float(row.get("interactions_seen", row["step"])),
    )


def winrate_including_ties(row: dict[str, Any]) -> float:
    wins_a = float(row["wins_a"])
    wins_b = float(row["wins_b"])
    ties = float(row["ties"])
    total = wins_a + wins_b + ties
    return 100.0 * (wins_a + 0.5 * ties) / total if total else 0.0


def main() -> None:
    args = parse_args()
    try:
        metrics_path = resolve_metrics_path(args.input)
        rows = load_metrics(metrics_path)
        rows = rows[:10] # 先取前100
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "error: matplotlib is required. Install it with: python -m pip install matplotlib"
        ) from exc

    x_values = [float(row.get("interactions_seen", row["step"])) for row in rows]
    x_label = "Interactions seen" if all("interactions_seen" in row for row in rows) else "Training step"
    wins_a = [float(row["wins_a"]) for row in rows]
    wins_b = [float(row["wins_b"]) for row in rows]
    ties = [float(row["ties"]) for row in rows]
    winrates = [float(row.get("winrate_a", 0.0)) for row in rows]
    winrates_with_ties = [winrate_including_ties(row) for row in rows]
    standard_errors = [float(row.get("se", 0.0)) for row in rows]
    current_avg_tokens = [float(row.get("current_avg_tokens", 0.0)) for row in rows]
    baseline_avg_tokens = [float(row.get("baseline_avg_tokens", 0.0)) for row in rows]

    output_path = args.output or metrics_path.with_name("eval_metrics.png")
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    ax = axes[0]
    ax.plot(x_values, winrates, "-o", label="Win rate A (excluding ties)")
    ax.plot(x_values, winrates_with_ties, "-o", label="Win rate A (ties = 0.5)")
    lower = [max(0.0, value - error) for value, error in zip(winrates, standard_errors)]
    upper = [min(100.0, value + error) for value, error in zip(winrates, standard_errors)]
    ax.fill_between(x_values, lower, upper, alpha=0.15, label="±1 SE (excluding ties)")
    ax.set_ylabel("Win rate A (%)")
    ax.set_ylim(-2, 102)
    ax.set_title("Pairwise evaluation result")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    for values in (winrates, winrates_with_ties):
        for x, value in zip(x_values, values):
            ax.annotate(f"{value:.1f}", (x, value), xytext=(0, 7),
                        textcoords="offset points", ha="center", fontsize=8)

    ax = axes[1]
    ax.plot(x_values, current_avg_tokens, "-o", label="Current avg tokens")
    ax.plot(x_values, baseline_avg_tokens, "-o", label="Baseline avg tokens")
    ax.set_ylabel("Tokens")
    ax.set_title("Average generated completion length")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    ax.set_xlabel(x_label)
    for values in (current_avg_tokens, baseline_avg_tokens):
        for x, value in zip(x_values, values):
            ax.annotate(f"{value:.2f}", (x, value), xytext=(0, 7),
                        textcoords="offset points", ha="center", fontsize=8)

    fig.suptitle(metrics_path.parent.name, y=0.995, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Read {len(rows)} records from {metrics_path}")
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
