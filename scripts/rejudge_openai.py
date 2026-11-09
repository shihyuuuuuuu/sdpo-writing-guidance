#!/usr/bin/env python3
"""Re-judge selected saved evaluations with an OpenAI judge.

This never regenerates completions or trains a model.  It reuses the
``current`` and ``baseline`` texts saved in ``eval_completions.jsonl``, and
recovers the private reference continuation from the validation JSONL named in
the run's ``summary.json``.

Example:
    set -a; source .env; set +a
    uv run python scripts/rejudge_openai.py \
      --run_dir eval/eval-online-sdpo-20261107-103029 \
      --model gpt-5.6-terra

By default, every fourth saved evaluation plus the final evaluation is judged.
The result directory contains a compact CSV/JSON comparison and a JSONL with
the OpenAI AB/BA rationales, so interrupted runs can resume without repaying
for completed steps.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from judge.helpers import invert_pairwise_decisions, parse_judgement
from judge.openai_judge import OpenAIJudge
from online_sdpo.helpers.evaluator import compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-judge sampled saved eval completions with OpenAI."
    )
    parser.add_argument(
        "--run_dir", required=True, type=Path,
        help="Existing eval run directory containing summary.json and eval_completions.jsonl.",
    )
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument(
        "--stride", type=int, default=4,
        help="Judge every Nth saved eval record; the final record is always included (default: 4).",
    )
    parser.add_argument(
        "--max_evals", type=int, default=0,
        help="Optional cap on sampled eval records; 0 means no cap.",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_workers", type=int, default=16)
    parser.add_argument("--max_output_tokens", type=int, default=768)
    parser.add_argument("--api_key_env", default="OPENAI_API_KEY")
    parser.add_argument(
        "--output_dir", type=Path, default=None,
        help="Default: <run_dir>/openai_rejudge",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Validate inputs and print selected steps without making API calls.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON on {path}:{line_number}") from error
    return records


def resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_references(validation_path: Path) -> tuple[Dict[str, str], str]:
    """Map each saved context back to its p_next, rejecting ambiguous matches."""
    by_context: Dict[str, set[str]] = defaultdict(set)
    author = "Unknown author"
    for record in read_jsonl(validation_path):
        context = str(record["p_context"]).strip()
        reference = str(record["p_next"]).strip()
        by_context[context].add(reference)
        if author == "Unknown author":
            author = str(record.get("metadata", {}).get("author", author))

    ambiguous = [context for context, references in by_context.items() if len(references) != 1]
    if ambiguous:
        raise ValueError(
            f"{validation_path} has {len(ambiguous)} p_context values with different "
            "p_next references; refusing to guess a reference."
        )
    return {context: next(iter(references)) for context, references in by_context.items()}, author


def select_records(
    records: Sequence[Dict[str, Any]], stride: int, max_evals: int,
) -> List[Dict[str, Any]]:
    if stride < 1:
        raise ValueError("--stride must be at least 1")
    if max_evals < 0:
        raise ValueError("--max_evals must be non-negative")
    if not records:
        raise ValueError("eval_completions.jsonl is empty")

    selected = list(records[::stride])
    if records[-1] is not selected[-1]:
        selected.append(records[-1])
    if max_evals and len(selected) > max_evals:
        if max_evals == 1:
            selected = [selected[-1]]
        else:
            selected = selected[: max_evals - 1] + [selected[-1]]
    return selected


def metric_view(decisions: Sequence[int], seed: int) -> Dict[str, Any]:
    metrics = compute_metrics(list(decisions), bootstrap_seed=seed)
    n = metrics["n"]
    # Match plot_eval_metrics.py and the training logger: a tie contributes
    # half a win to each candidate.
    metrics["winrate_a_including_ties"] = (
        100.0 * (metrics.get("wins_a", 0) + 0.5 * metrics.get("ties", 0)) / n
        if n else float("nan")
    )
    return metrics


def normalized_decision(value: Any) -> int:
    """Treat missing or malformed historical labels as uncertain, never as a loss."""
    return value if value in (0, 1, -1) else -1


def tie_breakdown(
    decisions_ab: Sequence[int], decisions_ba: Sequence[int], final: Sequence[int],
) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    inverted_ba = invert_pairwise_decisions(list(decisions_ba))
    for ab, ba, resolved in zip(decisions_ab, inverted_ba, final):
        if resolved != -1:
            counts["resolved"] += 1
        elif ab == -1 and ba == -1:
            counts["both_uncertain"] += 1
        else:
            counts["ab_ba_conflict"] += 1
    return dict(counts)


def agreement_view(qwen: Sequence[int], openai: Sequence[int]) -> Dict[str, Any]:
    labels = {0: "current", 1: "baseline", -1: "tie"}
    matrix: Counter[str] = Counter()
    exact = 0
    for qwen_decision, openai_decision in zip(qwen, openai):
        matrix[f"qwen_{labels[qwen_decision]}__openai_{labels[openai_decision]}"] += 1
        exact += qwen_decision == openai_decision
    return {"exact_agreement": exact, "matrix": dict(matrix)}


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, allow_nan=True)
        handle.write("\n")
        handle.flush()


def existing_steps(path: Path, model: str) -> set[int]:
    if not path.exists():
        return set()
    steps: set[int] = set()
    for record in read_jsonl(path):
        previous_model = record.get("openai_model")
        if previous_model != model:
            raise ValueError(
                f"{path} already contains results from {previous_model!r}, not {model!r}. "
                "Use a different --output_dir to keep judge models separate."
            )
        steps.add(int(record["step"]))
    return steps


def reparse_saved_results(path: Path, seed: int) -> None:
    """Repair parsed labels from saved raw outputs without making API calls."""
    if not path.exists():
        return

    repaired: List[Dict[str, Any]] = []
    changed = 0
    for result in read_jsonl(path):
        old_metrics = result.get("openai_metrics")
        openai_decisions: List[int] = []
        openai_ab: List[int] = []
        openai_ba: List[int] = []
        for pair in result.get("pairs", []):
            openai = pair["openai"]
            ab = parse_judgement(openai.get("ab_raw", ""))
            ba = parse_judgement(openai.get("ba_raw", ""))
            final = ab if ab == {0: 1, 1: 0}.get(ba, -1) else -1
            openai_ab.append(ab)
            openai_ba.append(ba)
            openai_decisions.append(final)
            old = (openai.get("ab_decision"), openai.get("ba_decision"), openai.get("final_decision"))
            new = (ab, ba, final)
            if old != new:
                changed += 1
            openai.update({
                "ab_decision": ab,
                "ba_decision": ba,
                "final_decision": final,
            })

        result["openai_metrics"] = metric_view(openai_decisions, seed)
        if old_metrics != result["openai_metrics"]:
            changed += 1
        result["openai_tie_breakdown"] = tie_breakdown(
            openai_ab, openai_ba, openai_decisions,
        )
        qwen_decisions = [
            normalized_decision(pair.get("qwen", {}).get("final_decision"))
            for pair in result.get("pairs", [])
        ]
        result["agreement"] = agreement_view(qwen_decisions, openai_decisions)
        repaired.append(result)

    if changed:
        with path.open("w", encoding="utf-8") as handle:
            for result in repaired:
                json.dump(result, handle, ensure_ascii=False, allow_nan=True)
                handle.write("\n")
        print(
            f"[OpenAI rejudge] repaired {changed} labels from saved raw outputs; "
            "no API calls made for this repair",
            flush=True,
        )


def write_comparison_files(
    output_dir: Path,
    results_path: Path,
    metadata: Mapping[str, Any],
) -> None:
    results = read_jsonl(results_path) if results_path.exists() else []
    results.sort(key=lambda record: record["step"])
    summary = dict(metadata)
    summary["completed_steps"] = [record["step"] for record in results]
    summary["results"] = [
        {
            "step": record["step"],
            "sample_n": record["sample_n"],
            "qwen": record["qwen_metrics"],
            "openai": record["openai_metrics"],
            "agreement": record["agreement"],
        }
        for record in results
    ]
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=True)

    columns = [
        "step", "sample_n",
        "qwen_wins", "qwen_losses", "qwen_ties", "qwen_coverage",
        "qwen_winrate_decisive", "qwen_winrate_including_ties",
        "openai_wins", "openai_losses", "openai_ties", "openai_coverage",
        "openai_winrate_decisive", "openai_winrate_including_ties",
        "exact_agreement",
    ]
    with (output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in results:
            qwen = record["qwen_metrics"]
            openai = record["openai_metrics"]
            writer.writerow({
                "step": record["step"],
                "sample_n": record["sample_n"],
                "qwen_wins": qwen["wins_a"],
                "qwen_losses": qwen["wins_b"],
                "qwen_ties": qwen["ties"],
                "qwen_coverage": qwen["coverage"],
                "qwen_winrate_decisive": qwen["winrate_a"],
                "qwen_winrate_including_ties": qwen["winrate_a_including_ties"],
                "openai_wins": openai["wins_a"],
                "openai_losses": openai["wins_b"],
                "openai_ties": openai["ties"],
                "openai_coverage": openai["coverage"],
                "openai_winrate_decisive": openai["winrate_a"],
                "openai_winrate_including_ties": openai["winrate_a_including_ties"],
                "exact_agreement": record["agreement"]["exact_agreement"],
            })


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    summary_path = run_dir / "summary.json"
    completions_path = run_dir / "eval_completions.jsonl"
    if not summary_path.is_file() or not completions_path.is_file():
        raise FileNotFoundError(
            "--run_dir must contain summary.json and eval_completions.jsonl: "
            f"{run_dir}"
        )

    run_summary = load_json(summary_path)
    validation_path = resolve_repo_path(run_summary["val_jsonl"])
    references, author = load_references(validation_path)
    records = select_records(read_jsonl(completions_path), args.stride, args.max_evals)
    steps = [int(record["step"]) for record in records]
    print(f"Selected {len(records)} evals at steps: {steps}", flush=True)

    for record in records:
        missing = [
            index for index, completion in enumerate(record.get("completions", []))
            if str(completion.get("prompt", "")).strip() not in references
        ]
        if missing:
            raise ValueError(
                f"Step {record.get('step')} has {len(missing)} prompt(s) that cannot be "
                f"matched to {validation_path}; first index: {missing[0]}"
            )
        if not record.get("completions"):
            raise ValueError(f"Step {record.get('step')} has no saved completions")

    if args.dry_run:
        print("Dry run complete; no OpenAI requests were made.", flush=True)
        return

    output_dir = (args.output_dir or run_dir / "openai_rejudge").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    # Existing results may have been parsed before the same-line decision fix.
    # Repair them locally before deciding which steps still need API calls.
    reparse_saved_results(results_path, int(run_summary.get("seed", 0)))
    finished_steps = existing_steps(results_path, args.model)
    seed = int(run_summary.get("seed", 0))
    judge = OpenAIJudge(
        model=args.model,
        max_workers=args.max_workers,
        api_key_env=args.api_key_env,
        author=author,
        language=run_summary["language"],
    )
    judge.max_output_tokens = args.max_output_tokens

    metadata = {
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_dir": str(run_dir),
        "source_eval_completions": str(completions_path),
        "validation_jsonl": str(validation_path),
        "openai_model": args.model,
        "author": author,
        "language": run_summary["language"],
        "sampling": {"stride": args.stride, "max_evals": args.max_evals, "selected_steps": steps},
        "notes": (
            "Qwen and OpenAI metrics are calculated over the same saved completion "
            "pairs. Both use the same strict AB/BA agreement rule."
        ),
    }

    for position, record in enumerate(records, start=1):
        step = int(record["step"])
        if step in finished_steps:
            print(f"[OpenAI rejudge] step {step}: already saved; skipping", flush=True)
            continue

        completions = record["completions"]
        prompts = [str(item["prompt"]).strip() for item in completions]
        current = [str(item["current"]) for item in completions]
        baseline = [str(item["baseline"]) for item in completions]
        references_for_step = [references[prompt] for prompt in prompts]
        qwen_ab = [
            normalized_decision(item["judge"].get("ab_decision"))
            for item in completions
        ]
        qwen_ba = [
            normalized_decision(item["judge"].get("ba_decision"))
            for item in completions
        ]
        qwen_final = [
            normalized_decision(item["judge"].get("final_decision"))
            for item in completions
        ]

        print(
            f"[OpenAI rejudge] step {step} ({position}/{len(records)}), "
            f"{len(completions)} pairs × AB/BA...",
            flush=True,
        )
        openai_final = judge.choose_batch_generated(
            prompts=prompts,
            references=references_for_step,
            completions_a=current,
            completions_b=baseline,
            batch_size=args.batch_size,
        )
        pair_records = [
            {
                "index": index,
                "qwen": completion["judge"],
                "openai": {
                    "ab_raw": judge.last_raw_outputs_ab[index],
                    "ba_raw": judge.last_raw_outputs_ba[index],
                    "ab_decision": judge.last_decisions_ab[index],
                    "ba_decision": judge.last_decisions_ba[index],
                    "final_decision": openai_final[index],
                },
            }
            for index, completion in enumerate(completions)
        ]
        result = {
            "step": step,
            "openai_model": args.model,
            "sample_n": len(completions),
            "qwen_metrics": metric_view(qwen_final, seed),
            "openai_metrics": metric_view(openai_final, seed),
            "qwen_tie_breakdown": tie_breakdown(qwen_ab, qwen_ba, qwen_final),
            "openai_tie_breakdown": tie_breakdown(
                judge.last_decisions_ab, judge.last_decisions_ba, openai_final,
            ),
            "agreement": agreement_view(qwen_final, openai_final),
            "pairs": pair_records,
        }
        append_jsonl(results_path, result)
        write_comparison_files(output_dir, results_path, metadata)

    write_comparison_files(output_dir, results_path, metadata)
    print(f"Saved comparison to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
