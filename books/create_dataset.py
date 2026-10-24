#!/usr/bin/env python3
"""Create fixed-context Phase 1 continuation datasets.

The input files are book JSONL files under ``books/paragraphs/<author>``.
Each input row must contain a ``paragraph`` field and may contain ``metadata``.

For a fixed context size k, a row at position i in a chapter becomes:

    p_context = rows[i-k:i]
    p_next    = rows[i]

Windows are created before shuffling.  Context never crosses a chapter
boundary.  The output is intentionally model-agnostic; token-length filtering
belongs in eval_online_sdpo.py because it depends on the model tokenizer.

example:
python sdpo_feedback/books/create_dataset.py \
  --train "射鵰英雄傳一" "射鵰英雄傳二" \
  --val "射鵰英雄傳三" \
  --context_n 3 \
  --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fixed-context p_context/p_next JSONL files from book JSONL files."
    )
    parser.add_argument(
        "--train",
        nargs="+",
        required=True,
        help="One or more book names to use for training (without .jsonl).",
    )
    parser.add_argument(
        "--val",
        nargs="+",
        required=True,
        help="One or more book names to use for validation (without .jsonl).",
    )
    parser.add_argument(
        "--context_n",
        type=int,
        required=True,
        help="Fixed number of preceding paragraphs in p_context.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used to deterministically shuffle the training windows.",
    )
    parser.add_argument(
        "--no_shuffle_train",
        action="store_true",
        help="Keep training windows in source order instead of shuffling them.",
    )
    return parser.parse_args()


def resolve_book_path(book: str, paragraphs_dir: Path) -> Path:
    """Find one book by name below ``paragraphs/<author>``."""
    book_name = Path(book).name.removesuffix(".jsonl")
    matches = list(paragraphs_dir.rglob(f"{book_name}.jsonl"))
    if not matches:
        raise FileNotFoundError(
            f"Book not found: {book!r}. Tried {paragraphs_dir}/**/{book_name}.jsonl"
        )
    if len(matches) > 1:
        files = "\n".join(f"  {path}" for path in matches)
        raise ValueError(f"Found multiple books with the same name:\n{files}")
    return matches[0]


def read_book(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                example = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            paragraph = example.get("paragraph")
            if not isinstance(paragraph, str) or not paragraph.strip():
                raise ValueError(
                    f"Missing non-empty string 'paragraph' in {path}:{line_number}"
                )
            metadata = example.get("metadata")
            if metadata is not None and not isinstance(metadata, dict):
                raise ValueError(f"'metadata' must be an object in {path}:{line_number}")
            rows.append(
                {
                    "paragraph": paragraph.strip(),
                    "metadata": dict(metadata or {}),
                    "source_line": line_number,
                }
            )
    if not rows:
        raise ValueError(f"Book is empty: {path}")
    return rows


def _chapter_key(row: Dict[str, Any]) -> str:
    # Missing chapter metadata means the complete book is one sequence.
    return str(row["metadata"].get("chapter", "__whole_book__"))


def build_windows(book_path: Path, context_n: int) -> List[Dict[str, Any]]:
    rows = read_book(book_path)
    book_windows: List[Dict[str, Any]] = []

    # Consecutive rows belonging to the same chapter form one sequence.  This
    # preserves source order and prevents a window from crossing a chapter.
    start = 0
    while start < len(rows):
        chapter = _chapter_key(rows[start])
        end = start + 1
        while end < len(rows) and _chapter_key(rows[end]) == chapter:
            end += 1

        chapter_rows = rows[start:end]
        for local_target_index in range(context_n, len(chapter_rows)):
            context_rows = chapter_rows[local_target_index - context_n : local_target_index]
            target = chapter_rows[local_target_index]
            target_metadata = dict(target["metadata"])
            p_context = "\n\n".join(row["paragraph"] for row in context_rows)
            p_next = target["paragraph"]
            context_token_count = sum(
                row["metadata"].get("token_count", 0) for row in context_rows
            )
            next_token_count = target_metadata.get("token_count", 0)
            target_metadata.pop("char_count", None)
            target_metadata.pop("token_count", None)
            target_metadata.update(
                {
                    "p_context_char_count": len(p_context),
                    "p_context_token_count": context_token_count,
                    "p_next_char_count": len(p_next),
                    "p_next_token_count": next_token_count,
                }
            )
            book_windows.append(
                {
                    "source_file": book_path.name,
                    "context_n": context_n,
                    "p_context": p_context,
                    "p_next": p_next,
                    "metadata": target_metadata,
                }
            )
        start = end

    return book_windows


def build_split(book_args: Sequence[str], books_dir: Path, context_n: int) -> List[Dict[str, Any]]:
    windows: List[Dict[str, Any]] = []
    seen_paths = set()
    for book_arg in book_args:
        path = resolve_book_path(book_arg, books_dir)
        resolved = path.resolve()
        if resolved in seen_paths:
            raise ValueError(f"Book was listed more than once: {path}")
        seen_paths.add(resolved)
        book_windows = build_windows(path, context_n)
        if not book_windows:
            raise ValueError(
                f"Book {path} produced no windows with context_n={context_n}."
            )
        windows.extend(book_windows)
    return windows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.context_n < 1:
        raise ValueError("--context_n must be >= 1")

    books_dir = Path(__file__).resolve().parent
    paragraphs_dir = books_dir / "paragraphs"
    train_paths = [resolve_book_path(book, paragraphs_dir).resolve() for book in args.train]
    validation_paths = [
        resolve_book_path(book, paragraphs_dir).resolve()
        for book in args.val
    ]
    overlap = set(train_paths) & set(validation_paths)
    if overlap:
        raise ValueError(
            "The same book appears in both train and validation: "
            + ", ".join(str(path) for path in sorted(overlap))
        )

    authors = {path.parent.name for path in train_paths + validation_paths}
    if len(authors) != 1:
        raise ValueError(
            "All train and validation books must belong to the same author; "
            f"found: {', '.join(sorted(authors))}"
        )
    author = next(iter(authors))

    train_rows = build_split(args.train, paragraphs_dir, args.context_n)
    validation_rows = build_split(args.val, paragraphs_dir, args.context_n)
    if not args.no_shuffle_train:
        random.Random(args.seed).shuffle(train_rows)

    output_dir = books_dir.parent / "data" / author / f"context_{args.context_n}"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "validation.jsonl", validation_rows)

    manifest = {
        "context_n": args.context_n,
        "seed": args.seed,
        "author": author,
        "train_books": [path.stem for path in train_paths],
        "validation_books": [path.stem for path in validation_paths],
        "train_examples": len(train_rows),
        "validation_examples": len(validation_rows),
        "train_shuffled": not args.no_shuffle_train,
        "source_format": "one JSON object per line with paragraph and optional metadata",
    }
    manifest["output_dir"] = str(output_dir)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
