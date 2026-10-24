#!/usr/bin/env python3
"""Split one converted book into 150--250-token JSONL records.

The input format is expected to contain chapter markers such as::

    Chapter: 第一章

Each following non-empty line is treated as one source paragraph until the
next ``Chapter:`` marker.  Sentence endings, including closing quotation
marks, are preferred as split points.  The output is written to
``paragraphs/<author>``.  Process one or more input books per invocation.  Token counts
use the Qwen3 tokenizer selected by ``--tokenizer``.
"""

# Usage: $ uv run python split_paragraphs.py "The Works of Edgar Allan Poe Volume 1" "The Works of Edgar Allan Poe Volume 2" "The Fall of the House of Usher" "At the mountains of madness" "天涯明月刀" "射鵰英雄傳一" --tokenizer="Qwen/Qwen3.5-9B"

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, Iterator

CHAPTER_RE = re.compile(r"^\s*Chapter:\s*(.*?)\s*$", re.IGNORECASE)
# Closing punctuation that may follow a sentence-ending mark.  These are
# consumed together so that a sentence such as ``He said, "Stop."`` is not
# split between the period and the closing quote.
CLOSING_MARKS = r"」』】》）〕〉”’\"'\)\]\}»›﹂﹁﹃﹄"
SENTENCE_END_RE = re.compile(
    rf"(?:[。！？!?](?:[{CLOSING_MARKS}]*)|"
    rf"\.(?=\s|$|[{CLOSING_MARKS}])(?:[{CLOSING_MARKS}]*))"
)

DEFAULT_TOKENIZER = "Qwen/Qwen3-8B"
TOKENIZER = None


def clean_text(text: str) -> str:
    """Remove formatting whitespace without joining words in Latin text."""
    text = text.strip()
    # Indentation in the Chinese source is formatting, not content.
    text = re.sub(r"^[\u3000\t ]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def token_count(text: str) -> int:
    """Count tokens with the configured Qwen3 tokenizer."""
    if TOKENIZER is None:
        raise RuntimeError(
            "尚未載入 tokenizer；請先呼叫 load_tokenizer() 或使用程式入口"
        )
    return len(TOKENIZER.encode(text, add_special_tokens=False))


def load_tokenizer(model_name_or_path: str = DEFAULT_TOKENIZER) -> None:
    """Load the tokenizer used for chunk boundaries and metadata counts."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "需要 transformers 才能載入 Qwen3 tokenizer；請先安裝專案依賴"
        ) from exc

    global TOKENIZER
    TOKENIZER = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)


def find_split_position(text: str, max_tokens: int) -> int:
    """Find the latest complete-sentence boundary within ``max_tokens``."""
    best = 0
    for match in SENTENCE_END_RE.finditer(text):
        position = match.end()
        if token_count(text[:position]) <= max_tokens:
            best = position
        else:
            break
    if best:
        return best

    # An unusually long sentence has no sentence boundary in range.  Prefer
    # an English whitespace boundary; otherwise use the largest character
    # prefix that remains within the token limit.
    for match in re.finditer(r"\s+", text):
        position = match.start()
        if token_count(text[:position]) <= max_tokens:
            best = position
        else:
            break
    if best:
        return best

    for position in range(1, len(text) + 1):
        if token_count(text[:position]) > max_tokens:
            return max(1, position - 1)
    return len(text)


def iter_source_paragraphs(path: Path) -> Iterator[tuple[str, str]]:
    """Yield ``(chapter, paragraph)`` pairs from one input file."""
    chapter = "未分類"
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        marker = CHAPTER_RE.match(raw_line)
        if marker:
            chapter = marker.group(1) or "未命名章節"
            continue
        paragraph = clean_text(raw_line)
        if paragraph:
            yield chapter, paragraph


def make_chunks(
    source: Iterable[tuple[str, str]],
    min_tokens: int = 150,
    max_tokens: int = 250,
) -> Iterator[tuple[str, str]]:
    """Combine source paragraphs into token-sized chunks.

    A chunk never crosses a chapter boundary.  When adding a paragraph would
    exceed ``max_tokens``, split near the limit at sentence punctuation and
    keep the remainder to combine with the following paragraph.  Chunks
    shorter than ``min_tokens`` are discarded, including the final chunk of a
    chapter.  Discarding happens after splitting, so the discarded text is
    not merged into the following chunk.
    """
    # 小於最小token 要去掉
    if min_tokens <= 0 or max_tokens < min_tokens:
        raise ValueError("需要符合 0 < min_tokens <= max_tokens")

    current = ""
    current_chapter = ""

    def flush() -> tuple[str, str] | None:
        nonlocal current
        if not current:
            return None
        result = (current_chapter, current)
        current = ""
        if token_count(result[1]) < min_tokens:
            return None
        return result

    for chapter, paragraph in source:
        if chapter != current_chapter and current:
            result = flush()
            if result:
                yield result

        current_chapter = chapter
        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        while token_count(candidate) > max_tokens:
            position = find_split_position(candidate, max_tokens)
            piece, candidate = candidate[:position].strip(), candidate[position:].strip()
            if piece and token_count(piece) >= min_tokens:
                yield current_chapter, piece
        current = candidate

    result = flush()
    if result:
        yield result


def split_file(
    input_path: Path,
    output_path: Path,
    *,
    min_tokens: int = 150,
    max_tokens: int = 250,
) -> int:
    """Convert one text file to JSONL and return the record count."""
    author = input_path.parent.name or "unknown"
    book_name = input_path.stem
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as output:
        source = iter_source_paragraphs(input_path)
        for chapter, paragraph in make_chunks(source, min_tokens, max_tokens):
            record = {
                "paragraph": paragraph,
                "metadata": {
                    "author": author,
                    "book_name": book_name,
                    "chapter": chapter,
                    "summary": "",
                    "char_count": len(paragraph),
                    "token_count": token_count(paragraph),
                },
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("book_names", nargs="+", help="書名（不含副檔名），可指定多本")
    parser.add_argument("--min-tokens", type=int, default=150)
    parser.add_argument("--max-tokens", type=int, default=250)
    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER,
        help=f"Hugging Face tokenizer 名稱或本地路徑（預設：{DEFAULT_TOKENIZER}）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    books_dir = Path(__file__).resolve().parent
    raw_texts_dir = books_dir / "raw_texts"
    output_dir = books_dir / "paragraphs"

    input_paths: list[Path] = []
    for requested_name in args.book_names:
        book_name = requested_name.removesuffix(".txt")
        matches = list(raw_texts_dir.rglob(f"{book_name}.txt"))
        if not matches:
            raise SystemExit(f"找不到輸入書籍：{raw_texts_dir}/**/{book_name}.txt")
        if len(matches) > 1:
            files = "\n".join(f"  {path}" for path in matches)
            raise SystemExit(f"找到多本同名書籍，請確認書名：\n{files}")
        input_paths.append(matches[0])

    print(f"載入 tokenizer：{args.tokenizer}")
    load_tokenizer(args.tokenizer)
    for input_path in input_paths:
        output_path = output_dir / input_path.relative_to(raw_texts_dir).with_suffix(".jsonl")
        count = split_file(
            input_path,
            output_path,
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
        )
        print(f"{input_path} -> {output_path}: {count} 筆")


if __name__ == "__main__":
    main()
