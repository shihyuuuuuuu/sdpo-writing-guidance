#!/usr/bin/env python3
"""Convert an EPUB to clean UTF-8 text for the writing pipeline.

Chapter names are read from the EPUB navigation (EPUB 2 NCX or EPUB 3 HTML
navigation) when available.  This keeps the converter independent of a
particular language or publisher's chapter naming convention.
"""

from __future__ import annotations

import argparse
import html
import posixpath
import re
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterator
from xml.etree import ElementTree


BLOCK_TAGS = {"address", "article", "aside", "blockquote", "div", "h1",
              "h2", "h3", "h4", "h5", "h6", "li", "p", "section"}
SKIP_TAGS = {"head", "script", "style", "title"}


class TextParser(HTMLParser):
  """Extract visible text while retaining HTML line and paragraph breaks."""

  def __init__(self) -> None:
    super().__init__(convert_charrefs=True)
    self.parts: list[str] = []
    self.skip_depth = 0

  def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
    tag = tag.lower()
    if tag in SKIP_TAGS:
      self.skip_depth += 1
    elif not self.skip_depth and tag == "br":
      self.parts.append("\n")
    elif not self.skip_depth and tag in BLOCK_TAGS:
      self.parts.append("\n")

  def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
    self.handle_starttag(tag, attrs)
    if tag.lower() in SKIP_TAGS:
      self.handle_endtag(tag)

  def handle_endtag(self, tag: str) -> None:
    tag = tag.lower()
    if tag in SKIP_TAGS and self.skip_depth:
      self.skip_depth -= 1
    elif not self.skip_depth and tag in BLOCK_TAGS:
      self.parts.append("\n")

  def handle_data(self, data: str) -> None:
    if not self.skip_depth:
      self.parts.append(data)


class NavigationParser(HTMLParser):
  """Collect labels and targets from an EPUB 3 HTML navigation document."""

  def __init__(self) -> None:
    super().__init__(convert_charrefs=True)
    self.target: str | None = None
    self.text: list[str] = []
    self.links: list[tuple[str, str]] = []

  def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
    if tag.lower() != "a":
      return
    attributes = dict(attrs)
    if attributes.get("href"):
      self.target = attributes["href"]
      self.text = []

  def handle_data(self, data: str) -> None:
    if self.target is not None:
      self.text.append(data)

  def handle_endtag(self, tag: str) -> None:
    if tag.lower() == "a" and self.target is not None:
      label = re.sub(r"\s+", " ", "".join(self.text)).strip()
      if label:
        self.links.append((self.target, label))
      self.target = None
      self.text = []


def visible_paragraphs(raw_html: str) -> list[str]:
  parser = TextParser()
  parser.feed(raw_html.lstrip("\ufeff"))
  text = html.unescape("".join(parser.parts))
  text = text.replace("\r\n", "\n").replace("\r", "\n")
  paragraphs: list[str] = []
  for block in re.split(r"\n\s*\n+", text):
    block = re.sub(r"\s*\n\s*", " ", block)
    block = re.sub(r"[ \t\u3000]+", " ", block).strip()
    if block:
      paragraphs.append(block)
  return paragraphs


def local_name(tag: str) -> str:
  return tag.rsplit("}", 1)[-1]


def read_xml(zf: zipfile.ZipFile, name: str) -> ElementTree.Element:
  return ElementTree.fromstring(zf.read(name))


def href_key(base_dir: str, href: str) -> str:
  """Normalize a manifest-relative href while retaining an optional anchor."""
  path, _, fragment = href.partition("#")
  path = posixpath.normpath(posixpath.join(base_dir, path))
  return f"{path}#{fragment}" if fragment else path


def navigation_titles(
    zf: zipfile.ZipFile,
    opf_dir: str,
    manifest: dict[str, tuple[str, str, str]],
) -> dict[str, str]:
  """Return ``document href -> chapter title`` from NCX and/or EPUB 3 nav."""
  titles: dict[str, str] = {}
  for media_type, href, properties in manifest.values():
    if media_type == "application/x-dtbncx+xml":
      root = read_xml(zf, href)
      for point in root.iter():
        if local_name(point.tag) != "navPoint":
          continue
        label = next((e.text.strip() for e in point.iter()
                      if local_name(e.tag) == "text" and e.text and e.text.strip()), "")
        src = next((e.attrib.get("src", "") for e in point.iter()
                    if local_name(e.tag) == "content"), "")
        if label and src:
          target = href_key(posixpath.dirname(href), src)
          titles.setdefault(target, label)
          # If a navigation target uses an anchor, retain a document-level
          # fallback for converters that emit one Topic per spine document.
          titles.setdefault(target.split("#", 1)[0], label)
    elif "nav" in properties.split():
      parser = NavigationParser()
      parser.feed(zf.read(href).decode("utf-8-sig", errors="replace"))
      for src, label in parser.links:
        target = href_key(posixpath.dirname(href), src)
        titles.setdefault(target, label)
        titles.setdefault(target.split("#", 1)[0], label)
  return titles


def spine_documents(zf: zipfile.ZipFile) -> Iterator[tuple[str, list[str]]]:
  """Yield chapter and paragraphs in the EPUB reading order."""
  container = read_xml(zf, "META-INF/container.xml")
  rootfile = next(e for e in container.iter() if local_name(e.tag) == "rootfile")
  opf_path = rootfile.attrib["full-path"]
  opf_dir = posixpath.dirname(opf_path)
  opf = read_xml(zf, opf_path)
  manifest: dict[str, tuple[str, str, str]] = {}
  nav_items: set[str] = set()
  for item in opf.iter():
    if local_name(item.tag) != "item":
      continue
    href = posixpath.normpath(posixpath.join(opf_dir, item.attrib["href"].split("#", 1)[0]))
    media_type = item.attrib.get("media-type", "").lower()
    properties = item.attrib.get("properties", "")
    manifest[item.attrib["id"]] = (media_type, href, properties)
    if "nav" in properties.split():
      nav_items.add(href)

  titles = navigation_titles(zf, opf_dir, manifest)

  for itemref in opf.iter():
    if local_name(itemref.tag) != "itemref":
      continue
    item = manifest.get(itemref.attrib.get("idref", ""))
    if not item or item[0] not in {"application/xhtml+xml", "text/html"}:
      continue
    href = item[1]
    content = zf.read(href).decode("utf-8-sig", errors="replace")
    paragraphs = visible_paragraphs(content)
    if paragraphs:
      # EPUB 3 marks its navigation document explicitly.  Older EPUBs often
      # place the TOC at 0.xhtml, which is handled by the content check below.
      if href in nav_items:
        continue
      # Some EPUBs put the table of contents in the first spine document.
      # It is navigation, not book content, so do not duplicate its chapter
      # names in the converted text.
      toc_items = paragraphs[0].split()
      is_numbered_toc = (
          posixpath.basename(href).split(".", 1)[0] == "0"
          and len(toc_items) >= 3
          and all(len(item) < 80 for item in toc_items)
      )
      if is_numbered_toc:
        continue
      chapter = titles.get(href, "")
      if not chapter and len(paragraphs[0]) <= 100:
        chapter = paragraphs[0]
      chapter = chapter or "未分類"
      yield chapter, paragraphs


def epub_to_clean_txt(epub_path: str | Path, output_txt_path: str | Path) -> int:
  """Convert an EPUB to UTF-8 text and return the number of paragraphs."""
  epub_path, output_txt_path = Path(epub_path), Path(output_txt_path)
  output_txt_path.parent.mkdir(parents=True, exist_ok=True)
  output: list[str] = []
  with zipfile.ZipFile(epub_path) as zf:
    for chapter, paragraphs in spine_documents(zf):
      output.append(f"Chapter: {chapter}")
      output.extend(p for p in paragraphs if p != chapter)
  output_txt_path.write_text("\n\n".join(output) + ("\n" if output else ""), encoding="utf-8")
  count = sum(not line.startswith("Chapter: ") for line in output)
  print(f"轉換完成：{epub_path} -> {output_txt_path}（{count} 段）")
  return count


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("book_name", help="書名（不含副檔名）")
  args = parser.parse_args()

  book_name = args.book_name.removesuffix(".epub")
  epubs_dir = Path("epubs")
  matches = list(epubs_dir.rglob(f"{book_name}.epub"))
  if not matches:
    raise SystemExit(f"找不到輸入檔案：{epubs_dir}/**/{book_name}.epub")
  if len(matches) > 1:
    files = "\n".join(f"  {path}" for path in matches)
    raise SystemExit(f"找到多本同名書籍，請確認書名：\n{files}")

  epub_file = matches[0]
  relative_epub_file = epub_file.relative_to(epubs_dir)
  output_txt_file = Path("raw_texts") / relative_epub_file.with_suffix(".txt")
  epub_to_clean_txt(epub_file, output_txt_file)


if __name__ == "__main__":
  main()
