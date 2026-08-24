"""PDF text extraction.

Each contract becomes a JSON file of cleaned, per-page text. Page numbers
survive because every citation has to name one.

Two cleaning rules: fix what the PDF layer broke (split words, exotic spaces,
running headers), and never "fix" the contract itself. Redaction markers
([***]) and placeholder dates are real content - removing them would hand the
model a silent gap to fill in.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pymupdf

from app.config import list_contracts, settings

# Markers that must survive extraction untouched.
REDACTION_MARKER = "[***]"
PLACEHOLDER_MARKER = "[·]"

# Two of the five PDFs are made almost entirely of non-breaking spaces. Left
# alone, "Article\xa0V" never matches a search for "Article V".
_UNICODE_SPACES = "       ⁠﻿"  # noqa: RUF001 - lookalike spaces are the point

# EDGAR stamps this on every page of a filed exhibit. Repeated 20 times it
# pollutes retrieval.
_EDGAR_FOOTER = re.compile(r"^Source:\s+.+,\s+.+,\s+\d{1,2}/\d{1,2}/\d{4}\s*$")

# A line holding nothing but a page number, optionally dashed: 4, -4-, - 4 -.
_PAGE_NUMBER = re.compile(r"^[-–—]?\s*\d{1,3}\s*[-–—]?$")  # noqa: RUF001 - en dashes appear in the PDFs

# "indemni-\nfication" -> "indemnification". Lowercase both sides, so
# hyphenated proper nouns like "third-\nParty" are left alone.
_HYPHEN_BREAK = re.compile(r"(?<=[a-z])-\n(?=[a-z])")


def derive_doc_meta(path: Path) -> tuple[str, str]:
    """Turn an EDGAR filename into a readable title and a short id.

    The filenames are ugly but follow a pattern - the human title is the last
    piece:

        BellringBrandsInc_..._EX-10.12_Manufacturing Agreement1
                                       ^^^^^^^^^^^^^^^^^^^^^^^^

    Returns ("manufacturing_agreement", "Manufacturing Agreement"). The title
    is what a citation shows; the id is what filters use.

    This assumes the filename convention holds. Each contract also states its
    own name on page one, so the title could be read from the text with the
    filename as a fallback - at the cost of a heuristic that fails quietly on
    a document that never announces itself.
    """
    candidate = path.stem.split("_")[-1].split("-")[-1]
    candidate = re.sub(r"\d+$", "", candidate).strip()  # "Agreement1" -> "Agreement"
    candidate = " ".join(candidate.split())
    if not candidate:
        candidate = path.stem
    title = candidate.title() if candidate.isupper() else candidate
    doc_id = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    return doc_id, title


def normalize_spaces(text: str) -> str:
    """Replace exotic space characters with ordinary ones."""
    for char in _UNICODE_SPACES:
        text = text.replace(char, " ")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def find_running_lines(
    pages: list[str],
    edge_lines: int = 2,
    min_pages: int = 4,
    ratio: float = 0.6,
) -> set[str]:
    """Detect running headers and footers by how often a line repeats.

    Only the first and last couple of lines per page, so a sentence that
    recurs in the body is never mistaken for furniture.

    Documents under `min_pages` are skipped: in a three-page contract
    "repeated on most pages" and "appears twice" are the same condition, and
    signature blocks appear twice. Those rely on the footer and page-number
    rules instead.
    """
    if len(pages) < min_pages:
        return set()

    counts: Counter[str] = Counter()
    for text in pages:
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        if not lines:
            continue
        counts.update(set(lines[:edge_lines] + lines[-edge_lines:]))

    threshold = max(2, math.ceil(len(pages) * ratio))
    return {line for line, count in counts.items() if count >= threshold}


def clean_page(text: str, running_lines: set[str]) -> tuple[str, list[str]]:
    """Clean one page. Returns the text and the lines that were dropped."""
    text = normalize_spaces(text)
    text = _HYPHEN_BREAK.sub("", text)

    kept: list[str] = []
    dropped: list[str] = []
    for raw_line in text.split("\n"):
        line = " ".join(raw_line.split())  # collapse runs of spaces, trim ends
        if not line:
            kept.append("")
            continue
        if line in running_lines or _EDGAR_FOOTER.match(line) or _PAGE_NUMBER.match(line):
            dropped.append(line)
            continue
        kept.append(line)

    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
    return cleaned, dropped


def extract_document(path: Path) -> dict[str, Any]:
    """Extract one PDF into the dictionary that gets written to JSON."""
    doc_id, title = derive_doc_meta(path)

    with pymupdf.open(path) as pdf:
        raw_pages = [page.get_text() for page in pdf]

    normalized = [normalize_spaces(text) for text in raw_pages]
    running_lines = find_running_lines(normalized)

    pages: list[dict[str, Any]] = []
    dropped: list[str] = []
    for page_no, text in enumerate(normalized, start=1):
        cleaned, page_dropped = clean_page(text, running_lines)
        pages.append({"page_no": page_no, "text": cleaned})
        dropped.extend(page_dropped)

    raw_all = "".join(raw_pages)
    clean_all = "\n".join(p["text"] for p in pages)

    return {
        "doc_id": doc_id,
        "title": title,
        "source_file": path.name,
        "page_count": len(pages),
        "pages": pages,
        # So the CLI and the tests can check the cleaning without re-opening
        # the PDF.
        "stats": {
            "raw_chars": len(raw_all),
            "clean_chars": len(clean_all),
            "redaction_markers": clean_all.count(REDACTION_MARKER),
            "redaction_markers_raw": raw_all.count(REDACTION_MARKER),
            "placeholder_markers": clean_all.count(PLACEHOLDER_MARKER),
            "placeholder_markers_raw": raw_all.count(PLACEHOLDER_MARKER),
            "running_lines": sorted(running_lines),
            "dropped_line_count": len(dropped),
            "dropped_samples": sorted(set(dropped))[:10],
        },
    }


def save_document(document: dict[str, Any], write_txt: bool = True) -> Path:
    """Write the JSON (and optionally a plain-text dump for eyeballing)."""
    settings.extracted_dir.mkdir(parents=True, exist_ok=True)
    json_path = settings.extracted_dir / f"{document['doc_id']}.json"
    json_path.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")

    if write_txt:
        dump = "\n\n".join(
            f"--- page {p['page_no']} ---\n{p['text']}" for p in document["pages"]
        )
        json_path.with_suffix(".txt").write_text(dump, encoding="utf-8")

    return json_path


def extract_all(doc_filter: str | None = None, write_txt: bool = True) -> list[dict[str, Any]]:
    """Extract every contract, or just the ones matching `doc_filter`."""
    documents = []
    for path in list_contracts():
        doc_id, _ = derive_doc_meta(path)
        if doc_filter and doc_filter.lower() not in doc_id:
            continue
        document = extract_document(path)
        save_document(document, write_txt=write_txt)
        documents.append(document)
    return documents


def load_document(doc_id: str) -> dict[str, Any]:
    """Read back a previously extracted document."""
    path = settings.extracted_dir / f"{doc_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run: uv run scripts/extract.py")
    return json.loads(path.read_text(encoding="utf-8"))
