"""Clause-aware chunking.

A chunk is a contract section, not a fixed window of characters: a chunk
straddling two clauses has no honest section number to cite.

The hard part is that the five contracts number their sections five different
ways, and the same pattern means different things in different documents. A
bare "1." is a heading in the manufacturing agreement (twenty of them) and a
list item in the transportation agreement (exactly two, inside Article IV). So
each document is inspected first and told which style it uses.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from app.config import settings
from app.ingest.extraction import load_document

# Max chunk size in words. bge-small takes 512 tokens at roughly 1.3 tokens
# per word, so 350 leaves headroom instead of truncating a clause silently.
# Changing either means re-running extract, chunk and index.
MAX_WORDS = settings.chunk_max_words
OVERLAP_WORDS = settings.chunk_overlap_words

# Headings a style must produce to be believed. Separates "organised in
# numbered sections" from "happens to contain a numbered list".
MIN_HEADINGS = 4


@dataclass(frozen=True)
class Style:
    name: str
    pattern: re.Pattern[str]
    level: int
    # Heading text on the line after the number: "2." / "PRODUCTION OF PRODUCT".
    title_on_next_line: bool = False


STYLES: tuple[Style, ...] = (
    Style("article_roman", re.compile(r"^Article\s+([IVXLC]+)\.?\s*(.*)$"), level=1),
    Style("numbered_inline", re.compile(r"^(\d{1,2})\.\s+([A-Z].*)$"), level=1),
    Style("numbered_bare", re.compile(r"^(\d{1,2})\.$"), level=1, title_on_next_line=True),
    Style("section_numeric", re.compile(r"^Section\s+(\d+(?:\.\d+)*)\.?\s*(.*)$"), level=2),
)

# Appendices only count as headings once the body is finished. "Schedule C"
# appears on its own line inside section 1 of the manufacturing agreement;
# treating that as an appendix would discard the twenty sections after it.
EXHIBIT_PATTERN = re.compile(r"^(EXHIBIT|Exhibit|SCHEDULE|Schedule|ANNEX|Annex)\s+([A-Z0-9]{1,3})\.?$")


# End of a heading title: a stop then a space, not part of a number like
# "6.1". "Payment Terms. Upon the signing of..." -> "Payment Terms".
_TITLE_END = re.compile(r"(?<=[^\d])\.\s")


def short_title(text: str, max_length: int = 70) -> str:
    """Reduce a heading line to the title a citation should display."""
    text = text.strip()
    match = _TITLE_END.search(text)
    if match:
        text = text[: match.start()]
    text = text.strip().rstrip(".").strip()
    if len(text) > max_length:
        text = text[:max_length].rsplit(" ", 1)[0] + "..."
    return text


@dataclass
class Heading:
    line_index: int
    style: str
    level: int
    section_id: str
    label: str  # how a citation names it: "Section 4", "Article V", "Exhibit A"
    heading: str  # the human title, may be empty
    consumed_lines: int = 0  # extra lines the heading itself occupies
    # True when the clause text runs on from the heading line itself, as in
    # "Section 1.1. License Grant. Subject to the terms and conditions...".
    # Such a section has no separate body lines but is emphatically not empty.
    has_inline_body: bool = False


@dataclass
class Line:
    text: str
    page_no: int


@dataclass
class Section:
    heading: Heading | None
    heading_lines: list[Line] = field(default_factory=list)
    body_lines: list[Line] = field(default_factory=list)
    parent: Heading | None = None

    @property
    def lines(self) -> list[Line]:
        """Heading and body together - the heading line usually carries the
        first sentence of the clause, so it belongs in the chunk text."""
        return self.heading_lines + self.body_lines


def flatten_pages(document: dict[str, Any]) -> list[Line]:
    """One flat list of lines, each remembering which page it came from."""
    return [
        Line(text=line, page_no=page["page_no"])
        for page in document["pages"]
        for line in page["text"].split("\n")
    ]


def detect_styles(lines: list[Line]) -> list[Style]:
    """Decide which heading styles this document actually uses.

    A style qualifies at MIN_HEADINGS hits. That threshold is what stops two
    stray list items being read as the scheme of a document organised into
    fifteen Articles.
    """
    qualifying = []
    for style in STYLES:
        matches = sum(1 for line in lines if style.pattern.match(line.text))
        if matches >= MIN_HEADINGS:
            qualifying.append(style)

    # The trademark licence has both "4. Termination." and "Section 4.3.
    # Termination for Breach." - keep both, coarse becomes the parent. But the
    # two coarse numbered styles must not fight each other.
    if any(s.name == "numbered_bare" for s in qualifying) and any(
        s.name == "numbered_inline" for s in qualifying
    ):
        qualifying = [s for s in qualifying if s.name != "numbered_inline"]
    return qualifying


def find_exhibit_start(lines: list[Line], body_headings: list[Heading]) -> int | None:
    """Index of the line where the appendices begin, if they do.

    "Exhibit A" only starts them with at least MIN_HEADINGS sections above it.
    That separates the hosting agreement's real Exhibit A (four above) from
    "Schedule C", a table row inside section 1 with one heading above.
    """
    for index, line in enumerate(lines):
        if not EXHIBIT_PATTERN.match(line.text):
            continue
        preceding = sum(1 for heading in body_headings if heading.line_index < index)
        if preceding >= MIN_HEADINGS:
            return index
    return None


def find_headings(lines: list[Line], styles: list[Style]) -> list[Heading]:
    """Locate every heading, in document order."""
    headings: list[Heading] = []
    for index, line in enumerate(lines):
        for style in styles:
            match = style.pattern.match(line.text)
            if not match:
                continue

            number, title = match.group(1), (match.group(2) if match.lastindex >= 2 else "")
            consumed = 0
            if style.title_on_next_line and not title:
                # Look ahead for a short line acting as the title, e.g. the
                # manufacturing agreement's all-caps "BASIC TERMS".
                for offset in (1, 2):
                    if index + offset < len(lines):
                        candidate = lines[index + offset].text.strip()
                        if candidate and len(candidate) <= 80 and not candidate.endswith("."):
                            title, consumed = candidate, offset
                            break

            label = (
                f"Article {number}" if style.name == "article_roman" else f"Section {number}"
            )
            display_title = short_title(title)
            remainder = title.strip()[len(display_title):].strip(" .")
            headings.append(
                Heading(
                    line_index=index,
                    style=style.name,
                    level=style.level,
                    section_id=number,
                    label=label,
                    heading=display_title,
                    consumed_lines=consumed,
                    has_inline_body=len(remainder.split()) >= 4,
                )
            )
            break

    # Appendices, once the body is done.
    if headings:
        exhibit_start = find_exhibit_start(lines, headings)
        if exhibit_start is not None:
            headings = [h for h in headings if h.line_index < exhibit_start]
            for index in range(exhibit_start, len(lines)):
                match = EXHIBIT_PATTERN.match(lines[index].text)
                if match:
                    label = f"{match.group(1).title()} {match.group(2)}"
                    headings.append(
                        Heading(
                            line_index=index,
                            style="exhibit",
                            level=1,
                            section_id=match.group(2),
                            label=label,
                            heading="",
                        )
                    )
    return headings


def build_sections(lines: list[Line], headings: list[Heading]) -> list[Section]:
    """Slice the document into sections, one per heading.

    Text before the first heading becomes the preamble - the parties clause
    and the recitals, which are quotable, so they get a section of their own.
    """
    sections: list[Section] = []
    if not headings:
        return [Section(heading=None, body_lines=lines)]

    if headings[0].line_index > 0:
        sections.append(Section(heading=None, body_lines=lines[: headings[0].line_index]))

    for position, heading in enumerate(headings):
        start = heading.line_index
        body_start = start + heading.consumed_lines + 1
        end = headings[position + 1].line_index if position + 1 < len(headings) else len(lines)
        sections.append(
            Section(
                heading=heading,
                heading_lines=lines[start:body_start],
                body_lines=lines[body_start:end],
            )
        )

    # An empty body means a group title - "1. Grant of Rights; Sublicensing."
    # followed by "Section 1.1. License Grant." Hand it down as the parent
    # rather than emitting an empty chunk.
    collapsed: list[Section] = []
    pending_parent: Heading | None = None
    for section in sections:
        body = [line for line in section.body_lines if line.text.strip()]
        if section.heading and not body and not section.heading.has_inline_body:
            # A bare group title such as "4. Termination." with the real
            # clauses in Sections 4.1-4.5 beneath it.
            pending_parent = section.heading
            continue
        if pending_parent and section.heading and section.heading.level > pending_parent.level:
            section.parent = pending_parent
        elif section.heading and section.heading.level <= (
            pending_parent.level if pending_parent else 99
        ):
            pending_parent = None
        section.heading_lines = [ln for ln in section.heading_lines if ln.text.strip()]
        section.body_lines = body
        collapsed.append(section)
    return collapsed


def split_by_size(lines: list[Line]) -> list[list[Line]]:
    """Split an over-long section on whole lines, with a little overlap.

    The overlap keeps a sentence landing on a boundary readable in one piece.
    """
    total_words = sum(len(line.text.split()) for line in lines)
    if total_words <= MAX_WORDS:
        return [lines]

    parts: list[list[Line]] = []
    current: list[Line] = []
    current_words = 0
    for line in lines:
        words = len(line.text.split())
        if current and current_words + words > MAX_WORDS:
            parts.append(current)
            # Carry the tail of the previous part into the next one.
            overlap: list[Line] = []
            overlap_words = 0
            for previous in reversed(current):
                previous_words = len(previous.text.split())
                if overlap_words + previous_words > OVERLAP_WORDS:
                    break
                overlap.insert(0, previous)
                overlap_words += previous_words
            current, current_words = list(overlap), overlap_words
        current.append(line)
        current_words += words
    if current:
        parts.append(current)
    return parts


def chunk_document(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Turn one extracted document into citable chunks."""
    lines = flatten_pages(document)
    styles = detect_styles(lines)
    headings = find_headings(lines, styles)
    sections = build_sections(lines, headings)

    chunks: list[dict[str, Any]] = []
    for section in sections:
        if not section.lines:
            continue
        heading = section.heading
        for part_number, part in enumerate(split_by_size(section.lines), start=1):
            text = "\n".join(line.text for line in part).strip()
            if not text:
                continue
            section_id = heading.section_id if heading else "preamble"
            label = heading.label if heading else "Preamble"
            chunks.append(
                {
                    "chunk_id": f"{document['doc_id']}::{section_id}::{part_number}",
                    "doc_id": document["doc_id"],
                    "doc_title": document["title"],
                    "section_id": section_id,
                    "section_label": label,
                    "section_heading": heading.heading if heading else "Recitals",
                    "parent_label": section.parent.label if section.parent else None,
                    "parent_heading": section.parent.heading if section.parent else None,
                    "page_start": part[0].page_no,
                    "page_end": part[-1].page_no,
                    "part": part_number,
                    "word_count": len(text.split()),
                    "text": text,
                }
            )
    return chunks


def citation_header(chunk: dict[str, Any]) -> str:
    """The one-line provenance stamp put above a chunk in the prompt."""
    pages = (
        f"p.{chunk['page_start']}"
        if chunk["page_start"] == chunk["page_end"]
        else f"pp.{chunk['page_start']}-{chunk['page_end']}"
    )
    title = f" - {chunk['section_heading']}" if chunk["section_heading"] else ""
    return f"[{chunk['doc_title']} | {chunk['section_label']}{title} | {pages}]"


def chunk_all(doc_filter: str | None = None, save: bool = True) -> list[dict[str, Any]]:
    """Chunk every extracted document (or those matching `doc_filter`)."""
    if not settings.extracted_dir.is_dir():
        raise FileNotFoundError("No extracted documents - run: uv run scripts/extract.py")

    all_chunks: list[dict[str, Any]] = []
    for path in sorted(settings.extracted_dir.glob("*.json")):
        doc_id = path.stem
        if doc_filter and doc_filter.lower() not in doc_id:
            continue
        chunks = chunk_document(load_document(doc_id))
        if save:
            settings.chunks_dir.mkdir(parents=True, exist_ok=True)
            (settings.chunks_dir / f"{doc_id}.json").write_text(
                json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        all_chunks.extend(chunks)
    return all_chunks


def load_chunks() -> list[dict[str, Any]]:
    """Read every saved chunk back, for indexing and evaluation."""
    if not settings.chunks_dir.is_dir():
        raise FileNotFoundError("No chunks - run: uv run scripts/chunk.py")
    chunks: list[dict[str, Any]] = []
    for path in sorted(settings.chunks_dir.glob("*.json")):
        chunks.extend(json.loads(path.read_text(encoding="utf-8")))
    return chunks
