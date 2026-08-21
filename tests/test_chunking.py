"""Phase 2 tests.

The assertions are deliberately specific - "section 1 of the hosting agreement
is one chunk titled Website Design and Development" - because a chunker that
silently drifts produces citations that look right and are wrong.
"""

import pytest

from app.chunking import (
    MAX_WORDS,
    chunk_document,
    citation_header,
    detect_styles,
    flatten_pages,
    short_title,
)
from app.extraction import extract_document
from app.config import list_contracts


@pytest.fixture(scope="module")
def chunks_by_doc() -> dict[str, list[dict]]:
    result = {}
    for path in list_contracts():
        document = extract_document(path)
        result[document["doc_id"]] = chunk_document(document)
    return result


def labels(chunks: list[dict]) -> list[str]:
    seen: list[str] = []
    for chunk in chunks:
        if chunk["section_label"] not in seen:
            seen.append(chunk["section_label"])
    return seen


# --- style detection is per document ---------------------------------------

def test_transportation_uses_articles_not_stray_numbers(chunks_by_doc):
    """The transportation agreement contains "1." and "2." as list items
    inside Article IV. They must not become sections."""
    chunks = chunks_by_doc["transportation_agreement"]
    assert "Article V" in labels(chunks)
    assert "Section 1" not in labels(chunks)
    assert len([lab for lab in labels(chunks) if lab.startswith("Article")]) == 15


def test_manufacturing_uses_bare_numbers_with_titles(chunks_by_doc):
    chunks = chunks_by_doc["manufacturing_agreement"]
    assert len([lab for lab in labels(chunks) if lab.startswith("Section")]) == 21
    section_one = next(c for c in chunks if c["section_label"] == "Section 1")
    assert section_one["section_heading"] == "BASIC TERMS"


def test_styles_chosen_per_document(chunks_by_doc):
    manufacturing = flatten_pages(extract_document(
        next(p for p in list_contracts() if "Manufacturing" in p.name)
    ))
    names = {style.name for style in detect_styles(manufacturing)}
    assert "numbered_bare" in names
    assert "article_roman" not in names


# --- the hosting agreement's Exhibit A trap --------------------------------

def test_exhibit_list_items_are_not_sections(chunks_by_doc):
    """Exhibit A of the hosting agreement is a numbered list of ten items.
    Those are deliverables, not clauses."""
    chunks = chunks_by_doc["hosting_agreement"]
    assert labels(chunks) == ["Preamble", "Section 1", "Section 2", "Section 3", "Section 4", "Exhibit A"]


def test_schedule_mentioned_mid_document_does_not_start_an_appendix(chunks_by_doc):
    """"Schedule C" sits on its own line inside section 1 of the manufacturing
    agreement. Treating it as the start of the appendices would discard
    sections 2 to 21."""
    labels_found = labels(chunks_by_doc["manufacturing_agreement"])
    assert "Section 21" in labels_found


# --- nested sections -------------------------------------------------------

def test_subsections_keep_their_parent(chunks_by_doc):
    chunks = chunks_by_doc["trademark_license_agreement"]
    section_43 = next(c for c in chunks if c["section_label"] == "Section 4.3")
    assert section_43["section_heading"] == "Termination for Breach"
    assert section_43["parent_label"] == "Section 4"
    assert section_43["parent_heading"] == "Termination"


def test_clause_text_on_the_heading_line_is_not_lost(chunks_by_doc):
    """Section 1.1 of the trademark licence is a single line: heading and
    clause together. An earlier version dropped it entirely."""
    chunks = chunks_by_doc["trademark_license_agreement"]
    section_11 = next(c for c in chunks if c["section_label"] == "Section 1.1")
    assert "non-exclusive, nontransferable" in section_11["text"]


def test_first_sentence_stays_in_the_chunk_text(chunks_by_doc):
    chunks = chunks_by_doc["hosting_agreement"]
    payment = next(c for c in chunks if c["section_label"] == "Section 2")
    assert payment["text"].startswith("2. Payment Terms.")
    assert "$5,000" in payment["text"]


# --- metadata every citation depends on ------------------------------------

def test_every_chunk_is_citable(chunks_by_doc):
    for doc_id, chunks in chunks_by_doc.items():
        for chunk in chunks:
            assert chunk["doc_title"], chunk["chunk_id"]
            assert chunk["section_label"], chunk["chunk_id"]
            assert 1 <= chunk["page_start"] <= chunk["page_end"], chunk["chunk_id"]
            assert chunk["text"].strip(), chunk["chunk_id"]


def test_chunk_ids_are_unique(chunks_by_doc):
    ids = [chunk["chunk_id"] for chunks in chunks_by_doc.values() for chunk in chunks]
    assert len(ids) == len(set(ids))


def test_no_chunk_exceeds_the_embedding_budget(chunks_by_doc):
    for chunks in chunks_by_doc.values():
        for chunk in chunks:
            assert chunk["word_count"] <= MAX_WORDS + 1, chunk["chunk_id"]


def test_nothing_is_lost_except_group_titles(chunks_by_doc):
    """Every line of every contract ends up in a chunk, apart from bare group
    headings, which are carried as parent metadata instead."""
    for path in list_contracts():
        document = extract_document(path)
        source = {
            line.strip()
            for page in document["pages"]
            for line in page["text"].split("\n")
            if line.strip()
        }
        chunked = {
            line.strip()
            for chunk in chunks_by_doc[document["doc_id"]]
            for line in chunk["text"].split("\n")
            if line.strip()
        }
        missing = source - chunked
        assert all(len(line.split()) <= 8 for line in missing), (document["doc_id"], missing)


def test_citation_header_format(chunks_by_doc):
    chunk = next(
        c for c in chunks_by_doc["trademark_license_agreement"] if c["section_label"] == "Section 4.3"
    )
    assert citation_header(chunk) == (
        "[Trademark License Agreement | Section 4.3 - Termination for Breach | p.2]"
    )


def test_short_title_stops_at_the_first_sentence():
    assert short_title("Payment Terms. Upon the signing of this Agreement") == "Payment Terms"
    assert short_title("Grant of Rights; Sublicensing.") == "Grant of Rights; Sublicensing"
