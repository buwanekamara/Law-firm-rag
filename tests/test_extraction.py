"""Phase 1 tests - run against the real PDFs, because the cleaning rules only
matter in relation to what is actually inside these files.
"""

from pathlib import Path

import pytest
from app.config import list_contracts
from app.ingest.extraction import (
    PLACEHOLDER_MARKER,
    REDACTION_MARKER,
    clean_page,
    derive_doc_meta,
    extract_document,
    find_running_lines,
)


@pytest.fixture(scope="module")
def documents() -> dict[str, dict]:
    """Extract every contract once and share it across the tests."""
    return {doc["doc_id"]: doc for doc in (extract_document(p) for p in list_contracts())}


def full_text(document: dict) -> str:
    return "\n".join(page["text"] for page in document["pages"])


# --- filename parsing ------------------------------------------------------

@pytest.mark.parametrize(
    "filename, expected_id, expected_title",
    [
        (
            "ACCELERATEDTECHNOLOGIESHOLDINGCORP_04_24_2003-EX-10.13-JOINT VENTURE AGREEMENT.PDF",
            "joint_venture_agreement",
            "Joint Venture Agreement",
        ),
        (
            "BellringBrandsInc_20190920_S-1_EX-10.12_11817081_EX-10.12_Manufacturing Agreement1.pdf",
            "manufacturing_agreement",
            "Manufacturing Agreement",
        ),
        (
            "Freecook_20180605_S-1_EX-10.3_11233807_EX-10.3_Hosting Agreement.pdf",
            "hosting_agreement",
            "Hosting Agreement",
        ),
    ],
)
def test_derive_doc_meta(filename, expected_id, expected_title):
    doc_id, title = derive_doc_meta(Path(filename))
    assert (doc_id, title) == (expected_id, expected_title)


# --- the corpus extracts as expected ---------------------------------------

def test_all_five_documents_extract(documents):
    assert set(documents) == {
        "joint_venture_agreement",
        "manufacturing_agreement",
        "hosting_agreement",
        "trademark_license_agreement",
        "transportation_agreement",
    }


@pytest.mark.parametrize(
    "doc_id, page_count",
    [
        ("joint_venture_agreement", 3),
        ("manufacturing_agreement", 14),
        ("hosting_agreement", 3),
        ("trademark_license_agreement", 7),
        ("transportation_agreement", 20),
    ],
)
def test_page_counts_match_the_pdfs(documents, doc_id, page_count):
    assert documents[doc_id]["page_count"] == page_count


def test_known_text_survives(documents):
    assert "Joint Venture" in full_text(documents["joint_venture_agreement"])
    assert "Website Design and Development" in full_text(documents["hosting_agreement"])
    # This one is the point of normalising spaces: the source PDF writes it
    # with non-breaking spaces, so a naive extractor never matches it.
    assert "Article I. Definitions" in full_text(documents["transportation_agreement"])


# --- the traps -------------------------------------------------------------

def test_redaction_markers_are_preserved(documents):
    stats = documents["manufacturing_agreement"]["stats"]
    assert stats["redaction_markers"] == stats["redaction_markers_raw"] == 94
    assert REDACTION_MARKER in full_text(documents["manufacturing_agreement"])


def test_placeholder_dates_are_preserved(documents):
    stats = documents["trademark_license_agreement"]["stats"]
    assert stats["placeholder_markers"] == stats["placeholder_markers_raw"] == 9
    assert f"the {PLACEHOLDER_MARKER} day of" in full_text(documents["trademark_license_agreement"])


# --- cleaning --------------------------------------------------------------

def test_no_unicode_spaces_remain(documents):
    for doc_id, document in documents.items():
        assert "\xa0" not in full_text(document), doc_id


def test_edgar_footer_is_stripped(documents):
    for doc_id, document in documents.items():
        assert "Source:" not in full_text(document), doc_id


def test_hyphenated_line_break_is_joined():
    cleaned, _ = clean_page("the indemni-\nfication clause", set())
    assert "indemnification" in cleaned


def test_short_documents_keep_their_repeated_lines():
    """A three-page contract repeats its signature block; that is content."""
    pages = ["Client: /s/ A\nbody", "Client: /s/ A\nbody"]
    assert find_running_lines(pages) == set()
