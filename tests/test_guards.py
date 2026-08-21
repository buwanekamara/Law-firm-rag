"""Tests for the checks that sit around the model.

Phase 5 uses these for evaluation; phase 6 puts them in the answer path.
"""

import pytest

from app.guards import (
    citation_matches,
    looks_like_refusal,
    normalise_label,
    parse_label,
    reports_placeholder,
    reports_redaction,
    verify_citations,
)
from app.prompting import available_versions, load_prompt


# --- label normalisation ---------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Section 4.2", "4.2"),
        ("4.2", "4.2"),
        ("4.2 - Termination for Convenience", "4.2"),
        ("SECTION 4.2.", "4.2"),
        ("  Section 4.2  ", "4.2"),
        ("Article X", "x"),
        ("Preamble", "preamble"),
    ],
)
def test_labels_reduce_to_the_same_identifier(raw, expected):
    """The model formats section labels differently between runs of the same
    question; comparing raw strings would flag honest citations as invented."""
    assert normalise_label(raw) == expected


def test_kind_is_kept_so_article_x_is_not_exhibit_x():
    assert parse_label("Article X") == ("article", "x")
    assert not citation_matches({"section": "Article X"}, {"section_label": "Exhibit X"})


def test_bare_identifier_still_matches():
    """The model often writes "4.2" where we wrote "Section 4.2"."""
    assert citation_matches({"section": "4.2"}, {"section_label": "Section 4.2"})


# --- citation verification -------------------------------------------------

RETRIEVED = [
    {"doc_title": "Trademark License Agreement", "section_label": "Section 4.2"},
    {"doc_title": "Trademark License Agreement", "section_label": "Section 4.3"},
]


def test_real_citations_are_supported():
    supported, unsupported = verify_citations(
        [{"doc_title": "Trademark License Agreement", "section": "Section 4.3"}], RETRIEVED
    )
    assert len(supported) == 1 and not unsupported


def test_invented_section_is_unsupported():
    supported, unsupported = verify_citations(
        [{"doc_title": "Trademark License Agreement", "section": "Section 99"}], RETRIEVED
    )
    assert not supported and len(unsupported) == 1


def test_right_section_from_the_wrong_document_is_unsupported():
    """A plausible failure: the section number exists, but in another contract."""
    supported, unsupported = verify_citations(
        [{"doc_title": "Manufacturing Agreement", "section": "Section 4.2"}], RETRIEVED
    )
    assert not supported and len(unsupported) == 1


def test_citation_without_a_document_is_given_the_benefit_of_the_doubt():
    supported, _ = verify_citations([{"doc_title": "", "section": "Section 4.2"}], RETRIEVED)
    assert len(supported) == 1


# --- answer shape detection ------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "The provided contracts do not contain a lease.",
        "The excerpts do not address the notice period.",
        "No relevant clause was found.",
    ],
)
def test_refusals_are_recognised(text):
    assert looks_like_refusal(text)


def test_a_real_answer_is_not_a_refusal():
    assert not looks_like_refusal("Licensor may terminate immediately on written notice.")


def test_redaction_and_placeholder_language_is_recognised():
    assert reports_redaction("The price is redacted in the filed copy.")
    assert reports_redaction("The figure was withheld before filing.")
    assert reports_placeholder("The date was left blank on the executed copy.")
    assert reports_placeholder("The effective date is shown as a placeholder.")


def test_echoing_the_marker_is_not_explaining_it():
    """v1 answered "the effective date is the [·] day of [·], 2019" - which
    reads as a date and could be copied into a document as one. Showing the
    marker is not the same as telling the reader the field was never filled
    in, so the explanation is what gets matched, not the symbol."""
    assert not reports_placeholder("The effective date is the [·] day of [·], 2019.")
    assert not reports_redaction("The amount is [***] under Schedule C.")


def test_saying_nothing_is_not_reporting_a_redaction():
    """The exact failure v1 made: 'not specified' is a different claim from
    'redacted', and only one of them is true."""
    assert not reports_redaction("The excerpts do not specify the price per unit.")


# --- prompt v2 -------------------------------------------------------------

def test_v2_exists_alongside_v1():
    assert {"v1", "v2"} <= set(available_versions())


def test_v2_covers_the_three_traps():
    system, _ = load_prompt("v2")
    assert "[***]" in system
    assert "[·]" in system
    assert "redact" in system.lower()
    assert "verbatim" in system.lower()


def test_v2_keeps_the_placeholders():
    _, user = load_prompt("v2")
    assert "{{EXCERPTS}}" in user and "{{QUESTION}}" in user


def test_target_without_a_document_title_still_matches():
    """Regression: tolerance for a missing title has to work on both sides.

    When it worked on only one, the answer evaluation compared real citations
    against a title-less target and reported nineteen correct answers as
    uncited.
    """
    assert citation_matches(
        {"doc_title": "Manufacturing Agreement", "section": "Section 12"},
        {"doc_title": "", "section_label": "Section 12"},
    )


def test_do_not_specify_is_a_refusal():
    """Regression: the refusal vocabulary and the not-stated vocabulary were
    separate lists, so "the excerpts do not specify who owns the content" -
    a perfectly good refusal - was scored as a failure to refuse."""
    assert looks_like_refusal("The provided excerpts do not specify who owns the material.")
    assert looks_like_refusal("It is unclear based on the provided information.")


def test_v3_requires_a_citation_when_the_value_is_unavailable():
    system, _ = load_prompt("v3")
    assert "Citations are required whenever" in system
    assert "defers the substance elsewhere" in system


# --- the judge prompt ------------------------------------------------------

def test_judge_prompt_renders_both_slots():
    from app.prompting import render

    system, user = render("judge_v1", EXCERPTS="THE EXCERPTS", ANSWER="THE ANSWER")
    assert "atomic factual claims" in system
    assert "THE EXCERPTS" in user and "THE ANSWER" in user
    assert "{{" not in user


def test_judge_prompt_covers_the_refusal_and_marker_cases():
    """A refusal is a claim too: "the excerpts do not contain X" is supported
    when X genuinely is not there. Without this the judge marks every correct
    refusal as unfaithful."""
    from app.prompting import load_named_prompt

    system, _ = load_named_prompt("judge_v1")
    assert "do not* contain" in system or "do not" in system
    assert "redacted" in system


def test_v4_requires_the_document_name_in_the_prose():
    """In a conversational interface the prose is what gets read, and
    "(Section 4.2)" does not say which of five contracts it came from."""
    system, _ = load_prompt("v4")
    assert "Name the contract in the sentence itself" in system
    assert "five agreements" in system


def test_v4_keeps_everything_v3_added():
    """v4 is v3 plus one rule - the trap handling must survive."""
    system, _ = load_prompt("v4")
    assert "Citations are required whenever" in system
    assert "defers the substance elsewhere" in system
    assert "[***]" in system and "[·]" in system


@pytest.mark.parametrize(
    "text",
    [
        "The Hosting Agreement does not explicitly state who owns the material.",
        "The excerpts do not clearly specify the price.",
        "The specifications themselves are not set out in the agreement.",
        "The agreement is silent on ownership.",
        "The provided contracts do not contain a lease.",
    ],
)
def test_negation_survives_an_adverb(text):
    """Regression: matching literal substrings meant "does not explicitly
    state" contained neither "does not state" nor "not stated", so a correct
    answer was scored as a failure to refuse. Up to two words may now sit
    between the negation and the verb."""
    assert looks_like_refusal(text)


@pytest.mark.parametrize(
    "text",
    [
        "Licensor may terminate immediately on written notice (Section 4.2).",
        "The Minimum Annual Order Volume is specified as [***] Units per year.",
        "Heritage bears the risk of loss until delivery to a carrier.",
        "The charge is redacted; Section 3 shows it as [***].",
    ],
)
def test_a_real_answer_is_still_not_a_refusal(text):
    """The looser pattern must not start swallowing genuine answers."""
    assert not looks_like_refusal(text)
