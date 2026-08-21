"""Phase 6.5 - PII masking.

Skipped when presidio is not installed, because masking is an optional
capability: the system runs without it and says so if it is switched on
without the dependency.
"""

import pytest

pytest.importorskip("presidio_analyzer", reason="presidio is optional")

from app.config import settings  # noqa: E402
from app.masking import (  # noqa: E402
    MASKED_ENTITIES,
    defined_terms,
    mask_excerpts,
    mask_text,
    unmask_text,
)

SAMPLE = (
    "This Agreement is entered into on January 11, 2018 by and between Natalija Tunevic, "
    "director of FreeCook, and Mitchell Vitalis. Notices to smoore@penntex.com. Heritage "
    "shall produce the Products at the Jasper facility in Santa Ana, California, and this "
    "Agreement is governed by Delaware law."
)


# --- what gets masked ------------------------------------------------------

def test_people_and_contact_details_are_masked():
    result = mask_text(SAMPLE)
    assert "Natalija Tunevic" not in result.text
    assert "Mitchell Vitalis" not in result.text
    assert "smoore@penntex.com" not in result.text
    assert "PERSON_1" in result.text
    assert "EMAIL_ADDRESS_1" in result.text


@pytest.mark.parametrize(
    "kept, why",
    [
        ("January 11, 2018", "dates are answers - 'when does this take effect'"),
        ("Delaware", "locations are answers - 'which state's law governs'"),
        ("Santa Ana", "same"),
        ("Heritage", "a contracting party, defined in quotes by the contract"),
        ("Jasper", "a facility, defined in quotes by the contract"),
        ("FreeCook", "an organisation, not a person"),
    ],
)
def test_load_bearing_text_survives(kept, why):
    """Masking that removes the answer is worse than no masking."""
    assert kept in mask_text(SAMPLE).text, why


def test_defined_terms_are_protected():
    terms = defined_terms()
    for term in ("Heritage", "Jasper", "Transporter", "Brand", "Premier"):
        assert term in terms


def test_dates_and_locations_are_not_even_candidates():
    assert "DATE_TIME" not in MASKED_ENTITIES
    assert "LOCATION" not in MASKED_ENTITIES
    assert "ORG" not in MASKED_ENTITIES


# --- reversibility ---------------------------------------------------------

def test_round_trip_is_exact():
    result = mask_text(SAMPLE)
    assert unmask_text(result.text, result.mapping) == SAMPLE


def test_the_same_person_gets_the_same_placeholder_everywhere():
    """Two excerpts, one mapping: otherwise the model sees two placeholders
    and cannot tell they are the same party."""
    first = mask_text("Natalija Tunevic signed for the Client.")
    second = mask_text("Natalija Tunevic is the director.", mapping=first.mapping)
    placeholder = next(k for k, v in first.mapping.items() if v == "Natalija Tunevic")
    assert placeholder in second.text
    assert len([v for v in second.mapping.values() if v == "Natalija Tunevic"]) == 1


def test_unmasking_handles_double_digit_placeholders():
    """PERSON_1 must not match the front of PERSON_10."""
    mapping = {"PERSON_1": "Ann", "PERSON_10": "Bob"}
    assert unmask_text("PERSON_10 met PERSON_1", mapping) == "Bob met Ann"


# --- the kill switch -------------------------------------------------------

def test_masking_is_off_by_default():
    from app.config import Settings

    assert Settings.model_fields["masking_enabled"].default is False


def test_kill_switch_bypasses_everything(monkeypatch):
    monkeypatch.setattr(settings, "masking_enabled", False)
    result = mask_excerpts(SAMPLE)
    assert result.text == SAMPLE
    assert result.mapping == {}


def test_switched_on_it_masks(monkeypatch):
    monkeypatch.setattr(settings, "masking_enabled", True)
    result = mask_excerpts(SAMPLE)
    assert "Natalija Tunevic" not in result.text
    assert result.mapping
