"""The guards in the answer path.

These test what happens *around* the model: refusing before it is called, and
refusing to publish a citation it made up.
"""

import json

import pytest
from app.answer import BELOW_THRESHOLD, answer_question
from app.config import settings

GOOD_REPLY = json.dumps(
    {
        "answer": "Confidentiality is governed by Section 9.",
        "citations": [{"doc_title": "Manufacturing Agreement", "section": "Section 9", "page": 8}],
        "confidence": "high",
    }
)

INVENTED_REPLY = json.dumps(
    {
        "answer": "Section 42 of the Leasing Agreement governs this.",
        "citations": [{"doc_title": "Leasing Agreement", "section": "Section 42", "page": 1}],
        "confidence": "high",
    }
)


@pytest.fixture
def stub_llm(monkeypatch):
    def install(*replies: str):
        calls = []

        def fake_complete(system_prompt, user_prompt, model=None, temperature=None):
            calls.append({"system": system_prompt, "user": user_prompt})
            return replies[min(len(calls) - 1, len(replies) - 1)]

        monkeypatch.setattr("app.answer.complete", fake_complete)
        return calls

    return install


# --- guard one: the relevance gate ----------------------------------------

def test_the_gate_threshold_is_configuration_not_a_code_default(env_template):
    """No threshold is worth baking into the code: 0 lets every question
    through and a guessed value refuses real ones. The setting is required, and
    the template carries the value scripts/calibrate_gate.py produces for this
    corpus - off-topic queries reach 0.55, the weakest real question 0.63."""
    from app.config import Settings

    assert Settings.model_fields["min_score"].is_required()
    assert float(env_template["MIN_SCORE"]) == 0.56


def test_no_gate_call_when_the_threshold_is_zero(indexed_client, stub_llm):
    """With gating off there is no extra similarity query at all."""
    calls = stub_llm(GOOD_REPLY)
    result = answer_question("confidentiality", top_k=2, client=indexed_client)
    assert len(calls) == 1
    assert result.gated is False
    assert result.gate_score is None


def test_low_similarity_refuses_without_calling_the_model(
    indexed_client, stub_llm, monkeypatch
):
    """The point of the gate: the model never gets the chance to improvise."""
    monkeypatch.setattr(settings, "min_score", 0.99)
    monkeypatch.setattr("app.answer.best_similarity", lambda *a, **k: 0.10)
    calls = stub_llm(GOOD_REPLY)

    result = answer_question("what colour is the sky", top_k=2, client=indexed_client)

    assert calls == []
    assert result.gated is True
    assert result.answer == BELOW_THRESHOLD
    assert result.citations == []
    assert result.confidence == "low"
    assert "below the threshold" in result.warnings[0]


def test_gate_still_reports_what_was_retrieved(indexed_client, stub_llm, monkeypatch):
    """A gated answer keeps its retrieval trace, so the refusal is debuggable."""
    monkeypatch.setattr(settings, "min_score", 0.99)
    monkeypatch.setattr("app.answer.best_similarity", lambda *a, **k: 0.10)
    stub_llm(GOOD_REPLY)
    result = answer_question("anything", top_k=3, client=indexed_client)
    assert len(result.retrieved) == 3


def test_a_relevant_question_passes_the_gate(indexed_client, stub_llm, monkeypatch):
    monkeypatch.setattr(settings, "min_score", 0.05)
    monkeypatch.setattr("app.answer.best_similarity", lambda *a, **k: 0.80)
    calls = stub_llm(GOOD_REPLY)
    result = answer_question("confidentiality", top_k=2, client=indexed_client)
    assert len(calls) == 1
    assert result.gated is False
    assert result.gate_score == 0.80


# --- guard two: citation verification --------------------------------------

def test_invented_citation_is_never_published(indexed_client, stub_llm, monkeypatch):
    monkeypatch.setattr(settings, "citation_retry", False)
    stub_llm(INVENTED_REPLY)
    result = answer_question("confidentiality", top_k=3, client=indexed_client)

    assert result.citations == []
    assert len(result.rejected_citations) == 1
    assert result.rejected_citations[0]["section"] == "Section 42"
    assert any("removed unsupported citation" in w for w in result.warnings)


def test_one_regeneration_is_attempted_first(indexed_client, stub_llm, monkeypatch):
    monkeypatch.setattr(settings, "citation_retry", True)
    calls = stub_llm(INVENTED_REPLY, GOOD_REPLY)

    result = answer_question("confidentiality obligations", top_k=8, client=indexed_client)

    assert len(calls) == 2
    assert "not among the excerpts" in calls[1]["user"]
    assert result.rejected_citations == []
    assert any("regenerated once" in w for w in result.warnings)


def test_a_second_invention_is_dropped_not_shown(indexed_client, stub_llm, monkeypatch):
    monkeypatch.setattr(settings, "citation_retry", True)
    stub_llm(INVENTED_REPLY, INVENTED_REPLY)
    result = answer_question("confidentiality", top_k=3, client=indexed_client)

    assert result.citations == []
    assert len(result.rejected_citations) == 1
    # The answer text is kept - it may still be useful - but nothing it points
    # at is presented as a verified source.
    assert result.answer


def test_real_citations_survive(indexed_client, stub_llm):
    stub_llm(GOOD_REPLY)
    result = answer_question("confidentiality obligations", top_k=8, client=indexed_client)
    assert result.rejected_citations == []
    assert result.warnings == []
    assert result.citations


# --- guard three: masking in the answer path -------------------------------

def test_the_reader_sees_real_names_even_though_the_model_did_not(
    indexed_client, stub_llm, monkeypatch
):
    """The model reasons about PERSON_1; the answer comes back with the name."""
    pytest.importorskip("presidio_analyzer", reason="presidio is optional")
    monkeypatch.setattr(settings, "masking_enabled", True)

    captured = {}

    def fake_complete(system_prompt, user_prompt, model=None, temperature=None):
        captured["user"] = user_prompt
        return json.dumps(
            {
                "answer": "The agreement was signed by PERSON_1.",
                "citations": [],
                "confidence": "high",
            }
        )

    monkeypatch.setattr("app.answer.complete", fake_complete)
    result = answer_question("who signed the hosting agreement", top_k=8, client=indexed_client)

    # Whatever was masked must be absent from what was sent and present in what
    # was returned - that is the entire contract of this feature.
    assert "PERSON_1" not in result.answer
    assert "Natalija" in result.answer or "Mitchell" in result.answer
    assert any("masking on" in warning for warning in result.warnings)


def test_masking_off_sends_the_text_unchanged(indexed_client, stub_llm, monkeypatch):
    monkeypatch.setattr(settings, "masking_enabled", False)
    calls = stub_llm(GOOD_REPLY)
    answer_question("who signed the hosting agreement", top_k=8, client=indexed_client)
    assert "PERSON_1" not in calls[0]["user"]
    assert "Natalija Tunevic" in calls[0]["user"]
