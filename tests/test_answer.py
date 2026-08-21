"""Phase 4 tests.

No network. The model call is replaced with a stub, because what is being
tested is the plumbing around it: what goes into the prompt, what comes back
out of a messy reply, and what happens when the model misbehaves.
"""

import json

import pytest
from fastapi.testclient import TestClient

from app.answer import (
    ModelAnswer,
    answer_question,
    build_excerpts,
    extract_json,
    parse_answer,
)
from app.api import app
from app.prompting import available_versions, load_prompt, render_user_prompt
from app.retrieval import search

GOOD_REPLY = json.dumps(
    {
        "answer": "Confidentiality is governed by Section 9.",
        "citations": [
            {"doc_title": "Manufacturing Agreement", "section": "Section 9", "page": 8}
        ],
        "confidence": "high",
    }
)


@pytest.fixture
def stub_llm(monkeypatch):
    """Replace the model with a scripted sequence of replies."""

    def install(*replies: str):
        calls = []

        def fake_complete(system_prompt, user_prompt, model=None, temperature=None):
            calls.append({"system": system_prompt, "user": user_prompt})
            return replies[min(len(calls) - 1, len(replies) - 1)]

        monkeypatch.setattr("app.answer.complete", fake_complete)
        return calls

    return install


# --- prompt files ----------------------------------------------------------

def test_prompt_file_splits_into_system_and_user():
    system, user = load_prompt("v1")
    assert "contract analyst" in system.lower()
    assert "{{EXCERPTS}}" in user and "{{QUESTION}}" in user


def test_unknown_prompt_version_is_an_error():
    with pytest.raises(FileNotFoundError):
        load_prompt("does_not_exist")


def test_placeholders_are_filled():
    rendered = render_user_prompt("Who pays?", "EXCERPT TEXT", "v1")
    assert "Who pays?" in rendered
    assert "EXCERPT TEXT" in rendered
    assert "{{" not in rendered


def test_at_least_one_version_exists():
    assert "v1" in available_versions()


# --- excerpt rendering -----------------------------------------------------

def test_excerpts_carry_their_source(indexed_client):
    results = search("confidentiality", top_k=3, client=indexed_client)
    excerpts = build_excerpts(results)
    assert "[1] [" in excerpts
    for result in results:
        assert result.chunk["section_label"] in excerpts


# --- tolerating messy replies ---------------------------------------------

@pytest.mark.parametrize(
    "raw",
    [
        GOOD_REPLY,
        f"```json\n{GOOD_REPLY}\n```",
        f"Sure, here is the answer:\n{GOOD_REPLY}\nLet me know if you need more.",
    ],
)
def test_json_survives_whatever_the_model_wraps_it_in(raw):
    assert extract_json(raw)["confidence"] == "high"


def test_reply_without_json_is_rejected():
    with pytest.raises(ValueError):
        extract_json("I am afraid I cannot answer that.")


def test_missing_optional_fields_get_defaults():
    parsed = parse_answer('{"answer": "Just this."}')
    assert parsed.citations == []
    assert parsed.confidence == "medium"


# --- orchestration ---------------------------------------------------------

def test_answer_uses_retrieved_chunks(indexed_client, stub_llm):
    calls = stub_llm(GOOD_REPLY)
    result = answer_question("confidentiality obligations", top_k=3, client=indexed_client)

    assert result.answer.startswith("Confidentiality")
    assert result.confidence == "high"
    assert len(result.retrieved) == 3
    assert result.parse_error is None
    # every retrieved chunk was actually put in front of the model
    for item in result.retrieved:
        assert item["section"] in calls[0]["user"]


def test_unparseable_reply_triggers_one_retry(indexed_client, stub_llm):
    calls = stub_llm("not json at all", GOOD_REPLY)
    result = answer_question("confidentiality", top_k=2, client=indexed_client)

    assert len(calls) == 2
    assert result.parse_error is None
    assert result.confidence == "high"
    assert "could not be parsed" in calls[1]["user"]


def test_two_bad_replies_degrade_instead_of_raising(indexed_client, stub_llm):
    """A readable answer with a flag on it beats a 500."""
    stub_llm("still not json", "nope")
    result = answer_question("confidentiality", top_k=2, client=indexed_client)

    assert result.parse_error
    assert result.answer == "nope"
    assert result.confidence == "low"


def test_debug_returns_the_exact_prompt(indexed_client, stub_llm):
    stub_llm(GOOD_REPLY)
    result = answer_question("force majeure", top_k=2, debug=True, client=indexed_client)

    assert result.debug["user_prompt"]
    assert result.debug["raw_response"] == GOOD_REPLY
    assert len(result.debug["chunks"]) == 2
    assert result.debug["prompt_version"]


def test_debug_is_absent_unless_requested(indexed_client, stub_llm):
    stub_llm(GOOD_REPLY)
    result = answer_question("force majeure", top_k=2, client=indexed_client)
    assert "debug" not in result.to_dict()


def test_invented_citations_do_not_reach_the_caller(indexed_client, stub_llm, monkeypatch):
    """Phase 4 reported whatever the model claimed. Phase 6 verifies it, so
    the same reply now yields no citations and a warning instead."""
    from app.config import settings

    monkeypatch.setattr(settings, "citation_retry", False)
    stub_llm(
        json.dumps(
            {
                "answer": "Invented.",
                "citations": [{"doc_title": "Nonexistent Agreement", "section": "Section 99"}],
                "confidence": "high",
            }
        )
    )
    result = answer_question("anything", top_k=2, client=indexed_client)
    assert result.citations == []
    assert result.rejected_citations[0]["section"] == "Section 99"


# --- the API surface -------------------------------------------------------

def test_ask_route_is_published():
    assert "/ask" in app.openapi()["paths"]


def test_empty_question_is_rejected_before_anything_expensive():
    client = TestClient(app)
    response = client.post("/ask", json={"question": "hi"})
    assert response.status_code == 422


# --- the doc_id filter must not fail silently ------------------------------

def test_unknown_doc_id_is_rejected(monkeypatch):
    """An unrecognised filter is an error, not an empty answer.

    The interactive docs page pre-fills optional strings with the word
    "string"; before this check that produced a confident "no relevant clause
    was found", which reads like a corpus problem rather than a typo.
    """
    import app.api as api

    monkeypatch.setattr(api, "collection_size", lambda *args, **kwargs: 120)
    monkeypatch.setattr(api, "get_client", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        api, "list_indexed_documents", lambda client: {"hosting_agreement": "Hosting Agreement"}
    )

    response = TestClient(app).post(
        "/ask", json={"question": "What are the confidentiality obligations?", "doc_id": "string"}
    )
    assert response.status_code == 400
    assert "hosting_agreement" in response.json()["detail"]["known_documents"]


def test_doc_id_fragment_is_accepted(monkeypatch):
    import app.api as api

    monkeypatch.setattr(api, "collection_size", lambda *args, **kwargs: 120)
    monkeypatch.setattr(api, "get_client", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        api, "list_indexed_documents", lambda client: {"hosting_agreement": "Hosting Agreement"}
    )
    seen = {}

    def fake_answer(question, top_k=None, doc_id=None, debug=False, **kwargs):
        seen["doc_id"] = doc_id
        from app.answer import AnswerResult

        return AnswerResult(question, "ok", [], "high", [], doc_filter=doc_id)

    monkeypatch.setattr(api, "answer_question", fake_answer)
    response = TestClient(app).post("/ask", json={"question": "Who pays?", "doc_id": "hosting"})
    assert response.status_code == 200
    assert seen["doc_id"] == "hosting_agreement"


# --- the tester page -------------------------------------------------------

def test_index_page_is_served():
    response = TestClient(app).get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Contract RAG" in response.text


def test_page_is_self_contained():
    """No CDN, no build step: everything the page needs is in the file. It has
    to work from inside the container with no outbound network."""
    html = TestClient(app).get("/").text
    assert "<script src=" not in html
    assert "<link rel=\"stylesheet\"" not in html
    assert "http://" not in html.replace("http://127.0.0.1", "")


def test_documents_endpoint_is_published():
    assert "/documents" in app.openapi()["paths"]
