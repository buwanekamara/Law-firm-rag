"""Progress streaming.

The answer is not streamed. Citations are verified and masked names restored
only after the model's whole reply arrives, so prose shown before those run
would be prose nothing had checked. What streams is what the system is doing
while the model works - which is the part a person is waiting through.
"""

import json

import app.api as api
import pytest
from app.answer import AnswerResult, answer_question
from fastapi.testclient import TestClient


def read_events(response) -> list[dict]:
    return [json.loads(line) for line in response.text.strip().split("\n") if line.strip()]


@pytest.fixture
def stubbed_api(monkeypatch):
    monkeypatch.setattr(api, "collection_size", lambda *a, **k: 120)
    monkeypatch.setattr(api, "resolve_doc_id", lambda fragment: None)
    return TestClient(api.app)


# --- the stages the pipeline reports ---------------------------------------

def test_every_stage_is_reported(indexed_client, monkeypatch):
    """A caller should be able to narrate the whole pipeline."""
    monkeypatch.setattr("app.answer.condense", lambda question, turns: question)
    monkeypatch.setattr(
        "app.answer.complete",
        lambda system, user, model=None, temperature=None: json.dumps(
            {"answer": "ok", "citations": [], "confidence": "high"}
        ),
    )
    seen = []
    answer_question("confidentiality obligations", top_k=8, client=indexed_client, progress=seen.append)

    stages = [event["stage"] for event in seen]
    assert stages == ["retrieving", "retrieved", "generating", "verifying"]


def test_the_retrieved_event_carries_something_worth_showing(indexed_client, monkeypatch):
    monkeypatch.setattr("app.answer.condense", lambda question, turns: question)
    monkeypatch.setattr(
        "app.answer.complete",
        lambda system, user, model=None, temperature=None: json.dumps(
            {"answer": "ok", "citations": [], "confidence": "high"}
        ),
    )
    seen = []
    answer_question("force majeure", top_k=8, client=indexed_client, progress=seen.append)

    retrieved = next(event for event in seen if event["stage"] == "retrieved")
    assert retrieved["count"] == 8
    assert retrieved["documents"]
    assert len(retrieved["top"]) == 3
    assert {"rank", "doc_title", "section"} <= set(retrieved["top"][0])


def test_progress_is_optional(indexed_client, monkeypatch):
    """Every existing caller passes nothing and must be unaffected."""
    monkeypatch.setattr("app.answer.condense", lambda question, turns: question)
    monkeypatch.setattr(
        "app.answer.complete",
        lambda system, user, model=None, temperature=None: json.dumps(
            {"answer": "ok", "citations": [], "confidence": "high"}
        ),
    )
    result = answer_question("confidentiality", top_k=3, client=indexed_client)
    assert result.answer == "ok"


# --- the endpoint ----------------------------------------------------------

def test_the_stream_ends_with_the_whole_answer(stubbed_api, monkeypatch):
    def fake(question, progress=None, **kwargs):
        progress({"stage": "retrieving"})
        progress({"stage": "retrieved", "count": 8, "documents": ["A"], "top": []})
        return AnswerResult(question, "the answer", [], "high", [])

    monkeypatch.setattr(api, "answer_question", fake)
    response = stubbed_api.post("/ask/stream", json={"question": "what are the terms?"})

    assert response.status_code == 200
    assert "x-ndjson" in response.headers["content-type"]
    events = read_events(response)
    assert [e["stage"] for e in events[:-1]] == ["retrieving", "retrieved"]
    assert events[-1]["stage"] == "done"
    assert events[-1]["result"]["answer"] == "the answer"


def test_a_failure_arrives_as_an_event_not_a_dropped_connection(stubbed_api, monkeypatch):
    def explode(question, progress=None, **kwargs):
        progress({"stage": "retrieving"})
        raise RuntimeError("gateway down")

    monkeypatch.setattr(api, "answer_question", explode)
    response = stubbed_api.post("/ask/stream", json={"question": "what are the terms?"})

    events = read_events(response)
    assert events[-1]["stage"] == "error"
    assert "gateway down" in events[-1]["detail"]


def test_an_empty_index_is_still_a_plain_error(stubbed_api, monkeypatch):
    monkeypatch.setattr(api, "collection_size", lambda *a, **k: 0)
    response = stubbed_api.post("/ask/stream", json={"question": "what are the terms?"})
    assert response.status_code == 503


def test_the_non_streaming_endpoint_still_exists():
    """Both are published: the plain one is what the CLIs and any API client
    should use."""
    paths = api.app.openapi()["paths"]
    assert "/ask" in paths and "/ask/stream" in paths


def test_the_page_consumes_the_stream():
    html = TestClient(api.app).get("/").text
    assert "/ask/stream" in html
    assert "getReader" in html


# --- the page's presentation of it -----------------------------------------

def test_configuration_is_hidden_unless_debug_is_on():
    """Similarity scores, retrieval mode and model names are diagnostics. The
    person waiting for an answer wants to know what is happening, not how it
    is configured."""
    html = TestClient(api.app).get("/").text
    assert 'const technical = $("debug").checked' in html
    # the score and the mode are both behind that flag
    assert 'technical ? `${event.mode}, top ${event.top_k}` : ""' in html
    assert 'technical ? String(event.score) : ""' in html


def test_the_answer_is_revealed_not_streamed():
    """The reveal is presentation. The text is complete and its citations are
    verified before a single word is shown - which is the difference between
    this and token streaming, and the reason it was built this way."""
    html = TestClient(api.app).get("/").text
    assert "async function reveal" in html
    assert "REDUCED_MOTION" in html          # honours the accessibility setting
    assert "await settle()" in html          # progress finishes before the answer starts


def test_citations_wait_for_the_prose_to_finish():
    html = TestClient(api.app).get("/").text
    assert "after-answer" in html
