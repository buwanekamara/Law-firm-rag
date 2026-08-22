"""FastAPI application.

Two endpoints: /health for configuration and liveness, /ask for questions.
The auto-generated documentation at /docs is the interface for now - it is a
usable demo on its own, so a hand-written UI can wait until everything else
works.
"""

import json
import queue
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.answer import answer_question
from app.config import list_contracts, settings
from app.generate.conversation import Turn
from app.generate.llm import MissingApiKey
from app.search.indexing import backend_description, collection_size, get_client
from app.search.retrieval import list_indexed_documents

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="Contract RAG",
    version="0.1.0",
    description="Ask questions about a small corpus of contracts and get "
    "answers cited down to the specific section.",
)


class HistoryTurn(BaseModel):
    question: str = Field(max_length=2000)
    answer: str = Field(default="", max_length=4000)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, examples=["What are the confidentiality obligations?"])
    history: list[HistoryTurn] = Field(
        default_factory=list,
        max_length=20,
        description="Previous turns, oldest first. The client keeps the conversation; "
        "the server stores nothing. A follow-up that refers back to an earlier turn is "
        "rewritten into a standalone question before retrieval.",
    )
    top_k: int | None = Field(default=None, ge=1, le=20)
    doc_id: str | None = Field(
        default=None,
        examples=["hosting_agreement"],
        description="Restrict the search to one contract; a fragment such as 'hosting' is "
        "enough. Left empty, a question that names a contract is narrowed to it "
        "automatically.",
    )


def resolve_doc_id(fragment: str | None) -> str | None:
    """Turn a document fragment into a real doc_id, or reject it.

    An unknown filter must be an error rather than an empty result set. The
    interactive docs page pre-fills optional strings with the word "string",
    and silently answering "no relevant clause was found" because of that
    looks like a corpus problem when it is a typo.
    """
    if not fragment:
        return None
    documents = list_indexed_documents(get_client())
    matches = [doc_id for doc_id in documents if fragment.lower() in doc_id.lower()]
    if len(matches) == 1:
        return matches[0]
    raise HTTPException(
        status_code=400,
        detail={
            "message": f"{fragment!r} matches {len(matches)} indexed documents; expected exactly one.",
            "known_documents": sorted(documents),
        },
    )


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """The tester page - one static file, no build step, no separate service."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/documents")
def documents() -> list[dict]:
    """The contracts currently indexed. The page uses this for its filter."""
    if collection_size() == 0:
        return []
    return [
        {"doc_id": doc_id, "title": title}
        for doc_id, title in sorted(list_indexed_documents(get_client()).items())
    ]


@app.get("/health")
def health() -> dict:
    """Liveness check plus a readable dump of the active configuration.

    Useful beyond 'is it up': it confirms the app found the contracts folder
    and picked up the environment you think it did. The API key is never
    returned - only whether one is present.
    """
    contracts = list_contracts()
    return {
        "status": "ok",
        "contracts_dir": str(settings.contracts_dir),
        "contracts_found": len(contracts),
        "contracts": [p.name for p in contracts],
        "llm_model": settings.llm_model,
        "judge_model": settings.judge_model,
        "prompt_version": settings.prompt_version,
        "vector_store": backend_description(),
        "search_mode": settings.search_mode,
        "top_k": settings.top_k,
        "relevance_gate": settings.min_score or "off",
        "qdrant_collection": settings.qdrant_collection,
        "indexed_chunks": collection_size(),
        "gateway_key_configured": bool(settings.ai_gateway_api_key),
        "masking_enabled": settings.masking_enabled,
    }


@app.post("/ask/stream")
def ask_stream(request: AskRequest, debug: bool = Query(default=False)) -> StreamingResponse:
    """The same answer, with the pipeline's progress reported as it happens.

    Newline-delimited JSON rather than server-sent events, because the request
    carries a body and EventSource can only issue GETs.

    The *answer* is not streamed - it arrives whole, in the final event.
    Citations are verified and masked names restored only once the model's
    reply is complete, and streaming the prose would put text on screen that
    nothing had checked yet. What is streamed is what the system is doing
    during the seconds that takes, which is the part a person is waiting on.
    """
    if collection_size() == 0:
        raise HTTPException(status_code=503, detail="No chunks are indexed. Run: uv run scripts/index.py")
    doc_id = resolve_doc_id(request.doc_id)

    events: queue.Queue = queue.Queue()
    DONE = object()

    def work() -> None:
        try:
            result = answer_question(
                request.question,
                top_k=request.top_k,
                doc_id=doc_id,
                debug=debug,
                history=[Turn(question=t.question, answer=t.answer) for t in request.history],
                progress=events.put,
            )
            events.put({"stage": "done", "result": result.to_dict()})
        except MissingApiKey as error:
            events.put({"stage": "error", "detail": str(error)})
        except Exception as error:  # the client needs to hear about it, whatever it was
            events.put({"stage": "error", "detail": f"{type(error).__name__}: {error}"})
        finally:
            events.put(DONE)

    def stream():
        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        while True:
            event = events.get()
            if event is DONE:
                return
            yield json.dumps(event) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.post("/ask")
def ask(
    request: AskRequest,
    debug: bool = Query(
        default=False,
        description="Also return the retrieved chunks and the exact prompt that was sent.",
    ),
) -> dict:
    """Answer a question from the indexed contracts.

    `debug=true` returns the retrieved excerpts and the rendered prompt
    alongside the answer. That is the difference between "the answer is wrong"
    and knowing which stage made it wrong.
    """
    if collection_size() == 0:
        raise HTTPException(
            status_code=503,
            detail="No chunks are indexed. Run: uv run scripts/index.py",
        )
    doc_id = resolve_doc_id(request.doc_id)
    try:
        result = answer_question(
            request.question,
            top_k=request.top_k,
            doc_id=doc_id,
            debug=debug,
            history=[Turn(question=t.question, answer=t.answer) for t in request.history],
        )
    except MissingApiKey as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    return result.to_dict()
