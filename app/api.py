"""FastAPI application.

Two endpoints: /health for configuration and liveness, /ask for questions.
The auto-generated documentation at /docs is the interface for now - it is a
usable demo on its own, so a hand-written UI can wait until everything else
works.
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.answer import answer_question
from app.config import list_contracts, settings
from app.indexing import backend_description, collection_size, get_client
from app.llm import MissingApiKey
from app.retrieval import list_indexed_documents

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="Contract RAG",
    version="0.1.0",
    description="Ask questions about a small corpus of contracts and get "
    "answers cited down to the specific section.",
)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, examples=["What are the confidentiality obligations?"])
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
            request.question, top_k=request.top_k, doc_id=doc_id, debug=debug
        )
    except MissingApiKey as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    return result.to_dict()
