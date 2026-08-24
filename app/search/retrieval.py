"""Hybrid retrieval with server-side fusion.

A dense search (meaning) and a BM25 search (exact wording) are combined by
Reciprocal Rank Fusion. RRF uses each result's rank rather than its score,
which matters because cosine similarity and BM25 are on different scales -
adding them would mean inventing a weight. Qdrant fuses server-side, so this
is one round trip.

Hybrid is the configured mode because dense search alone does not reliably
find clauses whose value is redacted or left as a placeholder: a dense vector
averages a whole chunk, so one fact inside a long one is diluted. BM25 does
not average, so an exact term match still scores. Numbers in
docs/retrieval-eval.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient, models

from app.config import settings
from app.search.embeddings import embed_query
from app.search.indexing import DENSE_VECTOR, SPARSE_VECTOR, get_client

# Each branch fetches this multiple of top_k before fusion, so a chunk that
# placed 12th on one side and 2nd on the other can still surface.
PREFETCH_MULTIPLIER = settings.prefetch_multiplier

# All three stay available so eval/retrieval_eval.py can compare them.
SEARCH_MODES = ("hybrid", "dense", "sparse")


@dataclass
class SearchResult:
    rank: int
    score: float
    chunk: dict[str, Any]

    @property
    def citation(self) -> dict[str, Any]:
        return {
            "doc_id": self.chunk["doc_id"],
            "doc_title": self.chunk["doc_title"],
            "section": self.chunk["section_label"],
            "heading": self.chunk["section_heading"],
            "page_start": self.chunk["page_start"],
            "page_end": self.chunk["page_end"],
        }


_document_cache: dict[int, dict[str, str]] = {}


def list_indexed_documents(client: QdrantClient) -> dict[str, str]:
    """Map of doc_id -> title for whatever is currently indexed."""
    key = id(client)
    if key in _document_cache:
        return _document_cache[key]

    documents: dict[str, str] = {}
    records, _ = client.scroll(
        collection_name=settings.qdrant_collection,
        limit=10_000,
        with_payload=["doc_id", "doc_title"],
        with_vectors=False,
    )
    for record in records:
        payload = record.payload or {}
        if payload.get("doc_id"):
            documents[payload["doc_id"]] = payload.get("doc_title", payload["doc_id"])
    _document_cache[key] = documents
    return documents


def infer_doc_filter(question: str, documents: dict[str, str]) -> str | None:
    """Spot a question that names one contract: "in the hosting agreement...".

    Only when exactly one matches - a question naming two is a comparison, and
    filtering to one would quietly remove half the answer.
    """
    asked = question.lower()
    matches = set()
    for doc_id, title in documents.items():
        phrase = title.lower().replace(" agreement", "").strip()
        if phrase and phrase in asked:
            matches.add(doc_id)
    return matches.pop() if len(matches) == 1 else None


def search(
    question: str,
    top_k: int | None = None,
    doc_id: str | None = None,
    auto_filter: bool = True,
    client: QdrantClient | None = None,
    mode: str | None = None,
) -> list[SearchResult]:
    """Retrieve the chunks most likely to answer `question`.

    `mode` defaults to SEARCH_MODE; the evaluation passes it explicitly.
    """
    mode = mode or settings.search_mode
    if mode not in SEARCH_MODES:
        raise ValueError(f"mode must be one of {SEARCH_MODES}, got {mode!r}")
    client = client or get_client()
    top_k = top_k or settings.top_k

    if doc_id is None and auto_filter:
        doc_id = infer_doc_filter(question, list_indexed_documents(client))

    query_filter = (
        models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])
        if doc_id
        else None
    )

    dense_vector, sparse_vector = embed_query(question)
    sparse_query = models.SparseVector(
        indices=sparse_vector.indices.tolist(), values=sparse_vector.values.tolist()
    )
    prefetch_limit = top_k * PREFETCH_MULTIPLIER

    common = {
        "collection_name": settings.qdrant_collection,
        "limit": top_k,
        "with_payload": True,
        "query_filter": query_filter,
    }

    if mode == "dense":
        response = client.query_points(query=dense_vector, using=DENSE_VECTOR, **common)
    elif mode == "sparse":
        response = client.query_points(query=sparse_query, using=SPARSE_VECTOR, **common)
    else:
        response = client.query_points(
            prefetch=[
                models.Prefetch(
                    query=dense_vector,
                    using=DENSE_VECTOR,
                    limit=prefetch_limit,
                    filter=query_filter,
                ),
                models.Prefetch(
                    query=sparse_query,
                    using=SPARSE_VECTOR,
                    limit=prefetch_limit,
                    filter=query_filter,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            **common,
        )

    return [
        SearchResult(rank=position, score=point.score, chunk=point.payload or {})
        for position, point in enumerate(response.points, start=1)
    ]


def best_similarity(
    question: str,
    doc_id: str | None = None,
    client: QdrantClient | None = None,
) -> float:
    """Cosine similarity between the question and the closest chunk.

    What the refusal gate uses, and deliberately not the fused score: RRF
    ranks by agreement between the two searches, so a nonsense query's top hit
    scores about as well as a good one's - something always comes first.
    Cosine similarity actually measures closeness.

    One extra query, only when gating is on.
    """
    results = search(question, top_k=1, doc_id=doc_id, client=client, mode="dense")
    return results[0].score if results else 0.0
