"""Phase 3b - hybrid retrieval with server-side fusion.

Two searches run against the same collection: one over dense vectors
(meaning), one over sparse BM25 vectors (exact wording). Their results are
combined by Reciprocal Rank Fusion.

Why RRF rather than adding the scores together: cosine similarity and BM25
live on different scales, and BM25's range shifts with the corpus. Adding them
means inventing a weight and defending it. RRF ignores the scores and uses
only each result's *rank* in each list - a chunk placed first by either search
contributes 1/(60+1), one placed fifth contributes 1/(60+5). Anything ranked
highly by both rises to the top, and nothing has to be normalised.

Qdrant performs the fusion itself, so this is one network round trip rather
than two searches merged in Python.

Measured outcome. On a first set of 17 straightforward questions, fusion
showed no benefit: dense retrieval alone placed the correct section first on
16 of 17 and hybrid on 15, so the default was set to dense. Adding three
harder questions reversed that. Asked for a redacted price and for a
placeholder effective date, dense retrieval missed both clauses entirely -
outside the top eight - while BM25 found them at ranks 3 and 1, and fusion at
5 and 2. Hybrid is now the only mode that puts a correct section in the top
five for every scored question.

The failure is structural rather than bad luck. A dense vector averages a
whole chunk into one point, so a single fact inside a long heterogeneous
chunk - the trademark preamble covers the parties, the brand and four
recitals - is diluted by everything around it. BM25 does not average: one
exact term match still scores. Neither half is reliable alone on contract
text, which is the actual argument for fusing them.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient, models

from app.config import settings
from app.embeddings import embed_query
from app.indexing import DENSE_VECTOR, SPARSE_VECTOR, get_client

# Each branch of the hybrid search fetches this multiple of the requested
# result count before fusion. Fusing two top-5 lists gives the ranking very
# little to work with; fusing two top-20 lists lets a chunk that placed 12th
# on one side and 2nd on the other still surface.
PREFETCH_MULTIPLIER = 4

# The configured default is "dense" (see the module docstring). All three
# remain available so eval/retrieval_eval.py can compare them.
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

    Only applied when exactly one document matches. A question mentioning two
    contracts is a comparison, and filtering it to one of them would quietly
    remove half the answer.
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

    `mode` defaults to the configured SEARCH_MODE. Pass it explicitly to
    compare retrieval strategies - that is what the evaluation does.
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

    This is the signal the refusal gate uses, and it is deliberately *not*
    the fused hybrid score. Reciprocal rank fusion scores a result by how
    highly the two searches ranked it, not by how close it is to the
    question - so the top result of a nonsense query scores about as well as
    the top result of a good one, because something always has to come first.
    Cosine similarity against the dense vector is an actual measure of
    closeness, which is what a "do we have anything relevant at all?" check
    needs.

    Costs one extra query, which only happens when gating is switched on.
    """
    results = search(question, top_k=1, doc_id=doc_id, client=client, mode="dense")
    return results[0].score if results else 0.0
