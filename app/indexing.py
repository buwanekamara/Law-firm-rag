"""Phase 3a - building the Qdrant index.

Every chunk becomes one point carrying two vectors under different names:
"dense" for meaning and "sparse" for exact wording. Storing both on the same
point is what lets Qdrant run the two searches and fuse them in a single
round trip, instead of the application making two calls and merging the
results itself with hand-rolled score maths.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Iterable

from qdrant_client import QdrantClient, models

from app.chunking import citation_header, load_chunks
from app.config import PROJECT_ROOT, settings
from app.embeddings import DENSE_DIMENSIONS, embed_documents

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"

# Embedding in batches keeps memory flat and gives the CLI something to report.
BATCH_SIZE = 32


def get_client(url: str | None = None, path: str | None = None) -> QdrantClient:
    """Connect to Qdrant, in one of three ways.

    1. A server, the normal case: QDRANT_URL=http://localhost:6333.
    2. Embedded on disk: set QDRANT_PATH=./qdrant_local and the same Qdrant
       engine runs inside this Python process, storing to that folder. No
       Docker, no server, same query API - including the fusion queries. The
       catch is that the storage folder takes an exclusive lock, so only one
       process may hold it at a time: the API server and a CLI script cannot
       both be open on it at once.
    3. In memory: QDRANT_URL=":memory:", used by the tests.

    Precedence is deliberate - ":memory:" wins so a developer's QDRANT_PATH
    setting can never leak into a test run.
    """
    url = url or settings.qdrant_url
    if url == ":memory:":
        return QdrantClient(":memory:")

    path = path if path is not None else settings.qdrant_path
    if path:
        storage = Path(path)
        if not storage.is_absolute():
            storage = PROJECT_ROOT / storage
        try:
            return QdrantClient(path=str(storage))
        except RuntimeError as error:
            raise RuntimeError(
                f"Cannot open the embedded Qdrant storage at {storage}.\n"
                "It is already open in another process - stop the API server "
                "or the other script and try again.\n"
                "(Embedded mode allows a single reader/writer. Switch to the "
                "Docker service if you need both at once.)"
            ) from error

    return QdrantClient(url=url)


def backend_description() -> str:
    """One line naming where the index lives, for the CLIs to print."""
    if settings.qdrant_url == ":memory:":
        return "in-memory Qdrant (nothing is persisted)"
    if settings.qdrant_path:
        return f"embedded Qdrant at {settings.qdrant_path}"
    return f"Qdrant server at {settings.qdrant_url}"


def point_id(chunk_id: str) -> str:
    """A stable UUID for a chunk.

    Qdrant point ids must be integers or UUIDs, and our chunk ids are strings
    like "hosting_agreement::2::1". Deriving the UUID from the chunk id means
    re-indexing the same chunk overwrites its old point rather than adding a
    duplicate.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def text_for_embedding(chunk: dict[str, Any]) -> str:
    """What actually gets embedded.

    The citation header is prepended so the document title, section number and
    parent heading are part of the searchable text. Without it, a question
    naming a section ("what does Article X say") has nothing to match, and the
    trademark licence's group headings - "Grant of Rights; Sublicensing",
    which live in metadata rather than in any chunk body - would be invisible
    to search.
    """
    header = citation_header(chunk)
    parent = f"{chunk['parent_label']} {chunk['parent_heading']}" if chunk["parent_label"] else ""
    return "\n".join(part for part in (header, parent, chunk["text"]) if part)


def ensure_collection(client: QdrantClient, recreate: bool = False) -> None:
    """Create the collection if it is missing."""
    exists = client.collection_exists(settings.qdrant_collection)
    if exists and not recreate:
        return
    if exists:
        client.delete_collection(settings.qdrant_collection)

    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config={
            DENSE_VECTOR: models.VectorParams(
                size=DENSE_DIMENSIONS, distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config={
            # IDF tells Qdrant to weight rare words more heavily than common
            # ones, which is the "inverse document frequency" half of BM25.
            # Without this modifier the sparse vectors are raw term counts and
            # every occurrence of "Agreement" counts as much as "Transporter".
            SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
    )


def build_points(chunks: list[dict[str, Any]]) -> Iterable[models.PointStruct]:
    """Embed chunks in batches and yield Qdrant points."""
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        dense_vectors, sparse_vectors = embed_documents([text_for_embedding(c) for c in batch])
        for chunk, dense, sparse in zip(batch, dense_vectors, sparse_vectors):
            yield models.PointStruct(
                id=point_id(chunk["chunk_id"]),
                vector={
                    DENSE_VECTOR: dense,
                    SPARSE_VECTOR: models.SparseVector(
                        indices=sparse.indices.tolist(), values=sparse.values.tolist()
                    ),
                },
                payload=chunk,
            )


def index_chunks(
    chunks: list[dict[str, Any]] | None = None,
    client: QdrantClient | None = None,
    recreate: bool = True,
) -> int:
    """Index every chunk. Returns how many points were written."""
    client = client or get_client()
    chunks = chunks if chunks is not None else load_chunks()
    ensure_collection(client, recreate=recreate)

    points = list(build_points(chunks))
    client.upsert(collection_name=settings.qdrant_collection, points=points, wait=True)
    return len(points)


def collection_size(client: QdrantClient | None = None) -> int:
    client = client or get_client()
    if not client.collection_exists(settings.qdrant_collection):
        return 0
    return client.count(settings.qdrant_collection, exact=True).count
