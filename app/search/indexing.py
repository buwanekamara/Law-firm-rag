"""Building the Qdrant index.

Each chunk is one point carrying two named vectors, "dense" and "sparse".
Keeping both on the same point lets Qdrant run and fuse the two searches in
one round trip instead of merging two result lists by hand.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models

from app.config import PROJECT_ROOT, settings
from app.ingest.chunking import citation_header, load_chunks
from app.search.embeddings import DENSE_DIMENSIONS, embed_documents

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"

# Batching keeps memory flat and gives the CLI progress to report.
BATCH_SIZE = settings.embed_batch_size


def get_client(url: str | None = None, path: str | None = None) -> QdrantClient:
    """Connect to Qdrant: a server, embedded on disk, or in memory.

    QDRANT_PATH runs the engine inside this process - no Docker, same query
    API, but the storage folder takes an exclusive lock, so the API server and
    a CLI script cannot both hold it.

    ":memory:" is checked first so a local QDRANT_PATH cannot leak into tests.
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
    """One line naming where the index lives, for the CLIs."""
    if settings.qdrant_url == ":memory:":
        return "in-memory Qdrant (nothing is persisted)"
    if settings.qdrant_path:
        return f"embedded Qdrant at {settings.qdrant_path}"
    return f"Qdrant server at {settings.qdrant_url}"


def point_id(chunk_id: str) -> str:
    """A stable UUID for a chunk.

    Qdrant ids must be ints or UUIDs; ours are strings like
    "hosting_agreement::2::1". Deriving it means re-indexing overwrites rather
    than duplicates.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def text_for_embedding(chunk: dict[str, Any]) -> str:
    """What actually gets embedded.

    The citation header goes in front so the title, section number and parent
    heading are searchable. Without it "what does Article X say" has nothing
    to match, and group headings that live only in metadata are invisible.
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
            # IDF weights rare words above common ones - the "inverse
            # document frequency" half of BM25. Without it these are raw term
            # counts and "Agreement" counts as much as "Transporter".
            SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
    )


def build_points(chunks: list[dict[str, Any]]) -> Iterable[models.PointStruct]:
    """Embed chunks in batches and yield Qdrant points."""
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        dense_vectors, sparse_vectors = embed_documents([text_for_embedding(c) for c in batch])
        for chunk, dense, sparse in zip(batch, dense_vectors, sparse_vectors, strict=True):
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
