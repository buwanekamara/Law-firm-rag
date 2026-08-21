"""Shared test fixtures.

The retrieval tests need vectors but not *good* vectors: what they check is
that the collection is built correctly, that fusion returns ranked results,
and that filters do what they claim. Downloading a 130MB embedding model to
prove that would make the suite slow and network-dependent, so the models are
replaced with deterministic stand-ins that hash words into vectors.

Retrieval *quality* is measured separately, by eval/retrieval_eval.py, against
the real models and a real Qdrant.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter

import numpy as np
import pytest

DIMENSIONS = 384


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _hash(word: str, buckets: int) -> int:
    return int(hashlib.md5(word.encode()).hexdigest(), 16) % buckets


def fake_dense(text: str) -> list[float]:
    """Bag of words hashed into a fixed-width vector, then normalised."""
    vector = np.zeros(DIMENSIONS)
    for word in _tokens(text):
        vector[_hash(word, DIMENSIONS)] += 1.0
    norm = np.linalg.norm(vector)
    return (vector / norm if norm else vector).tolist()


class FakeSparse:
    def __init__(self, indices, values):
        self.indices = np.array(indices)
        self.values = np.array(values)


def fake_sparse(text: str) -> FakeSparse:
    counts = Counter(_hash(word, 100_000) for word in _tokens(text))
    return FakeSparse(list(counts), [float(count) for count in counts.values()])


@pytest.fixture(autouse=True, scope="session")
def hermetic_settings():
    """Run the suite against known settings, not the developer's .env.

    Found the hard way: a local MIN_SCORE=0.56 made the relevance gate fire in
    every test, because the stand-in embedder below produces similarities
    around 0.11. Ten tests failed on one machine and passed on another. A test
    that depends on someone's local configuration is not a test, so the
    values that change behaviour are pinned here and restored afterwards.

    Individual tests still override these with monkeypatch when the setting is
    what they are testing.
    """
    from app.config import settings

    pinned = {
        "min_score": 0.0,
        "citation_retry": True,
        "top_k": 8,
        "search_mode": "hybrid",
        "prompt_version": "v1",
    }
    saved = {name: getattr(settings, name) for name in pinned}
    for name, value in pinned.items():
        setattr(settings, name, value)
    yield
    for name, value in saved.items():
        setattr(settings, name, value)


@pytest.fixture(scope="session")
def stub_embeddings(session_mocker=None):
    """Patch the two embedding entry points for the whole session."""
    import app.indexing as indexing
    import app.retrieval as retrieval

    original_documents = indexing.embed_documents
    original_query = retrieval.embed_query

    indexing.embed_documents = lambda texts: (
        [fake_dense(text) for text in texts],
        [fake_sparse(text) for text in texts],
    )
    retrieval.embed_query = lambda text: (fake_dense(text), fake_sparse(text))
    yield
    indexing.embed_documents = original_documents
    retrieval.embed_query = original_query


@pytest.fixture(scope="session")
def indexed_client(stub_embeddings):
    """An in-process Qdrant holding every chunk. No Docker, no network."""
    from app.chunking import chunk_all
    from app.indexing import get_client, index_chunks

    client = get_client(":memory:")
    index_chunks(chunk_all(save=False), client=client)
    return client
