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

import os

# Every setting is required and read from the environment, so the suite
# supplies its own. This runs before any app module is imported, and takes
# precedence over a developer's .env - which is the point: a test that depends
# on someone's local configuration is not a test. Values that change
# behaviour are pinned again in hermetic_settings below.
TEST_ENVIRONMENT = {
    "AI_GATEWAY_API_KEY": "test-key-not-used",
    "AI_GATEWAY_BASE_URL": "https://ai-gateway.invalid/v1",
    "LLM_MODEL": "test/model",
    "JUDGE_MODEL": "test/judge",
    "LLM_TEMPERATURE": "0",
    "QDRANT_URL": "http://localhost:6333",
    "QDRANT_PATH": "",
    "QDRANT_COLLECTION": "contract_chunks",
    "DENSE_MODEL": "BAAI/bge-small-en-v1.5",
    "SPARSE_MODEL": "Qdrant/bm25",
    "SEARCH_MODE": "hybrid",
    "TOP_K": "8",
    "PREFETCH_MULTIPLIER": "4",
    "MIN_SCORE": "0.0",
    "CITATION_RETRY": "true",
    "CHUNK_MAX_WORDS": "350",
    "CHUNK_OVERLAP_WORDS": "50",
    "EMBED_BATCH_SIZE": "32",
    "HISTORY_TURNS": "6",
    "LLM_TIMEOUT_SECONDS": "90",
    "PROMPT_VERSION": "v1",
    "MASKING_ENABLED": "false",
    "MASKING_MIN_CONFIDENCE": "0.6",
}
os.environ.update(TEST_ENVIRONMENT)

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


@pytest.fixture(scope="session")
def env_template() -> dict[str, str]:
    """.env.example, parsed into a mapping.

    With no fallback values in code, this file is not documentation - it is
    the complete description of what a deployment must supply, and the values
    a deployment starts from. Tests assert against it for that reason.
    """
    from app.config import PROJECT_ROOT

    values: dict[str, str] = {}
    for line in (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        values[name.strip()] = value.strip()
    return values


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
    import app.search.indexing as indexing
    import app.search.retrieval as retrieval

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
    from app.ingest.chunking import chunk_all
    from app.search.indexing import get_client, index_chunks

    client = get_client(":memory:")
    index_chunks(chunk_all(save=False), client=client)
    return client
