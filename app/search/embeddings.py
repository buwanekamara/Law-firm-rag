"""Embedding models, loaded once and shared.

Two different models produce two different views of the same chunk:

- a *dense* model (bge-small-en-v1.5) turns text into 384 numbers that capture
  meaning, so "can we end the contract early?" finds a clause about
  termination even though neither word appears in it;
- a *sparse* model (BM25) scores exact word overlap, so a defined term like
  "Transporter" or "the Brand" lands on the clause that actually uses that
  word.

Legal text needs both. Dense retrieval alone misses defined terms, which in a
contract carry precise meanings that a paraphrase does not preserve. Keyword
search alone misses every question a person phrases in their own words.

Both models run locally on CPU. Nothing is sent anywhere to embed a query.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache

from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.sparse.sparse_embedding_base import SparseEmbedding

from app.config import settings

# bge-small-en-v1.5 produces vectors of this length. Qdrant needs to know it
# up front, when the collection is created.
DENSE_DIMENSIONS = 384


@lru_cache(maxsize=1)
def dense_model() -> TextEmbedding:
    """Loaded lazily: the first call downloads ~130MB and takes a few seconds."""
    return TextEmbedding(model_name=settings.dense_model)


@lru_cache(maxsize=1)
def sparse_model() -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name=settings.sparse_model)


def embed_documents(texts: Iterable[str]) -> tuple[list[list[float]], list[SparseEmbedding]]:
    """Embed chunk text for indexing."""
    texts = list(texts)
    dense = [vector.tolist() for vector in dense_model().embed(texts)]
    sparse = list(sparse_model().embed(texts))
    return dense, sparse


def embed_query(text: str) -> tuple[list[float], SparseEmbedding]:
    """Embed a question.

    BM25 deliberately embeds queries differently from documents - query terms
    are not length-normalised - which is why this calls query_embed rather
    than reusing embed_documents.
    """
    dense = next(iter(dense_model().query_embed(text))).tolist()
    sparse = next(iter(sparse_model().query_embed(text)))
    return dense, sparse
