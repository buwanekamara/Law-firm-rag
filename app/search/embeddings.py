"""Embedding models, loaded once and shared.

Two views of the same chunk: a dense model for meaning ("end the contract
early" finds a termination clause), BM25 for exact words (defined terms like
"Transporter" carry precise meanings a paraphrase loses). Contracts need both.

Both run locally on CPU.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache

from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.sparse.sparse_embedding_base import SparseEmbedding

from app.config import settings

# bge-small-en-v1.5 vector length. Qdrant needs it when the collection is made.
DENSE_DIMENSIONS = 384


@lru_cache(maxsize=1)
def dense_model() -> TextEmbedding:
    """Lazy: the first call downloads ~130MB."""
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

    query_embed, not embed_documents: BM25 does not length-normalise query
    terms the way it does document terms.
    """
    dense = next(iter(dense_model().query_embed(text))).tolist()
    sparse = next(iter(sparse_model().query_embed(text)))
    return dense, sparse
