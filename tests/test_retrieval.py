"""Retrieval tests - the plumbing, not the ranking quality.

Ranking quality is a measurement, not an assertion; it lives in
eval/retrieval_eval.py and is run against the real models.
"""

import pytest
from app.config import settings
from app.search.indexing import (
    DENSE_VECTOR,
    SPARSE_VECTOR,
    collection_size,
    index_chunks,
    point_id,
    text_for_embedding,
)
from app.search.retrieval import infer_doc_filter, list_indexed_documents, search
from qdrant_client import models


def test_every_chunk_is_indexed(indexed_client):
    assert collection_size(indexed_client) == 120


def test_collection_carries_both_vector_types(indexed_client):
    info = indexed_client.get_collection(settings.qdrant_collection)
    assert DENSE_VECTOR in info.config.params.vectors
    assert SPARSE_VECTOR in info.config.params.sparse_vectors


def test_sparse_vectors_use_idf(indexed_client):
    """BM25 without the IDF modifier is just term counting - every occurrence
    of "Agreement" would weigh as much as "Transporter"."""
    info = indexed_client.get_collection(settings.qdrant_collection)
    assert info.config.params.sparse_vectors[SPARSE_VECTOR].modifier == models.Modifier.IDF


def test_reindexing_does_not_duplicate(indexed_client, stub_embeddings):
    """Point ids are derived from chunk ids, so a rerun overwrites."""
    from app.ingest.chunking import chunk_all

    before = collection_size(indexed_client)
    index_chunks(chunk_all(save=False), client=indexed_client, recreate=False)
    assert collection_size(indexed_client) == before


def test_point_id_is_stable():
    assert point_id("hosting_agreement::2::1") == point_id("hosting_agreement::2::1")
    assert point_id("hosting_agreement::2::1") != point_id("hosting_agreement::2::2")


# --- what gets embedded ----------------------------------------------------

def test_embedded_text_includes_provenance_and_parent():
    chunk = {
        "doc_title": "Trademark License Agreement",
        "section_label": "Section 4.3",
        "section_heading": "Termination for Breach",
        "parent_label": "Section 4",
        "parent_heading": "Termination",
        "page_start": 2,
        "page_end": 2,
        "text": "If either party materially breaches...",
    }
    embedded = text_for_embedding(chunk)
    assert "Trademark License Agreement" in embedded
    assert "Section 4.3" in embedded
    assert "Termination" in embedded  # the parent heading, which has no chunk of its own
    assert "materially breaches" in embedded


# --- document filtering ----------------------------------------------------

def test_all_documents_are_discoverable(indexed_client):
    assert set(list_indexed_documents(indexed_client)) == {
        "hosting_agreement",
        "joint_venture_agreement",
        "manufacturing_agreement",
        "trademark_license_agreement",
        "transportation_agreement",
    }


def test_question_naming_one_contract_is_filtered(indexed_client):
    documents = list_indexed_documents(indexed_client)
    assert infer_doc_filter("in the hosting agreement, who pays?", documents) == "hosting_agreement"


def test_question_naming_two_contracts_is_not_filtered(indexed_client):
    """A comparison must see both documents; filtering to one would delete
    half the answer."""
    documents = list_indexed_documents(indexed_client)
    assert infer_doc_filter("compare termination in hosting and manufacturing", documents) is None


def test_explicit_filter_restricts_results(indexed_client):
    results = search("payment", top_k=5, doc_id="hosting_agreement", client=indexed_client)
    assert results
    assert {r.chunk["doc_id"] for r in results} == {"hosting_agreement"}


# --- fusion ----------------------------------------------------------------

@pytest.mark.parametrize("mode", ["hybrid", "dense", "sparse"])
def test_every_mode_returns_ranked_results(indexed_client, mode):
    results = search("termination for breach", top_k=5, client=indexed_client, mode=mode)
    assert len(results) == 5
    assert [r.rank for r in results] == [1, 2, 3, 4, 5]
    assert all(r.score is not None for r in results)


def test_unknown_mode_is_rejected(indexed_client):
    with pytest.raises(ValueError):
        search("anything", client=indexed_client, mode="magic")


def test_results_carry_a_citation(indexed_client):
    result = search("force majeure", top_k=1, client=indexed_client)[0]
    citation = result.citation
    assert citation["doc_title"]
    assert citation["section"]
    assert citation["page_start"] >= 1
