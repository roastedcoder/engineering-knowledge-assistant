import pytest
from app.models import Chunk
from app.retrieval.hybrid_retriever import HybridRetriever
from unittest.mock import Mock


def test_rrf_combines_two_ranked_lists():
    chunk_a = Chunk("Redis database", {"source": "redis.md"})
    chunk_b = Chunk("MySQL database", {"source": "mysql.md"})
    chunk_c = Chunk("Kafka messaging", {"source": "kafka.md"})

    faiss_results = [
        {"chunk": chunk_a, "score": 0.95},
        {"chunk": chunk_b, "score": 0.85},
    ]

    bm25_results = [
        {"chunk": chunk_b, "score": 4.5},
        {"chunk": chunk_c, "score": 3.5},
    ]

    retriever = HybridRetriever(
        faiss_store=Mock(),
        bm25_store=Mock(),
    )

    results = retriever.reciprocal_rank_fusion(
        faiss_results,
        bm25_results,
    )

    assert len(results) == 3
    assert results[0]["chunk"].content == "MySQL database"

def test_rrf_accumulates_scores_for_duplicate_chunks():
    chunk_a = Chunk("Redis database", {"source": "redis.md"})
    chunk_b = Chunk("MySQL database", {"source": "mysql.md"})

    faiss_results = [
        {"chunk": chunk_a, "score": 0.95},
        {"chunk": chunk_b, "score": 0.85},
    ]

    bm25_results = [
        {"chunk": chunk_a, "score": 4.5},
    ]

    retriever = HybridRetriever(
        faiss_store=Mock(),
        bm25_store=Mock(),
    )

    results = retriever.reciprocal_rank_fusion(
        faiss_results,
        bm25_results,
    )

    assert len(results) == 2
    assert results[0]["chunk"].content == "Redis database"
    assert results[0]["score"] == pytest.approx(
        (1 / 61) + (1 / 61)
    )

def test_rrf_handles_empty_ranked_lists():
    retriever = HybridRetriever(
        faiss_store=Mock(),
        bm25_store=Mock(),
    )

    results = retriever.reciprocal_rank_fusion([], [])

    assert results == []

@pytest.mark.parametrize("rrf_k", [0, -1])
def test_rrf_rejects_invalid_k(rrf_k):
    with pytest.raises(
        ValueError,
        match="rrf_k must be greater than 0",
    ):
        HybridRetriever(
            faiss_store=Mock(),
            bm25_store=Mock(),
            rrf_k=rrf_k,
        )

def test_search_calls_both_retrievers():
    chunk_a = Chunk("Redis database", {"source": "redis.md"})
    chunk_b = Chunk("MySQL database", {"source": "mysql.md"})

    faiss_store = Mock()
    bm25_store = Mock()

    faiss_store.search.return_value = [
        {"chunk": chunk_a, "score": 0.95},
    ]

    bm25_store.search.return_value = [
        {"chunk": chunk_b, "score": 4.5},
        {"chunk": chunk_a, "score": 3.5},
    ]

    retriever = HybridRetriever(faiss_store, bm25_store)

    results = retriever.search(
        query="Redis",
        query_embedding=[0.1] * 384,
        top_k=3,
    )

    faiss_store.search.assert_called_once_with([0.1] * 384, 3)
    bm25_store.search.assert_called_once_with("Redis", 3)

    assert len(results) == 2
    assert results[0]["chunk"].content == "Redis database"