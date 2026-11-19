import pytest

from app.models import Chunk
from app.retrieval.bm25_store import BM25Store


@pytest.fixture
def sample_data():
    chunks = [
        Chunk("Redis database", {"source": "redis.md"}),
        Chunk("MySQL database", {"source": "mysql.md"}),
        Chunk("Kafka messaging", {"source": "kafka.md"}),
    ]

    return chunks

def test_search_empty_query():
    store = BM25Store()
    store_results = store.search("")
    assert store_results == []

def test_search_empty_index():
    store = BM25Store()

    results = store.search("Redis")

    assert results == []

def test_search_returns_matching_chunk(sample_data):
    store = BM25Store()
    store.add(sample_data)

    results = store.search("Redis")

    assert len(results) > 0
    assert results[0]["chunk"].content == "Redis database"

def test_search_rejects_invalid_top_k():
    store = BM25Store()

    with pytest.raises(ValueError, match="top_k must be greater than 0"):
        store.search("Redis", top_k=0)

def test_search_returns_empty_when_no_keyword_matches(sample_data):
    store = BM25Store()
    store.add(sample_data)

    results = store.search("PostgreSQL")

    assert results == []