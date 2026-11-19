import pytest

from app.models import Chunk
from app.retrieval.faiss_store import FAISSStore


@pytest.fixture
def sample_data():
    chunks = [
        Chunk("Redis database", {"source": "redis.md"}),
        Chunk("MySQL database", {"source": "mysql.md"}),
        Chunk("Kafka messaging", {"source": "kafka.md"}),
    ]

    embeddings = [
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.0, 0.0, 1.0],
    ]

    return chunks, embeddings


def test_add_and_search(sample_data):
    chunks, embeddings = sample_data

    store = FAISSStore(dimension=3)
    store.add(chunks, embeddings)

    results = store.search([1.0, 0.0, 0.0], top_k=2)

    assert len(results) == 2
    assert results[0]["chunk"].content == "Redis database"
    assert results[0]["score"] == pytest.approx(1.0)


def test_search_empty_index():
    store = FAISSStore(dimension=3)

    assert store.search([1.0, 0.0, 0.0]) == []


def test_top_k_larger_than_index(sample_data):
    chunks, embeddings = sample_data

    store = FAISSStore(dimension=3)
    store.add(chunks, embeddings)

    results = store.search([1.0, 0.0, 0.0], top_k=10)

    assert len(results) == 3


def test_reject_mismatched_chunks_and_embeddings():
    store = FAISSStore(dimension=3)

    chunks = [Chunk("Redis", {"source": "redis.md"})]
    embeddings = []

    with pytest.raises(ValueError):
        store.add(chunks, embeddings)


def test_reject_wrong_embedding_dimension(sample_data):
    chunks, _ = sample_data

    store = FAISSStore(dimension=3)

    with pytest.raises(ValueError):
        store.add(chunks, [[1.0, 0.0]])


def test_remove_by_source_removes_matching_chunks():
    store = FAISSStore(dimension=3)

    chunks = [
        Chunk("Redis one", {"source": "redis.md"}),
        Chunk("Kafka one", {"source": "kafka.pdf"}),
        Chunk("Redis two", {"source": "redis.md"}),
    ]

    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    store.add(chunks, embeddings)

    removed_count = store.remove_by_source("kafka.pdf")

    assert removed_count == 1
    assert store.index.ntotal == 2
    assert len(store.chunks) == 2

    assert [
        chunk.content for chunk in store.chunks
    ] == ["Redis one", "Redis two"]


def test_remove_by_source_returns_zero_for_missing_source():
    store = FAISSStore(dimension=3)

    chunks = [
        Chunk("Redis one", {"source": "redis.md"})
    ]

    store.add(chunks, [[1.0, 0.0, 0.0]])

    removed_count = store.remove_by_source("kafka.pdf")

    assert removed_count == 0
    assert store.index.ntotal == 1
    assert len(store.chunks) == 1


def test_remove_by_source_removes_all_chunks_of_document():
    store = FAISSStore(dimension=3)

    chunks = [
        Chunk("Kafka one", {"source": "kafka.pdf"}),
        Chunk("Redis one", {"source": "redis.md"}),
        Chunk("Kafka two", {"source": "kafka.pdf"}),
    ]

    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    store.add(chunks, embeddings)

    removed_count = store.remove_by_source("kafka.pdf")

    assert removed_count == 2
    assert store.index.ntotal == 1
    assert len(store.chunks) == 1
    assert store.chunks[0].metadata["source"] == "redis.md"
