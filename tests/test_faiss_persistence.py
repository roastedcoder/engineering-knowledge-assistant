
import pytest

from app.models import Chunk
from app.retrieval.faiss_store import FAISSStore
from app.retrieval.faiss_persistence import FAISSPersistence


def test_save_creates_index_and_chunks_files(tmp_path):
    store = FAISSStore(dimension=3)

    chunks = [
        Chunk(
            content="Redis is an in-memory database.",
            metadata={"source": "redis.md"},
        )
    ]

    embeddings = [[0.1, 0.2, 0.3]]
    store.add(chunks, embeddings)

    persistence = FAISSPersistence(
        storage_dir=str(tmp_path)
    )

    persistence.save(store)

    assert (tmp_path / "index.faiss").exists()
    assert (tmp_path / "chunks.pkl").exists()


def test_save_creates_directory_if_missing(tmp_path):
    storage_dir = tmp_path / "nested" / "faiss"

    store = FAISSStore(dimension=3)

    persistence = FAISSPersistence(
        storage_dir=str(storage_dir)
    )

    persistence.save(store)

    assert (storage_dir / "index.faiss").exists()
    assert (storage_dir / "chunks.pkl").exists()


def test_load_restores_index_and_chunks(tmp_path):
    store = FAISSStore(dimension=3)

    chunks = [
        Chunk(
            content="Redis is an in-memory database.",
            metadata={"source": "redis.md"},
        ),
        Chunk(
            content="Kafka is a messaging platform.",
            metadata={"source": "kafka.md"},
        ),
    ]

    embeddings = [
        [0.1, 0.2, 0.3],
        [0.3, 0.2, 0.1],
    ]

    store.add(chunks, embeddings)

    persistence = FAISSPersistence(
        storage_dir=str(tmp_path)
    )

    persistence.save(store)

    loaded_store = persistence.load()

    assert loaded_store.index.ntotal == 2
    assert len(loaded_store.chunks) == 2

    assert loaded_store.chunks[0].content == chunks[0].content
    assert loaded_store.chunks[1].content == chunks[1].content


def test_loaded_index_returns_same_search_results(tmp_path):
    store = FAISSStore(dimension=3)

    chunks = [
        Chunk(
            content="Redis database",
            metadata={"source": "redis.md"},
        ),
        Chunk(
            content="Kafka messaging",
            metadata={"source": "kafka.md"},
        ),
    ]

    embeddings = [
        [0.1, 0.2, 0.3],
        [0.3, 0.2, 0.1],
    ]

    store.add(chunks, embeddings)

    persistence = FAISSPersistence(
        storage_dir=str(tmp_path)
    )

    persistence.save(store)
    loaded_store = persistence.load()

    query = [0.1, 0.2, 0.3]

    original_results = store.search(query, top_k=2)
    loaded_results = loaded_store.search(query, top_k=2)

    assert [
        result["chunk"].content for result in original_results
    ] == [
        result["chunk"].content for result in loaded_results
    ]

    assert [
        result["score"] for result in original_results
    ] == pytest.approx([
        result["score"] for result in loaded_results
    ])
