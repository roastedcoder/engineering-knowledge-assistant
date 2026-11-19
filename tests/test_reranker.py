
from unittest.mock import Mock

import pytest

from app.retrieval.reranker import Reranker
from app.models import Chunk


@pytest.fixture
def sample_results():
    return [
        {
            "chunk": Chunk(
                content="Redis is an in-memory database.",
                metadata={"source": "redis.md"},
            ),
            "score": 0.80,
        },
        {
            "chunk": Chunk(
                content="Kafka is a distributed messaging system.",
                metadata={"source": "kafka.md"},
            ),
            "score": 0.75,
        },
        {
            "chunk": Chunk(
                content="MySQL is a relational database.",
                metadata={"source": "mysql.md"},
            ),
            "score": 0.70,
        },
    ]


@pytest.fixture
def reranker():
    instance = Reranker.__new__(Reranker)
    instance.model = Mock()
    return instance


def test_rerank_returns_empty_list_for_empty_results(reranker):
    result = reranker.rerank(
        query="What is Redis?",
        results=[],
    )

    assert result == []


def test_rerank_rejects_invalid_top_k(reranker, sample_results):
    with pytest.raises(ValueError, match="top_k must be greater than 0"):
        reranker.rerank(
            query="What is Redis?",
            results=sample_results,
            top_k=0,
        )


def test_rerank_sorts_results_by_score(
    reranker,
    sample_results,
):
    reranker.model.predict.return_value = [0.2, 0.9, 0.5]

    result = reranker.rerank(
        query="What is Kafka?",
        results=sample_results,
    )

    assert result[0]["chunk"].metadata["source"] == "kafka.md"
    assert result[1]["chunk"].metadata["source"] == "mysql.md"
    assert result[2]["chunk"].metadata["source"] == "redis.md"


def test_rerank_returns_only_top_k(
    reranker,
    sample_results,
):
    reranker.model.predict.return_value = [0.2, 0.9, 0.5]

    result = reranker.rerank(
        query="What is Kafka?",
        results=sample_results,
        top_k=2,
    )

    assert len(result) == 2
    assert result[0]["chunk"].metadata["source"] == "kafka.md"


def test_rerank_preserves_chunk_metadata(
    reranker,
    sample_results,
):
    reranker.model.predict.return_value = [0.8, 0.6, 0.4]

    result = reranker.rerank(
        query="What is Redis?",
        results=sample_results,
    )

    assert result[0]["chunk"].metadata == {
        "source": "redis.md",
    }


def test_rerank_passes_query_and_chunks_to_model(
    reranker,
    sample_results,
):
    reranker.model.predict.return_value = [0.8, 0.6, 0.4]

    reranker.rerank(
        query="What is Redis?",
        results=sample_results,
    )

    expected_pairs = [
        ("What is Redis?", "Redis is an in-memory database."),
        ("What is Redis?", "Kafka is a distributed messaging system."),
        ("What is Redis?", "MySQL is a relational database."),
    ]

    reranker.model.predict.assert_called_once_with(expected_pairs)
