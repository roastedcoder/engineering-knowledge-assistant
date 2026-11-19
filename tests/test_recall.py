import pytest

from app.models import Chunk
from evaluation.metrics.recall import recall_at_k


def create_result(source: str) -> dict:
    chunk = Chunk(
        content="Sample content",
        metadata={"source": source},
    )

    return {
        "chunk": chunk,
        "score": 0.9,
    }


def test_recall_at_k_finds_all_relevant_sources():
    results = [
        create_result("sample.md"),
        create_result("sample.pdf"),
        create_result("other.md"),
    ]

    recall = recall_at_k(
        results,
        ["sample.md", "sample.pdf"],
        k=2,
    )

    assert recall == 1.0


def test_recall_at_k_finds_one_of_two_sources():
    results = [
        create_result("sample.md"),
        create_result("other.md"),
    ]

    recall = recall_at_k(
        results,
        ["sample.md", "sample.pdf"],
        k=2,
    )

    assert recall == 0.5


def test_recall_at_k_ignores_results_after_k():
    results = [
        create_result("other.md"),
        create_result("sample.md"),
    ]

    recall = recall_at_k(
        results,
        ["sample.md"],
        k=1,
    )

    assert recall == 0.0


def test_recall_at_k_counts_duplicate_chunks_as_one_source():
    results = [
        create_result("sample.md"),
        create_result("sample.md"),
    ]

    recall = recall_at_k(
        results,
        ["sample.md", "sample.pdf"],
        k=2,
    )

    assert recall == 0.5


def test_recall_at_k_rejects_invalid_k():
    with pytest.raises(ValueError):
        recall_at_k([], ["sample.md"], k=0)


def test_recall_at_k_rejects_empty_relevant_sources():
    with pytest.raises(ValueError):
        recall_at_k([], [], k=3)