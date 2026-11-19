import pytest

from app.models import Chunk
from evaluation.metrics.mrr import (
    reciprocal_rank,
    mean_reciprocal_rank,
)


def create_result(source: str) -> dict:
    return {
        "chunk": Chunk(
            content="Sample content",
            metadata={"source": source},
        ),
        "score": 0.9,
    }


def test_reciprocal_rank_first_position():
    results = [create_result("sample.md")]

    assert reciprocal_rank(results, ["sample.md"]) == 1.0


def test_reciprocal_rank_second_position():
    results = [
        create_result("other.md"),
        create_result("sample.md"),
    ]

    assert reciprocal_rank(results, ["sample.md"]) == 0.5


def test_reciprocal_rank_not_found():
    results = [create_result("other.md")]

    assert reciprocal_rank(results, ["sample.md"]) == 0.0


def test_reciprocal_rank_uses_first_relevant_document():
    results = [
        create_result("sample.md"),
        create_result("sample.pdf"),
    ]

    assert reciprocal_rank(
        results,
        ["sample.md", "sample.pdf"],
    ) == 1.0


def test_mean_reciprocal_rank():
    all_results = [
        [create_result("sample.md")],
        [
            create_result("other.md"),
            create_result("sample.pdf"),
        ],
    ]

    relevant_sources = [
        ["sample.md"],
        ["sample.pdf"],
    ]

    assert mean_reciprocal_rank(
        all_results,
        relevant_sources,
    ) == 0.75


def test_mean_reciprocal_rank_rejects_mismatched_inputs():
    with pytest.raises(ValueError):
        mean_reciprocal_rank(
            [[create_result("sample.md")]],
            [],
        )


def test_mean_reciprocal_rank_rejects_empty_dataset():
    with pytest.raises(ValueError):
        mean_reciprocal_rank([], [])