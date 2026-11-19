import pytest

from evaluation.metrics.citation_accuracy import CitationEvaluator


class FakeLLMService:

    def __init__(self, response):
        self.response = response

    def generate(self, prompt, system_prompt=""):
        return self.response


CONTEXT = """
[1] Redis is an in-memory data structure store.

[2] Redis supports strings, lists, sets and hashes.
"""


def test_all_citations_are_correct():
    response = """
    {
        "claims": [
            {
                "claim": "Redis is an in-memory store",
                "citations": [1],
                "supported": true
            },
            {
                "claim": "Redis supports strings and lists",
                "citations": [2],
                "supported": true
            }
        ]
    }
    """

    evaluator = CitationEvaluator(FakeLLMService(response))

    score = evaluator.evaluate(
        CONTEXT,
        "Redis is an in-memory store [1]. It supports strings [2].",
    )

    assert score == 1.0


def test_partially_correct_citations():
    response = """
    {
        "claims": [
            {
                "claim": "Redis is an in-memory store",
                "citations": [1],
                "supported": true
            },
            {
                "claim": "Redis supports replication",
                "citations": [2],
                "supported": false
            }
        ]
    }
    """

    evaluator = CitationEvaluator(FakeLLMService(response))

    score = evaluator.evaluate(
        CONTEXT,
        "Redis is an in-memory store [1]. Redis supports replication [2].",
    )

    assert score == 0.5


def test_nonexistent_citation_is_invalid():
    response = """
    {
        "claims": [
            {
                "claim": "Redis is an in-memory store",
                "citations": [99],
                "supported": true
            }
        ]
    }
    """

    evaluator = CitationEvaluator(FakeLLMService(response))

    score = evaluator.evaluate(CONTEXT, "Redis is an in-memory store [99].")

    assert score == 0.0


def test_uncited_claim_is_invalid():
    response = """
    {
        "claims": [
            {
                "claim": "Redis is an in-memory store",
                "citations": [],
                "supported": false
            }
        ]
    }
    """

    evaluator = CitationEvaluator(FakeLLMService(response))

    score = evaluator.evaluate(CONTEXT, "Redis is an in-memory store.")

    assert score == 0.0


def test_empty_context_is_rejected():
    evaluator = CitationEvaluator(FakeLLMService("{}"))

    with pytest.raises(ValueError):
        evaluator.evaluate("", "Redis is an in-memory store [1].")


def test_empty_answer_is_rejected():
    evaluator = CitationEvaluator(FakeLLMService("{}"))

    with pytest.raises(ValueError):
        evaluator.evaluate(CONTEXT, "")


def test_invalid_json_is_rejected():
    evaluator = CitationEvaluator(FakeLLMService("not JSON"))

    with pytest.raises(ValueError):
        evaluator.evaluate(CONTEXT, "Redis is an in-memory store [1].")


def test_empty_claims_are_rejected():
    evaluator = CitationEvaluator(
        FakeLLMService('{"claims": []}')
    )

    with pytest.raises(ValueError):
        evaluator.evaluate(CONTEXT, "Redis is an in-memory store [1].")


def test_invalid_citation_id_type_is_rejected():
    response = """
    {
        "claims": [
            {
                "claim": "Redis is an in-memory store",
                "citations": ["1"],
                "supported": true
            }
        ]
    }
    """

    evaluator = CitationEvaluator(FakeLLMService(response))

    with pytest.raises(ValueError):
        evaluator.evaluate(CONTEXT, "Redis is an in-memory store [1].")