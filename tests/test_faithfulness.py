import pytest

from evaluation.metrics.faithfulness import FaithfulnessEvaluator


class FakeLLMService:

    def __init__(self, response):
        self.response = response

    def generate(self, prompt, system_prompt=""):
        return self.response


def test_faithfulness_all_claims_supported():
    response = """
    {
        "claims": [
            {"claim": "Redis is in-memory", "supported": true},
            {"claim": "Redis supports strings", "supported": true}
        ]
    }
    """

    evaluator = FaithfulnessEvaluator(FakeLLMService(response))

    score = evaluator.evaluate(
        context="Redis is an in-memory data store. It supports strings.",
        answer="Redis is in-memory and supports strings.",
    )

    assert score == 1.0


def test_faithfulness_partially_supported():
    response = """
    {
        "claims": [
            {"claim": "Redis is in-memory", "supported": true},
            {"claim": "Redis supports replication", "supported": false}
        ]
    }
    """

    evaluator = FaithfulnessEvaluator(FakeLLMService(response))

    score = evaluator.evaluate(
        context="Redis is an in-memory data store.",
        answer="Redis is in-memory and supports replication.",
    )

    assert score == 0.5


def test_faithfulness_no_claims_supported():
    response = """
    {
        "claims": [
            {"claim": "Redis is a relational database", "supported": false}
        ]
    }
    """

    evaluator = FaithfulnessEvaluator(FakeLLMService(response))

    score = evaluator.evaluate(
        context="Redis is an in-memory data store.",
        answer="Redis is a relational database.",
    )

    assert score == 0.0


def test_faithfulness_rejects_empty_context():
    evaluator = FaithfulnessEvaluator(FakeLLMService("{}"))

    with pytest.raises(ValueError):
        evaluator.evaluate("", "Redis is in-memory.")


def test_faithfulness_rejects_empty_answer():
    evaluator = FaithfulnessEvaluator(FakeLLMService("{}"))

    with pytest.raises(ValueError):
        evaluator.evaluate("Redis is in-memory.", "")


def test_faithfulness_rejects_invalid_json():
    evaluator = FaithfulnessEvaluator(
        FakeLLMService("This is not JSON")
    )

    with pytest.raises(ValueError):
        evaluator.evaluate("Redis is in-memory.", "Redis is in-memory.")


def test_faithfulness_rejects_empty_claims():
    evaluator = FaithfulnessEvaluator(
        FakeLLMService('{"claims": []}')
    )

    with pytest.raises(ValueError):
        evaluator.evaluate("Redis is in-memory.", "Redis is in-memory.")


def test_faithfulness_rejects_invalid_claim_format():
    evaluator = FaithfulnessEvaluator(
        FakeLLMService(
            '{"claims": [{"claim": "Redis is in-memory", "supported": "yes"}]}'
        )
    )

    with pytest.raises(ValueError):
        evaluator.evaluate("Redis is in-memory.", "Redis is in-memory.")