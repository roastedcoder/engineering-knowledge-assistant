
import logging
from fastapi.testclient import TestClient

from app.main import app
from app.core.dependencies import get_rag_service


class FakeRAGService:

    def answer(self, question: str, top_k: int = 3):
        return {
            "answer": "Redis is an in-memory data structure store. [1]",
            "sources": [
                {
                    "citation": 1,
                    "metadata": {"source": "redis.md"},
                    "score": 0.95,
                }
            ],
        }


def test_ask_question():

    app.dependency_overrides[get_rag_service] = lambda: FakeRAGService()

    client = TestClient(app)

    response = client.post(
        "/api/v1/ask",
        json={
            "question": "What is Redis?",
            "top_k": 3,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "Redis" in data["answer"]
    assert len(data["sources"]) == 1

    app.dependency_overrides.clear()


def test_ask_question_rejects_invalid_top_k():

    app.dependency_overrides[get_rag_service] = lambda: FakeRAGService()

    client = TestClient(app)

    response = client.post(
        "/api/v1/ask",
        json={
            "question": "What is Redis?",
            "top_k": 0,
        },
    )

    assert response.status_code == 422

    app.dependency_overrides.clear()

from app.core.exceptions import RAGServiceError


class FailingRAGService:
    def answer(self, question: str, top_k: int = 3):
        raise RAGServiceError("LLM service is unavailable")


def test_ask_question_returns_503_on_rag_failure():
    app.dependency_overrides[get_rag_service] = lambda: FailingRAGService()

    client = TestClient(app)

    response = client.post(
        "/api/v1/ask",
        json={"question": "What is Redis?", "top_k": 3},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "LLM service is unavailable"

    app.dependency_overrides.clear()


def test_successful_request_logs_latency(caplog):
    app.dependency_overrides[get_rag_service] = lambda: FakeRAGService()

    try:
        with caplog.at_level(logging.INFO):
            client = TestClient(app)

            response = client.post(
                "/api/v1/ask",
                json={"question": "What is Redis?", "top_k": 3},
            )

        assert response.status_code == 200
        assert "Question answered successfully in" in caplog.text
        assert "ms" in caplog.text

    finally:
        app.dependency_overrides.clear()


def test_failed_request_logs_error(caplog):
    app.dependency_overrides[get_rag_service] = lambda: FailingRAGService()

    try:
        with caplog.at_level(logging.ERROR):
            client = TestClient(app)

            response = client.post(
                "/api/v1/ask",
                json={"question": "What is Redis?", "top_k": 3},
            )

        assert response.status_code == 503
        assert "RAG request failed after" in caplog.text
        assert "LLM service is unavailable" in caplog.text

    finally:
        app.dependency_overrides.clear()
