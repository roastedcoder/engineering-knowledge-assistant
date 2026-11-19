
import httpx
import pytest

from app.core.exceptions import RAGServiceError
from app.generation.llm_service import LLMService


def test_generate_returns_llm_response(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "message": {
                    "content": "Redis is an in-memory data store."
                }
            }

    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: FakeResponse())

    service = LLMService()

    result = service.generate("What is Redis?")

    assert result == "Redis is an in-memory data store."


def test_generate_handles_connection_error(monkeypatch):
    def raise_connection_error(*args, **kwargs):
        raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr(httpx, "post", raise_connection_error)

    service = LLMService()

    with pytest.raises(RAGServiceError, match="Could not connect to Ollama"):
        service.generate("What is Redis?")


def test_generate_handles_timeout(monkeypatch):
    def raise_timeout(*args, **kwargs):
        raise httpx.ReadTimeout("Request timed out")

    monkeypatch.setattr(httpx, "post", raise_timeout)

    service = LLMService()

    with pytest.raises(RAGServiceError, match="Could not connect to Ollama"):
        service.generate("What is Redis?")


def test_generate_handles_http_error(monkeypatch):
    def raise_http_error(*args, **kwargs):
        request = httpx.Request("POST", "http://localhost:11434/api/chat")
        response = httpx.Response(404, request=request)
        raise httpx.HTTPStatusError(
            "Not Found",
            request=request,
            response=response,
        )

    monkeypatch.setattr(httpx, "post", raise_http_error)

    service = LLMService()

    with pytest.raises(RAGServiceError, match="Ollama returned HTTP 404"):
        service.generate("What is Redis?")
