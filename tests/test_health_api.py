
from fastapi.testclient import TestClient

from app.main import app
from app.core.dependencies import rag_container


def test_liveness_check():
    client = TestClient(app)

    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json()["status"] == "alive"


def test_readiness_check_when_initialized():
    original_service = rag_container.service

    try:
        rag_container.service = object()

        client = TestClient(app)
        response = client.get("/health/ready")

        assert response.status_code == 200
        assert response.json()["status"] == "ready"

    finally:
        rag_container.service = original_service


def test_readiness_check_when_not_initialized():
    original_service = rag_container.service

    try:
        rag_container.service = None

        client = TestClient(app)
        response = client.get("/health/ready")

        assert response.status_code == 503
        assert response.json()["detail"] == "RAG service is not initialized"

    finally:
        rag_container.service = original_service
