import pytest

from app.embeddings.embedding_service import EmbeddingService


class FakeEmbedding:
    def tolist(self):
        return [0.1] * 384


class FakeModel:
    def encode(self, texts):
        if isinstance(texts, str):
            return FakeEmbedding()

        return FakeBatchEmbedding(len(texts))


class FakeBatchEmbedding:
    def __init__(self, count):
        self.count = count

    def tolist(self):
        return [[0.1] * 384 for _ in range(self.count)]


@pytest.fixture
def embedding_service(monkeypatch):
    monkeypatch.setattr(
        "app.embeddings.embedding_service.SentenceTransformer",
        lambda model_name: FakeModel(),
    )

    return EmbeddingService()


def test_embed_single_text(embedding_service):
    embedding = embedding_service.embed_text("Redis is an in-memory database.")

    assert isinstance(embedding, list)
    assert len(embedding) == 384


def test_embed_multiple_texts(embedding_service):
    texts = ["Redis", "MySQL", "Kafka"]

    embeddings = embedding_service.embed_texts(texts)

    assert len(embeddings) == 3
    assert all(len(embedding) == 384 for embedding in embeddings)