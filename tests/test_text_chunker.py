import pytest

from app.chunking.text_chunker import TextChunker
from app.models import Document


def test_split_document_into_chunks():
    document = Document(
        content="A" * 1000,
        metadata={"source": "sample.pdf", "page": 1},
    )

    chunks = TextChunker(chunk_size=500, overlap=50).split(document)

    assert len(chunks) == 3
    assert [len(chunk.content) for chunk in chunks] == [500, 500, 100]


def test_chunks_have_correct_overlap():
    document = Document(
        content="A" * 1000,
        metadata={"source": "sample.pdf"},
    )

    chunks = TextChunker(chunk_size=500, overlap=50).split(document)

    assert chunks[0].content[450:] == chunks[1].content[:50]


def test_preserve_document_metadata():
    document = Document(
        content="Redis architecture",
        metadata={"source": "redis.pdf", "page": 3},
    )

    chunks = TextChunker().split(document)

    assert chunks[0].metadata["source"] == "redis.pdf"
    assert chunks[0].metadata["page"] == 3
    assert chunks[0].metadata["chunk_index"] == 0


def test_reject_invalid_chunk_configuration():
    with pytest.raises(ValueError):
        TextChunker(chunk_size=100, overlap=100)

    with pytest.raises(ValueError):
        TextChunker(chunk_size=100, overlap=-1)
        