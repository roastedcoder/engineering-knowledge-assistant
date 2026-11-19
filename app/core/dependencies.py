
from app.generation.rag_service import RAGService


class RAGContainer:

    def __init__(self):
        self.service: RAGService | None = None


rag_container = RAGContainer()


def get_rag_service() -> RAGService:

    if rag_container.service is None:
        raise RuntimeError("RAG service is not initialized")

    return rag_container.service
