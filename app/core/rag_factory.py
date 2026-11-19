
from pathlib import Path

from app.embeddings.embedding_service import EmbeddingService
from app.retrieval.faiss_store import FAISSStore
from app.retrieval.faiss_persistence import FAISSPersistence
from app.retrieval.bm25_store import BM25Store
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import Reranker
from app.generation.llm_service import LLMService
from app.generation.rag_service import RAGService

from app.ingestion.document_tracker import DocumentTracker
from app.ingestion.document_ingestion_service import DocumentIngestionService
from app.ingestion.incremental_ingestion_service import IncrementalIngestionService

from app.chunking.text_chunker import TextChunker

import logging

logger = logging.getLogger(__name__)


def build_rag_service() -> RAGService:
    embedding_service = EmbeddingService()

    persistence = FAISSPersistence("storage/faiss")
    tracker = DocumentTracker("storage/manifest.json")

    index_path = Path("storage/faiss/index.faiss")
    chunks_path = Path("storage/faiss/chunks.pkl")
    manifest_path = Path("storage/manifest.json")

    if index_path.exists() and chunks_path.exists() and manifest_path.exists():
        vector_store = persistence.load()
    else:
        vector_store = FAISSStore(dimension=384)

    incremental_service = IncrementalIngestionService(
        tracker=tracker,
        ingestion_service=DocumentIngestionService(),
        chunker=TextChunker(chunk_size=500, overlap=50),
        embedding_service=embedding_service,
        persistence=persistence,
    )

    update_result = incremental_service.update(
        directory_path="documents",
        vector_store=vector_store,
    )

    logger.info("Document ingestion completed: %s", update_result)

    bm25_store = BM25Store()
    bm25_store.add(vector_store.chunks)

    hybrid_retriever = HybridRetriever(
        faiss_store=vector_store,
        bm25_store=bm25_store,
    )

    reranker = Reranker()
    llm_service = LLMService()

    return RAGService(
        embedding_service=embedding_service,
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        llm_service=llm_service,
    )
