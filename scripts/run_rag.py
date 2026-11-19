
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


# 1. Initialize storage and embedding services

embedding_service = EmbeddingService()

persistence = FAISSPersistence("storage/faiss")
tracker = DocumentTracker("storage/manifest.json")

index_path = Path("storage/faiss/index.faiss")
chunks_path = Path("storage/faiss/chunks.pkl")
manifest_path = Path("storage/manifest.json")


# 2. Load existing index or create an empty one

if index_path.exists() and chunks_path.exists() and manifest_path.exists():

    print("Loading existing FAISS index...")
    vector_store = persistence.load()

else:

    print("Starting with an empty FAISS index...")
    vector_store = FAISSStore(dimension=384)


# 3. Initialize incremental ingestion

incremental_service = IncrementalIngestionService(
    tracker=tracker,
    ingestion_service=DocumentIngestionService(),
    chunker=TextChunker(chunk_size=500, overlap=50),
    embedding_service=embedding_service,
    persistence=persistence,
)


# 4. Process document changes

update_result = incremental_service.update(
    directory_path="documents",
    vector_store=vector_store,
)

print("\nDocument update summary:")
print(update_result)


# 5. Rebuild BM25 from the updated chunks

bm25_store = BM25Store()
bm25_store.add(vector_store.chunks)


# 6. Initialize hybrid retrieval

hybrid_retriever = HybridRetriever(
    faiss_store=vector_store,
    bm25_store=bm25_store,
)


# 7. Initialize reranker and LLM

reranker = Reranker()
llm_service = LLMService()


# 8. Initialize RAG service

rag_service = RAGService(
    embedding_service=embedding_service,
    hybrid_retriever=hybrid_retriever,
    reranker=reranker,
    llm_service=llm_service,
)


# 9. Ask a question

question = "How does Apache Kafka guarantee message ordering across partitions?"

result = rag_service.answer(question)

print("\nQuestion:", question)
print("\nAnswer:", result["answer"])

print("\nSources:")

for source in result["sources"]:
    print(source)
