
from app.ingestion.document_ingestion_service import DocumentIngestionService
from app.chunking.text_chunker import TextChunker
from app.embeddings.embedding_service import EmbeddingService
from app.retrieval.faiss_store import FAISSStore
from app.generation.llm_service import LLMService
from app.generation.rag_service import RAGService


# 1. Load documents
ingestion_service = DocumentIngestionService()
documents = ingestion_service.load_directory("documents")

print(f"Loaded {len(documents)} documents")


# 2. Split documents into chunks
chunker = TextChunker(chunk_size=500, overlap=50)

chunks = []

for document in documents:
    chunks.extend(chunker.split(document))

print(f"Created {len(chunks)} chunks")


# 3. Generate embeddings
embedding_service = EmbeddingService()

embeddings = embedding_service.embed_texts(
    [chunk.content for chunk in chunks]
)


# 4. Build FAISS index
vector_store = FAISSStore(dimension=384)
vector_store.add(chunks, embeddings)


# 5. Initialize LLM and RAG service
llm_service = LLMService()

rag_service = RAGService(
    embedding_service=embedding_service,
    vector_store=vector_store,
    llm_service=llm_service,
)


# 6. Ask a question
question = "How does Apache Kafka guarantee message ordering across partitions?"

result = rag_service.answer(question)

print("\nQuestion:", question)
print("\nAnswer:", result["answer"])

print("\nSources:")

for source in result["sources"]:
    print(source)
