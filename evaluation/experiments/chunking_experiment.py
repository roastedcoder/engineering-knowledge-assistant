import json

from app.chunking.text_chunker import TextChunker
from app.embeddings.embedding_service import EmbeddingService
from app.ingestion.document_ingestion_service import DocumentIngestionService
from app.retrieval.faiss_store import FAISSStore

from evaluation.metrics.recall import recall_at_k
from evaluation.metrics.mrr import reciprocal_rank


CONFIGURATIONS = [
    (100, 0),
    (100, 20),
    (200, 0),
    (200, 40),
    (500, 50),
]

TOP_K = 3


def run_experiment():
    ingestion_service = DocumentIngestionService()
    documents = ingestion_service.load_directory("documents")

    embedding_service = EmbeddingService()

    with open("evaluation/datasets/rag_eval.json") as file:
        dataset = json.load(file)

    questions = [
        item for item in dataset
        if item["answerable"]
    ]

    for chunk_size, overlap in CONFIGURATIONS:
        chunker = TextChunker(
            chunk_size=chunk_size,
            overlap=overlap,
        )

        chunks = []

        for document in documents:
            chunks.extend(chunker.split(document))

        embeddings = embedding_service.embed_texts(
            [chunk.content for chunk in chunks]
        )

        store = FAISSStore(dimension=384)
        store.add(chunks, embeddings)

        recalls = []
        reciprocal_ranks = []

        for item in questions:
            query_embedding = embedding_service.embed_text(
                item["question"]
            )

            results = store.search(
                query_embedding,
                top_k=TOP_K,
            )

            recalls.append(
                recall_at_k(
                    results,
                    item["relevant_sources"],
                    TOP_K,
                )
            )

            reciprocal_ranks.append(
                reciprocal_rank(
                    results,
                    item["relevant_sources"],
                )
            )

        average_recall = sum(recalls) / len(recalls)
        average_mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

        print(
            f"Chunk size: {chunk_size:3} | "
            f"Overlap: {overlap:2} | "
            f"Chunks: {len(chunks):2} | "
            f"Recall@{TOP_K}: {average_recall:.3f} | "
            f"MRR: {average_mrr:.3f}"
        )


if __name__ == "__main__":
    run_experiment()