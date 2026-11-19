import json
import time

from app.chunking.text_chunker import TextChunker
from app.embeddings.embedding_service import EmbeddingService
from app.ingestion.document_ingestion_service import DocumentIngestionService
from app.retrieval.faiss_store import FAISSStore

from evaluation.metrics.recall import recall_at_k
from evaluation.metrics.mrr import reciprocal_rank


MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
]

TOP_K = 3


def run_experiment():
    documents = DocumentIngestionService().load_directory("documents")

    with open("evaluation/datasets/rag_eval.json") as file:
        dataset = json.load(file)

    questions = [
        item for item in dataset
        if item["answerable"]
    ]

    chunker = TextChunker(
        chunk_size=200,
        overlap=40,
    )

    chunks = []

    for document in documents:
        chunks.extend(chunker.split(document))

    question_texts = [
        item["question"] for item in questions
    ]

    for model_name in MODELS:
        print(f"\nEvaluating: {model_name}")

        embedding_service = EmbeddingService(
            model_name=model_name
        )

        dimension = (
            embedding_service.model.get_embedding_dimension()
        )

        start_time = time.perf_counter()

        document_embeddings = embedding_service.embed_texts(
            [chunk.content for chunk in chunks]
        )

        store = FAISSStore(dimension=dimension)

        store.add(chunks, document_embeddings)

        indexing_time = time.perf_counter() - start_time

        start_time = time.perf_counter()

        query_embeddings = embedding_service.embed_texts(
            question_texts
        )

        recalls = []
        reciprocal_ranks = []

        for item, query_embedding in zip(
            questions,
            query_embeddings,
        ):
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

        query_time = time.perf_counter() - start_time

        average_recall = sum(recalls) / len(recalls)
        average_mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

        print(f"Dimensions:       {dimension}")
        print(f"Chunks:           {len(chunks)}")
        print(f"Indexing time:    {indexing_time:.3f}s")
        print(f"Query eval time:  {query_time:.3f}s")
        print(f"Recall@{TOP_K}:        {average_recall:.3f}")
        print(f"MRR:              {average_mrr:.3f}")


if __name__ == "__main__":
    run_experiment()