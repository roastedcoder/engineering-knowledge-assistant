
from app.embeddings.embedding_service import EmbeddingService
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import Reranker
from app.generation.llm_service import LLMService


class RAGService:

    def __init__(
        self,
        embedding_service: EmbeddingService,
        hybrid_retriever: HybridRetriever,
        reranker: Reranker,
        llm_service: LLMService,
    ):
        self.embedding_service = embedding_service
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.llm_service = llm_service

    def answer(
        self,
        question: str,
        top_k: int = 3,
        retrieval_k: int = 10,
    ) -> dict:

        # 1. Convert question into an embedding
        query_embedding = self.embedding_service.embed_text(question)

        # 2. Retrieve candidate chunks using hybrid search
        results = self.hybrid_retriever.search(
            query=question,
            query_embedding=query_embedding,
            top_k=retrieval_k,
        )

        if not results:
            return {
                "answer": "I couldn't find relevant information in the documents.",
                "sources": [],
            }

        # 3. Rerank retrieved chunks
        results = self.reranker.rerank(
            query=question,
            results=results,
            top_k=top_k,
        )

        # 4. Build context and source references
        context_parts = []
        sources = []

        for index, result in enumerate(results, start=1):
            chunk = result["chunk"]

            context_parts.append(
                f"[{index}] {chunk.content}"
            )

            sources.append({
                "citation": index,
                "metadata": chunk.metadata,
                "score": result["score"],
            })

        context = "\n\n".join(context_parts)

        # 5. Ask the LLM to answer using retrieved context
        system_prompt = """
        You are an engineering knowledge assistant.

        Answer questions using ONLY the provided context.
        Do not use external knowledge or invent information.

        Cite supporting information using [1], [2], etc.
        If the context does not contain the answer,
        explicitly say that you don't know.
        Keep answers concise.
        """

        prompt = f"""
        Context:
        {context}

        Question:
        {question}

        Answer:
        """

        answer = self.llm_service.generate(
            prompt=prompt,
            system_prompt=system_prompt,
        )

        return {
            "answer": answer,
            "sources": sources,
        }
