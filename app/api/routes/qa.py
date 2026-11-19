
import logging
import time
from fastapi import APIRouter, Depends
from app.core.dependencies import get_rag_service
from app.generation.rag_service import RAGService
from app.schemas.qa import QuestionRequest, QuestionResponse
from fastapi import APIRouter, Depends, HTTPException
from app.core.exceptions import RAGServiceError


logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/api/v1",
    tags=["Question Answering"],
)


@router.post("/ask", response_model=QuestionResponse)
def ask_question(
    request: QuestionRequest,
    rag_service: RAGService = Depends(get_rag_service),
):
    start_time = time.perf_counter()

    try:
        result = rag_service.answer(
            question=request.question,
            top_k=request.top_k,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "Question answered successfully in %.2f ms",
            elapsed_ms,
        )

        return result

    except RAGServiceError as error:
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        logger.error(
            "RAG request failed after %.2f ms: %s",
            elapsed_ms,
            error,
        )

        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error