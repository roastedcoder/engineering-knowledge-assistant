
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes.qa import router as qa_router
from app.core.dependencies import rag_container
from app.core.rag_factory import build_rag_service
from app.core.logging_config import setup_logging


setup_logging()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Engineering Knowledge Assistant")
    rag_container.service = build_rag_service()
    logger.info("RAG service initialized successfully")

    yield

    logger.info("Shutting down Engineering Knowledge Assistant")
    rag_container.service = None


app = FastAPI(
    title="Engineering Knowledge Assistant",
    description="A local RAG system with citations and evaluation",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(qa_router)


@app.get("/health/live")
def liveness_check():
    return {
        "status": "alive",
        "service": "engineering-knowledge-assistant",
    }


@app.get("/health/ready")
def readiness_check():
    if rag_container.service is None:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=503,
            detail="RAG service is not initialized",
        )

    return {
        "status": "ready",
        "service": "engineering-knowledge-assistant",
    }
