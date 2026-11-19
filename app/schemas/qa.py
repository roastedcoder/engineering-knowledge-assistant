
from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=2000,
    )

    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
    )


class SourceResponse(BaseModel):
    citation: int
    metadata: dict
    score: float


class QuestionResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
