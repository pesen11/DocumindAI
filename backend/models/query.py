"""
Query / response Pydantic models.
"""

from typing import Optional
from pydantic import BaseModel, Field


class ConversationTurn(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class SourceCitation(BaseModel):
    """A single source cited in an answer."""
    file: str
    page: int
    text_excerpt: str
    similarity_score: float = 0.0
    chunk_index: int = 0


class QueryRequest(BaseModel):
    collection_id: str
    question: str = Field(..., min_length=1, max_length=2000)
    conversation_history: list[ConversationTurn] = []
    top_k: int = Field(default=5, ge=1, le=20)
    temperature: float = Field(default=0.3, ge=0.0, le=1.0)


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
    confidence: float = Field(..., ge=0.0, le=100.0)
    model_used: str = ""


class ClearHistoryRequest(BaseModel):
    collection_id: str


class ErrorResponse(BaseModel):
    error: str
    detail: str = ""
