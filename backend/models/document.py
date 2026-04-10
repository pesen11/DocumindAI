"""
Document-related Pydantic models.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """A single chunk stored in the vector database."""
    chunk_id: str
    collection_id: str
    file_name: str
    page_number: int
    chunk_index: int
    text: str
    upload_timestamp: datetime = Field(default_factory=datetime.utcnow)


class DocumentInfo(BaseModel):
    """Metadata about a single uploaded document."""
    file_name: str
    page_count: int
    chunk_count: int
    upload_timestamp: datetime = Field(default_factory=datetime.utcnow)
    file_size_bytes: int = 0


class CollectionInfo(BaseModel):
    """Metadata about a ChromaDB collection (user session)."""
    collection_id: str
    files: list[DocumentInfo]
    total_chunks: int
    created_at: datetime = Field(default_factory=datetime.utcnow)


class UploadResponse(BaseModel):
    collection_id: str
    uploaded_files: list[str]
    total_chunks: int
    status: str = "success"
    message: str = ""
    suggested_questions: list[str] = []
