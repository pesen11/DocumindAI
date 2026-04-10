"""
FastAPI route definitions for DocuMind AI.
All endpoint logic lives here; shared service instances are injected via
the FastAPI dependency system.
"""

from __future__ import annotations
import logging
from datetime import datetime
from typing import Annotated

from fastapi import (
    APIRouter, Depends, File, Form, HTTPException, UploadFile, status,
)
from fastapi.responses import JSONResponse

from backend.config import MAX_FILE_SIZE
from backend.models.document import CollectionInfo, UploadResponse
from backend.models.query import (
    ClearHistoryRequest, ErrorResponse, QueryRequest, QueryResponse,
)
from backend.services.document_processor import DocumentProcessor
from backend.services.rag_engine import RAGEngine, generate_query_suggestions
from backend.services.vector_store import VectorStore

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["DocuMind AI"])


# ── Shared service singletons (initialised once in main.py) ──────────────────
_vector_store: VectorStore | None = None
_doc_processor: DocumentProcessor | None = None
_rag_engine: RAGEngine | None = None


def init_services() -> None:
    global _vector_store, _doc_processor, _rag_engine
    _vector_store = VectorStore()
    _doc_processor = DocumentProcessor(_vector_store)
    _rag_engine = RAGEngine(_vector_store)


def get_vector_store() -> VectorStore:
    if _vector_store is None:
        raise RuntimeError("Services not initialised — call init_services() first")
    return _vector_store


def get_doc_processor() -> DocumentProcessor:
    if _doc_processor is None:
        raise RuntimeError("Services not initialised")
    return _doc_processor


def get_rag_engine() -> RAGEngine:
    if _rag_engine is None:
        raise RuntimeError("Services not initialised")
    return _rag_engine


# ── Helper ────────────────────────────────────────────────────────────────────
def _validate_pdf(upload: UploadFile) -> None:
    """Raise HTTP 400 if the file is not a valid PDF or is too large."""
    if not upload.filename or not upload.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"'{upload.filename}' is not a PDF file.",
        )
    # content-length header may not be set; we check after reading


# ── Endpoints ─────────────────────────────────────────────────────────────────
@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload PDF documents",
    description="Upload one or more PDF files to create a new collection.",
)
async def upload_documents(
    files: list[UploadFile] = File(..., description="PDF files to upload"),
    collection_id: str | None = Form(default=None),
    processor: DocumentProcessor = Depends(get_doc_processor),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    file_tuples: list[tuple[str, bytes]] = []
    for upload in files:
        _validate_pdf(upload)
        raw = await upload.read()
        if len(raw) > MAX_FILE_SIZE:
            size_mb = MAX_FILE_SIZE // (1024 * 1024)
            raise HTTPException(
                status_code=413,
                detail=f"'{upload.filename}' exceeds the {size_mb} MB limit.",
            )
        if not raw.startswith(b"%PDF"):
            raise HTTPException(
                status_code=400,
                detail=f"'{upload.filename}' does not appear to be a valid PDF.",
            )
        file_tuples.append((upload.filename, raw))

    try:
        info: CollectionInfo = processor.process_pdfs(file_tuples, collection_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected error during upload")
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}")

    # Generate suggested questions (best-effort; failure doesn't break upload)
    summaries = [
        {"file_name": f.file_name, "excerpt": ""}
        for f in info.files
    ]
    suggestions = generate_query_suggestions(summaries)

    return UploadResponse(
        collection_id=info.collection_id,
        uploaded_files=[f.file_name for f in info.files],
        total_chunks=info.total_chunks,
        suggested_questions=suggestions,
        message=f"Processed {len(info.files)} document(s) into {info.total_chunks} chunks.",
    )


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question",
    description="Query the RAG pipeline with a natural-language question.",
)
async def query_documents(
    request: QueryRequest,
    engine: RAGEngine = Depends(get_rag_engine),
    vs: VectorStore = Depends(get_vector_store),
):
    if not vs.collection_exists(request.collection_id):
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{request.collection_id}' not found. Upload documents first.",
        )
    try:
        response = engine.query(
            collection_id=request.collection_id,
            question=request.question,
            conversation_history=request.conversation_history,
            top_k=request.top_k,
            temperature=request.temperature,
        )
        return response
    except TimeoutError:
        raise HTTPException(status_code=504, detail="LLM request timed out.")
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/collections/{collection_id}",
    response_model=CollectionInfo,
    summary="Get collection details",
)
async def get_collection(
    collection_id: str,
    vs: VectorStore = Depends(get_vector_store),
):
    if not vs.collection_exists(collection_id):
        raise HTTPException(status_code=404, detail="Collection not found.")
    stats = vs.get_collection_stats(collection_id)
    from backend.models.document import DocumentInfo
    return CollectionInfo(
        collection_id=collection_id,
        files=[DocumentInfo(file_name=f, page_count=0, chunk_count=0) for f in stats["files"]],
        total_chunks=stats["total_chunks"],
    )


@router.delete(
    "/collections/{collection_id}",
    summary="Delete a collection",
)
async def delete_collection(
    collection_id: str,
    vs: VectorStore = Depends(get_vector_store),
):
    if not vs.collection_exists(collection_id):
        raise HTTPException(status_code=404, detail="Collection not found.")
    vs.delete_collection(collection_id)
    return {"status": "deleted", "collection_id": collection_id}


@router.post(
    "/clear-history",
    summary="Clear conversation history",
    description=(
        "Conversation history is managed client-side (Streamlit session_state). "
        "This endpoint confirms the collection exists and returns success."
    ),
)
async def clear_history(
    request: ClearHistoryRequest,
    vs: VectorStore = Depends(get_vector_store),
):
    if not vs.collection_exists(request.collection_id):
        raise HTTPException(status_code=404, detail="Collection not found.")
    return {"status": "cleared", "collection_id": request.collection_id}


@router.get("/health", include_in_schema=False)
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}
