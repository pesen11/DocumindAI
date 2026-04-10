"""
Document processing pipeline:
  PDF bytes → pages → chunks → vector store
"""

from __future__ import annotations
import logging
import uuid
from datetime import datetime

from backend.config import CHUNK_SIZE, CHUNK_OVERLAP
from backend.models.document import DocumentInfo, CollectionInfo
from backend.services.vector_store import VectorStore
from backend.utils.pdf_parser import extract_pages
from backend.utils.text_splitter import split_text

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Orchestrates ingestion of one or more PDFs into a VectorStore collection."""

    def __init__(self, vector_store: VectorStore):
        self.vs = vector_store

    def process_pdfs(
        self,
        files: list[tuple[str, bytes]],   # [(file_name, file_bytes), …]
        collection_id: str | None = None,
    ) -> CollectionInfo:
        """
        Process a list of (name, bytes) tuples.
        Creates a new ChromaDB collection (or reuses supplied id).
        Returns CollectionInfo with per-file stats.
        """
        # Create/reuse collection
        collection_id = self.vs.create_collection(collection_id)

        doc_infos: list[DocumentInfo] = []

        for file_name, file_bytes in files:
            try:
                info = self._ingest_single(file_name, file_bytes, collection_id)
                doc_infos.append(info)
            except Exception as exc:
                logger.error("Failed to process '%s': %s", file_name, exc)
                raise

        total_chunks = sum(d.chunk_count for d in doc_infos)
        return CollectionInfo(
            collection_id=collection_id,
            files=doc_infos,
            total_chunks=total_chunks,
        )

    # ── Internal ──────────────────────────────────────────────────────────────
    def _ingest_single(
        self, file_name: str, file_bytes: bytes, collection_id: str
    ) -> DocumentInfo:
        """Extract, chunk, embed, and store a single PDF."""
        pages = extract_pages(file_bytes, file_name)
        if not pages:
            raise ValueError(f"No text could be extracted from '{file_name}'")

        all_chunks: list[str] = []
        all_metadatas: list[dict] = []
        all_ids: list[str] = []
        chunk_index = 0
        timestamp = datetime.utcnow().isoformat()

        for page in pages:
            if not page.cleaned_text.strip():
                continue  # Skip blank pages

            page_chunks = split_text(
                page.cleaned_text,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )

            for chunk_text in page_chunks:
                chunk_id = str(uuid.uuid4())
                all_ids.append(chunk_id)
                all_chunks.append(chunk_text)
                all_metadatas.append({
                    "file_name": file_name,
                    "page_number": page.page_number,
                    "chunk_index": chunk_index,
                    "upload_timestamp": timestamp,
                    # Store the first 300 chars as a quick excerpt
                    "excerpt": chunk_text[:300],
                })
                chunk_index += 1

        if not all_chunks:
            raise ValueError(f"'{file_name}' produced no text chunks after processing")

        self.vs.add_chunks(
            collection_id=collection_id,
            chunks=all_chunks,
            metadatas=all_metadatas,
            ids=all_ids,
        )

        return DocumentInfo(
            file_name=file_name,
            page_count=len(pages),
            chunk_count=len(all_chunks),
            file_size_bytes=len(file_bytes),
        )
