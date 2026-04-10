"""
ChromaDB vector store service.
One ChromaDB collection = one user session (collection_id).
"""

from __future__ import annotations
import logging
import os
import uuid
from datetime import datetime
from typing import Any

import chromadb
from chromadb.config import Settings

from backend.config import CHROMA_PERSIST_DIR, TOP_K
from backend.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


def _make_client() -> chromadb.ClientAPI:
    if os.getenv("CHROMA_USE_EPHEMERAL", "false").lower() == "true":
        # In-memory client — required for Streamlit Cloud (ephemeral filesystem)
        client = chromadb.EphemeralClient(
            settings=Settings(anonymized_telemetry=False),
        )
    else:
        client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
    # Force SQLite schema migrations to run before any real operations.
    # Without this, chromadb's lazy init can raise "no such table: collections"
    # on the first actual call (especially on Streamlit Cloud).
    client.list_collections()
    return client


class VectorStore:
    """Manages ChromaDB collections and similarity search."""

    def __init__(self):
        self._client = _make_client()
        self._embedder = EmbeddingService.get_instance()

    # ── Collection lifecycle ──────────────────────────────────────────────────
    def create_collection(self, collection_id: str | None = None) -> str:
        """Create a new collection and return its id."""
        if collection_id is None:
            collection_id = str(uuid.uuid4())
        # ChromaDB collection names must be alphanumeric + hyphens
        safe_id = collection_id.replace("_", "-")
        self._client.get_or_create_collection(
            name=safe_id,
            metadata={"created_at": datetime.utcnow().isoformat()},
        )
        logger.info("Created/retrieved collection '%s'", safe_id)
        return safe_id

    def delete_collection(self, collection_id: str) -> None:
        try:
            self._client.delete_collection(name=collection_id)
            logger.info("Deleted collection '%s'", collection_id)
        except Exception as exc:
            logger.warning("Could not delete collection '%s': %s", collection_id, exc)

    def collection_exists(self, collection_id: str) -> bool:
        try:
            self._client.get_collection(name=collection_id)
            return True
        except Exception:
            return False

    # ── Document storage ──────────────────────────────────────────────────────
    def add_chunks(
        self,
        collection_id: str,
        chunks: list[str],
        metadatas: list[dict[str, Any]],
        ids: list[str] | None = None,
    ) -> None:
        """
        Embed *chunks* and store them with *metadatas*.
        *ids* are auto-generated UUIDs if not supplied.
        """
        if not chunks:
            return

        collection = self._client.get_collection(name=collection_id)

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in chunks]

        # Batch-embed all chunks at once for efficiency
        embeddings = self._embedder.embed_batch(chunks)

        # ChromaDB upsert in batches of 500 to stay within limits
        batch_size = 500
        for start in range(0, len(chunks), batch_size):
            end = start + batch_size
            collection.upsert(
                ids=ids[start:end],
                documents=chunks[start:end],
                embeddings=embeddings[start:end],
                metadatas=metadatas[start:end],
            )
        logger.info("Stored %d chunks in collection '%s'", len(chunks), collection_id)

    # ── Similarity search ─────────────────────────────────────────────────────
    def similarity_search(
        self,
        collection_id: str,
        query: str,
        top_k: int = TOP_K,
        where: dict | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the top-k most similar chunks to *query*.
        Returns a list of dicts: {text, metadata, similarity_score}.
        """
        collection = self._client.get_collection(name=collection_id)
        query_embedding = self._embedder.embed_text(query)

        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, collection.count() or 1),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = collection.query(**kwargs)

        hits: list[dict[str, Any]] = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, distances):
            # ChromaDB returns L2 distance; convert to cosine similarity in [0,1]
            # For normalised vectors: similarity = 1 - distance/2
            similarity = max(0.0, 1.0 - dist / 2.0)
            hits.append({"text": doc, "metadata": meta, "similarity_score": similarity})

        return hits

    # ── Collection metadata ───────────────────────────────────────────────────
    def get_collection_stats(self, collection_id: str) -> dict[str, Any]:
        try:
            col = self._client.get_collection(name=collection_id)
            count = col.count()
            # Collect unique file names from a sample of metadata
            sample = col.get(limit=1000, include=["metadatas"])
            files: set[str] = set()
            for meta in sample.get("metadatas", []):
                if meta and "file_name" in meta:
                    files.add(meta["file_name"])
            return {"total_chunks": count, "files": sorted(files)}
        except Exception as exc:
            logger.error("Could not get stats for '%s': %s", collection_id, exc)
            return {"total_chunks": 0, "files": []}
