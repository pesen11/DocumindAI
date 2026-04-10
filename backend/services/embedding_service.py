"""
Embedding service — wraps sentence-transformers.
Lazy-loads the model on first use to keep startup time fast.
"""

from __future__ import annotations
import logging
import threading
from typing import Optional

import numpy as np

from backend.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Thread-safe singleton wrapper around a SentenceTransformer model."""

    _instance: Optional["EmbeddingService"] = None
    _lock = threading.Lock()

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None
        self._model_lock = threading.Lock()

    # ── Singleton factory ─────────────────────────────────────────────────────
    @classmethod
    def get_instance(cls) -> "EmbeddingService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ── Lazy model loading ────────────────────────────────────────────────────
    def _ensure_model(self) -> None:
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    logger.info("Loading embedding model '%s' …", self.model_name)
                    from sentence_transformers import SentenceTransformer
                    self._model = SentenceTransformer(self.model_name)
                    logger.info("Embedding model loaded.")

    # ── Public API ────────────────────────────────────────────────────────────
    def embed_text(self, text: str) -> list[float]:
        """Embed a single string. Returns a flat Python list of floats."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """
        Embed a list of strings in batches.
        Returns a list of float lists (one per input string).
        """
        if not texts:
            return []
        self._ensure_model()
        try:
            embeddings: np.ndarray = self._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            return embeddings.tolist()
        except Exception as exc:
            logger.error("Embedding generation failed: %s", exc)
            raise RuntimeError(f"Failed to generate embeddings: {exc}") from exc

    @property
    def dimension(self) -> int:
        """Embedding vector size (loaded lazily)."""
        self._ensure_model()
        return self._model.get_sentence_embedding_dimension()
