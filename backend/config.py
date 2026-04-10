"""
DocuMind AI - Configuration Module
All settings loaded from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Base paths ───────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "chroma_db"))

# ── LLM provider ─────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Auto-select provider based on which key is present
def get_llm_provider() -> str:
    if ANTHROPIC_API_KEY:
        return "anthropic"
    if OPENAI_API_KEY:
        return "openai"
    return "none"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", get_llm_provider())
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-6" if LLM_PROVIDER == "anthropic" else "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "30"))

# ── Document processing ───────────────────────────────────────────────────────
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", str(10 * 1024 * 1024)))  # 10 MB
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "5"))

# ── Embedding model ───────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── Conversation memory ───────────────────────────────────────────────────────
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "10"))

# ── CORS ─────────────────────────────────────────────────────────────────────
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# ── Validate critical settings ────────────────────────────────────────────────
def validate_config() -> list[str]:
    """Return list of configuration warnings (not fatal errors)."""
    warnings = []
    if LLM_PROVIDER == "none":
        warnings.append(
            "No LLM API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env"
        )
    return warnings
