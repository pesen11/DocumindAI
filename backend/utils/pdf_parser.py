"""
PDF text extraction with page number tracking.
Uses PyPDF2; falls back to pypdf if available.
"""

import io
import re
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PageContent:
    page_number: int    # 1-indexed
    raw_text: str
    cleaned_text: str


def _clean_text(text: str) -> str:
    """Normalize whitespace and fix common PDF extraction artefacts."""
    # Replace various unicode spaces / non-breaking spaces
    text = text.replace("\xa0", " ").replace("\u200b", "")
    # Collapse runs of whitespace (but preserve paragraph breaks)
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing whitespace on each line
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines).strip()
    return text


def extract_pages(file_bytes: bytes, file_name: str = "") -> list[PageContent]:
    """
    Extract text from a PDF byte string.
    Returns a list of PageContent objects, one per page.
    Raises ValueError on unreadable files.
    """
    pages: list[PageContent] = []

    # Try PyPDF2 first
    try:
        import PyPDF2

        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        for i, page in enumerate(reader.pages, start=1):
            raw = page.extract_text() or ""
            cleaned = _clean_text(raw)
            pages.append(PageContent(page_number=i, raw_text=raw, cleaned_text=cleaned))
        logger.info("Extracted %d pages from '%s' via PyPDF2", len(pages), file_name)
        return pages

    except ImportError:
        logger.warning("PyPDF2 not installed, trying pypdf")
    except Exception as exc:
        logger.warning("PyPDF2 failed on '%s': %s — trying pypdf", file_name, exc)

    # Fallback to pypdf
    try:
        import pypdf

        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        for i, page in enumerate(reader.pages, start=1):
            raw = page.extract_text() or ""
            cleaned = _clean_text(raw)
            pages.append(PageContent(page_number=i, raw_text=raw, cleaned_text=cleaned))
        logger.info("Extracted %d pages from '%s' via pypdf", len(pages), file_name)
        return pages

    except ImportError as exc:
        raise ValueError(
            "Neither PyPDF2 nor pypdf is installed. Run: pip install PyPDF2"
        ) from exc
    except Exception as exc:
        raise ValueError(f"Cannot read PDF '{file_name}': {exc}") from exc
