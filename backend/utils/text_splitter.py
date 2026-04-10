"""
Sentence-aware recursive text splitter.
Mirrors LangChain's RecursiveCharacterTextSplitter logic without the dependency.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class TextChunk:
    text: str
    start_char: int   # offset within the full page text (for highlighting)
    end_char: int


# Ordered list of separators tried in sequence
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _split_by_separator(text: str, separator: str) -> list[str]:
    if separator == "":
        return list(text)
    if separator == ". ":
        # Keep the period with the sentence
        parts = re.split(r"(?<=\.) ", text)
    else:
        parts = text.split(separator)
    return [p for p in parts if p]


def _merge_splits(splits: list[str], separator: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Greedily merge small splits back up to chunk_size."""
    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0
    sep_len = len(separator)

    for part in splits:
        part_len = len(part)
        # +sep_len for the rejoining separator (except at the start)
        joining_cost = sep_len if current_parts else 0

        if current_len + joining_cost + part_len > chunk_size and current_parts:
            # Flush current buffer
            chunk_text = separator.join(current_parts).strip()
            if chunk_text:
                chunks.append(chunk_text)

            # Overlap: keep tail of current_parts that fits in chunk_overlap
            overlap_parts: list[str] = []
            overlap_len = 0
            for p in reversed(current_parts):
                addition = len(p) + (sep_len if overlap_parts else 0)
                if overlap_len + addition <= chunk_overlap:
                    overlap_parts.insert(0, p)
                    overlap_len += addition
                else:
                    break
            current_parts = overlap_parts
            current_len = overlap_len

        current_parts.append(part)
        current_len += len(part) + (sep_len if len(current_parts) > 1 else 0)

    if current_parts:
        chunk_text = separator.join(current_parts).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks


def split_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: Optional[list[str]] = None,
) -> list[str]:
    """
    Recursively split *text* into chunks of at most *chunk_size* characters
    with *chunk_overlap* overlap, always trying to break at natural boundaries.
    Returns plain strings (no offset tracking needed for storage).
    """
    if separators is None:
        separators = _SEPARATORS

    sep = separators[0]
    remaining = separators[1:]

    splits = _split_by_separator(text, sep)

    good_splits: list[str] = []
    final_chunks: list[str] = []

    for split in splits:
        if len(split) <= chunk_size:
            good_splits.append(split)
        else:
            # This split is still too long — recurse
            if good_splits:
                merged = _merge_splits(good_splits, sep, chunk_size, chunk_overlap)
                final_chunks.extend(merged)
                good_splits = []
            if remaining:
                sub_chunks = split_text(split, chunk_size, chunk_overlap, remaining)
                final_chunks.extend(sub_chunks)
            else:
                # Last resort: hard cut
                for i in range(0, len(split), chunk_size - chunk_overlap):
                    final_chunks.append(split[i : i + chunk_size])

    if good_splits:
        merged = _merge_splits(good_splits, sep, chunk_size, chunk_overlap)
        final_chunks.extend(merged)

    return [c for c in final_chunks if c.strip()]
