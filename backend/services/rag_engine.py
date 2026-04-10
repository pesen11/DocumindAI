"""
RAG query pipeline:
  question + collection_id → retrieve chunks → build prompt → LLM → parse response
"""

from __future__ import annotations
import logging
import re
import time
from typing import Any

import numpy as np

from backend.config import (
    LLM_PROVIDER, LLM_MODEL, LLM_TEMPERATURE, MAX_TOKENS, LLM_TIMEOUT,
    OPENAI_API_KEY, ANTHROPIC_API_KEY, TOP_K, MAX_HISTORY_TURNS,
)
from backend.models.query import (
    ConversationTurn, QueryResponse, SourceCitation,
)
from backend.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

# ── Prompt templates ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are DocuMind AI, a precise document analysis assistant.
Answer questions using ONLY the provided context.
If the answer is not in the context, respond exactly:
"I cannot find this information in the uploaded documents."
After each factual claim add a citation: [Source: filename.pdf, Page X]
Be concise and accurate. Never fabricate information."""

CONTEXT_TEMPLATE = """Context (retrieved document excerpts):
---
{context_blocks}
---
"""

HISTORY_TEMPLATE = """Previous conversation:
{history}
"""


def _build_context_blocks(hits: list[dict[str, Any]]) -> str:
    blocks = []
    for i, hit in enumerate(hits, start=1):
        meta = hit["metadata"]
        score = hit["similarity_score"]
        blocks.append(
            f"[{i}] File: {meta.get('file_name','unknown')} | "
            f"Page: {meta.get('page_number','?')} | "
            f"Relevance: {score:.2f}\n"
            f"{hit['text']}"
        )
    return "\n\n".join(blocks)


def _build_history_str(history: list[ConversationTurn]) -> str:
    lines = []
    for turn in history[-MAX_HISTORY_TURNS * 2:]:
        prefix = "User" if turn.role == "user" else "Assistant"
        lines.append(f"{prefix}: {turn.content}")
    return "\n".join(lines)


# ── Citation extraction ───────────────────────────────────────────────────────
_CITATION_PATTERN = re.compile(r"\[Source:\s*([^,\]]+?),\s*Page\s*(\d+)\]")


def _extract_citations(text: str) -> list[tuple[str, int]]:
    """Return list of (filename, page_number) tuples from LLM output."""
    return [
        (m.group(1).strip(), int(m.group(2)))
        for m in _CITATION_PATTERN.finditer(text)
    ]


# ── Confidence scoring ────────────────────────────────────────────────────────
def _calculate_confidence(
    similarity_scores: list[float], num_sources: int
) -> float:
    if not similarity_scores:
        return 0.0
    avg_sim = float(np.mean(similarity_scores))
    source_factor = min(num_sources / 3, 1.0)
    confidence = (avg_sim * 0.7 + source_factor * 0.3) * 100
    return round(min(confidence, 100.0), 2)


# ── LLM call wrappers ─────────────────────────────────────────────────────────
def _call_openai(messages: list[dict], temperature: float) -> str:
    import openai
    client = openai.OpenAI(api_key=OPENAI_API_KEY, timeout=LLM_TIMEOUT)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=MAX_TOKENS,
    )
    return response.choices[0].message.content or ""


def _call_anthropic(messages: list[dict], system: str, temperature: float) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, timeout=LLM_TIMEOUT)
    response = client.messages.create(
        model=LLM_MODEL,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=messages,
        temperature=temperature,
    )
    return response.content[0].text if response.content else ""


# ── Query suggestion generation ───────────────────────────────────────────────
def generate_query_suggestions(
    doc_summaries: list[dict[str, str]],
    temperature: float = 0.7,
) -> list[str]:
    """
    Given a list of {file_name, excerpt} dicts, return 5 suggested questions.
    Returns empty list on any failure (non-critical feature).
    """
    if LLM_PROVIDER == "none":
        return []
    try:
        summaries_text = "\n".join(
            f"- {d['file_name']}: {d['excerpt'][:200]}" for d in doc_summaries
        )
        prompt = (
            f"Based on these document excerpts:\n{summaries_text}\n\n"
            "Generate 5 insightful questions a user might want to ask about these documents. "
            "Format as a numbered list (1. ... 2. ...). "
            "Questions only, no answers."
        )
        if LLM_PROVIDER == "anthropic":
            answer = _call_anthropic(
                [{"role": "user", "content": prompt}],
                system="You generate concise, useful questions about documents.",
                temperature=temperature,
            )
        else:
            answer = _call_openai(
                [
                    {"role": "system", "content": "You generate concise, useful questions about documents."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
            )
        # Parse numbered list
        questions = []
        for line in answer.split("\n"):
            line = line.strip()
            match = re.match(r"^\d+[\.\)]\s+(.+)", line)
            if match:
                questions.append(match.group(1).strip())
        return questions[:5]
    except Exception as exc:
        logger.warning("Could not generate query suggestions: %s", exc)
        return []


# ── Main RAG engine ───────────────────────────────────────────────────────────
class RAGEngine:
    def __init__(self, vector_store: VectorStore):
        self.vs = vector_store

    def query(
        self,
        collection_id: str,
        question: str,
        conversation_history: list[ConversationTurn] | None = None,
        top_k: int = TOP_K,
        temperature: float = LLM_TEMPERATURE,
    ) -> QueryResponse:
        """
        Full RAG pipeline:
          1. Retrieve relevant chunks
          2. Build context + conversation history
          3. Call LLM
          4. Parse response + citations
          5. Return structured QueryResponse
        """
        t_start = time.perf_counter()
        history = conversation_history or []

        # ── 1. Retrieve ───────────────────────────────────────────────────────
        hits = self.vs.similarity_search(
            collection_id=collection_id,
            query=question,
            top_k=top_k,
        )
        if not hits:
            return QueryResponse(
                answer="I cannot find this information in the uploaded documents.",
                sources=[],
                confidence=0.0,
                model_used=LLM_MODEL,
            )

        # ── 2. Build prompt ───────────────────────────────────────────────────
        context_str = _build_context_blocks(hits)
        context_section = CONTEXT_TEMPLATE.format(context_blocks=context_str)

        history_section = ""
        if history:
            history_section = HISTORY_TEMPLATE.format(
                history=_build_history_str(history)
            )

        user_message = (
            f"{context_section}\n"
            f"{history_section}\n"
            f"Question: {question}\n\n"
            "Answer with citations [Source: filename.pdf, Page X] after each claim:"
        )

        # ── 3. Call LLM ───────────────────────────────────────────────────────
        if LLM_PROVIDER == "none":
            raw_answer = (
                "[LLM not configured] Retrieved chunks:\n\n"
                + context_str
            )
        elif LLM_PROVIDER == "anthropic":
            messages = [{"role": "user", "content": user_message}]
            raw_answer = _call_anthropic(messages, SYSTEM_PROMPT, temperature)
        elif LLM_PROVIDER == "openai":
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ]
            raw_answer = _call_openai(messages, temperature)
        else:
            raise ValueError(f"Unknown LLM provider: {LLM_PROVIDER}")

        # ── 4. Parse citations ────────────────────────────────────────────────
        cited_pairs = _extract_citations(raw_answer)
        # Map (file, page) → best matching hit for the excerpt
        hit_lookup: dict[tuple[str, int], dict] = {}
        for hit in hits:
            key = (hit["metadata"]["file_name"], hit["metadata"]["page_number"])
            if key not in hit_lookup or hit["similarity_score"] > hit_lookup[key]["similarity_score"]:
                hit_lookup[key] = hit

        seen: set[tuple[str, int]] = set()
        sources: list[SourceCitation] = []
        for file_name, page_num in cited_pairs:
            key = (file_name, page_num)
            if key in seen:
                continue
            seen.add(key)
            hit = hit_lookup.get(key)
            sources.append(SourceCitation(
                file=file_name,
                page=page_num,
                text_excerpt=hit["text"][:300] if hit else "",
                similarity_score=hit["similarity_score"] if hit else 0.0,
                chunk_index=hit["metadata"].get("chunk_index", 0) if hit else 0,
            ))

        # Fallback: if LLM didn't produce citations, use top hits
        if not sources:
            for hit in hits[:3]:
                meta = hit["metadata"]
                sources.append(SourceCitation(
                    file=meta["file_name"],
                    page=meta["page_number"],
                    text_excerpt=hit["text"][:300],
                    similarity_score=hit["similarity_score"],
                    chunk_index=meta.get("chunk_index", 0),
                ))

        # ── 5. Confidence score ───────────────────────────────────────────────
        sim_scores = [h["similarity_score"] for h in hits]
        confidence = _calculate_confidence(sim_scores, len(sources))

        elapsed = time.perf_counter() - t_start
        logger.info(
            "RAG query answered in %.2fs | confidence=%.1f | sources=%d",
            elapsed, confidence, len(sources),
        )

        return QueryResponse(
            answer=raw_answer,
            sources=sources,
            confidence=confidence,
            model_used=LLM_MODEL,
        )
