"""
DocuMind AI — Streamlit frontend.

Run with:
    streamlit run frontend/streamlit_app.py
"""

from __future__ import annotations
import json
import time
from datetime import datetime
from io import BytesIO
from typing import Any

import requests
import streamlit as st

# ── Configuration ─────────────────────────────────────────────────────────────
import os
from dotenv import load_dotenv

load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
API_TIMEOUT = 120  # seconds

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Chat bubbles */
    .user-bubble {
        background: #2563eb;
        color: white;
        border-radius: 18px 18px 4px 18px;
        padding: 12px 16px;
        margin: 4px 0;
        max-width: 85%;
        margin-left: auto;
    }
    .assistant-bubble {
        background: #f1f5f9;
        color: #1e293b;
        border-radius: 18px 18px 18px 4px;
        padding: 12px 16px;
        margin: 4px 0;
        max-width: 85%;
    }
    /* Confidence bar */
    .conf-bar {
        height: 8px;
        border-radius: 4px;
        background: linear-gradient(90deg, #ef4444, #f59e0b, #22c55e);
    }
    /* Source badge */
    .src-badge {
        background: #e0f2fe;
        color: #0369a1;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 0.75rem;
        display: inline-block;
        margin: 2px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state initialisation ──────────────────────────────────────────────
def _init_state() -> None:
    defaults = {
        "collection_id": None,
        "messages": [],          # [{"role", "content", "sources"?, "confidence"?}]
        "uploaded_docs": [],     # [{"name", "chunks"}]
        "top_k": 5,
        "temperature": 0.3,
        "suggested_questions": [],
        "processing": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ── API helpers ───────────────────────────────────────────────────────────────
def _api(method: str, path: str, **kwargs) -> dict[str, Any] | None:
    url = f"{BACKEND_URL}{path}"
    try:
        resp = getattr(requests, method)(url, timeout=API_TIMEOUT, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach backend. Is `uvicorn backend.main:app` running?")
        return None
    except requests.exceptions.Timeout:
        st.error("Request timed out. Try again.")
        return None
    except requests.exceptions.HTTPError as exc:
        detail = ""
        try:
            detail = exc.response.json().get("detail", "")
        except Exception:
            pass
        st.error(f"API error {exc.response.status_code}: {detail or exc}")
        return None


def upload_files(files: list) -> bool:
    """Upload files to the backend. Returns True on success."""
    file_tuples = []
    for f in files:
        file_tuples.append(("files", (f.name, f.getvalue(), "application/pdf")))

    params = {}
    if st.session_state.collection_id:
        params["collection_id"] = st.session_state.collection_id

    result = _api("post", "/api/upload", files=file_tuples, data=params)
    if result is None:
        return False

    st.session_state.collection_id = result["collection_id"]
    st.session_state.suggested_questions = result.get("suggested_questions", [])

    # Merge uploaded doc list
    existing = {d["name"] for d in st.session_state.uploaded_docs}
    for fname in result.get("uploaded_files", []):
        if fname not in existing:
            st.session_state.uploaded_docs.append({
                "name": fname,
                "chunks": "?",
            })

    return True


def ask_question(question: str) -> dict[str, Any] | None:
    """Send a question to the RAG pipeline."""
    payload = {
        "collection_id": st.session_state.collection_id,
        "question": question,
        "conversation_history": [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[-20:]  # last 10 turns
        ],
        "top_k": st.session_state.top_k,
        "temperature": st.session_state.temperature,
    }
    return _api("post", "/api/query", json=payload)


# ── Export helpers ────────────────────────────────────────────────────────────
def _export_as_markdown() -> str:
    lines = [
        f"# DocuMind AI — Conversation Export",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Collection:** `{st.session_state.collection_id}`",
        "",
    ]
    for msg in st.session_state.messages:
        role_label = "**You**" if msg["role"] == "user" else "**DocuMind AI**"
        lines.append(f"{role_label}: {msg['content']}")
        if msg.get("sources"):
            lines.append("\n*Sources:*")
            for src in msg["sources"]:
                lines.append(f"- {src['file']} p.{src['page']}")
        lines.append("")
    return "\n".join(lines)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/books.png",
        width=60,
    )
    st.title("DocuMind AI")
    st.caption("Ask questions about your documents")

    st.divider()

    # ── File uploader ─────────────────────────────────────────────────────────
    st.subheader("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Drop PDF files here",
        type=["pdf"],
        accept_multiple_files=True,
        help="Max 10 MB per file. Multiple files supported.",
        label_visibility="collapsed",
    )

    if uploaded_files:
        process_btn = st.button(
            "⚡ Process Documents",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.processing,
        )
        if process_btn:
            st.session_state.processing = True
            with st.spinner("Extracting text & generating embeddings…"):
                progress = st.progress(0, text="Uploading…")
                ok = upload_files(uploaded_files)
                progress.progress(100, text="Done!")
            st.session_state.processing = False
            if ok:
                st.success(f"Processed {len(uploaded_files)} file(s)!")
                st.rerun()

    # ── Document list ─────────────────────────────────────────────────────────
    if st.session_state.uploaded_docs:
        st.divider()
        st.subheader("📚 Loaded Documents")
        for doc in st.session_state.uploaded_docs:
            col1, col2 = st.columns([5, 1])
            col1.markdown(f"📄 `{doc['name']}`")
        # Delete entire collection
        if st.button("🗑 Remove All Documents", use_container_width=True):
            if st.session_state.collection_id:
                _api("delete", f"/api/collections/{st.session_state.collection_id}")
            st.session_state.collection_id = None
            st.session_state.uploaded_docs = []
            st.session_state.messages = []
            st.session_state.suggested_questions = []
            st.rerun()

    # ── Conversation controls ─────────────────────────────────────────────────
    st.divider()
    st.subheader("💬 Conversation")

    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        if st.session_state.collection_id:
            _api("post", "/api/clear-history",
                 json={"collection_id": st.session_state.collection_id})
        st.rerun()

    if st.session_state.messages:
        md_content = _export_as_markdown()
        st.download_button(
            "⬇️ Export as Markdown",
            data=md_content,
            file_name=f"documind_export_{datetime.now():%Y%m%d_%H%M}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    # ── Settings ──────────────────────────────────────────────────────────────
    with st.expander("⚙️ Settings"):
        st.session_state.top_k = st.slider(
            "Sources to retrieve (top-k)",
            min_value=1, max_value=15,
            value=st.session_state.top_k,
            help="How many document chunks to pass to the LLM.",
        )
        st.session_state.temperature = st.slider(
            "LLM Temperature",
            min_value=0.0, max_value=1.0,
            value=st.session_state.temperature, step=0.05,
            help="Higher = more creative. Lower = more factual.",
        )

    # ── Backend status ────────────────────────────────────────────────────────
    st.divider()
    health = _api("get", "/api/health")
    if health:
        st.success("Backend: connected", icon="✅")
    else:
        st.error("Backend: offline", icon="❌")


# ── Main area ─────────────────────────────────────────────────────────────────
st.title("DocuMind AI — Document Intelligence Chat")

if not st.session_state.collection_id:
    st.info(
        "👈 Upload one or more PDF documents using the sidebar to get started.",
        icon="ℹ️",
    )
    st.markdown("""
    ### What can DocuMind AI do?
    - 📄 **Multi-document Q&A** — Ask questions across multiple PDFs simultaneously
    - 🔍 **Precise citations** — Every answer includes exact page references
    - 💬 **Conversation memory** — Follow-up questions understand context
    - 📊 **Confidence scoring** — See how sure the AI is about each answer
    - 📥 **Export chats** — Download your Q&A session as Markdown
    """)
else:
    # ── Suggested questions ───────────────────────────────────────────────────
    if st.session_state.suggested_questions and not st.session_state.messages:
        st.subheader("💡 Suggested Questions")
        cols = st.columns(min(3, len(st.session_state.suggested_questions)))
        for i, question in enumerate(st.session_state.suggested_questions[:3]):
            with cols[i % 3]:
                if st.button(question, key=f"sugg_{i}", use_container_width=True):
                    st.session_state["_pending_question"] = question
                    st.rerun()

    # ── Chat history ──────────────────────────────────────────────────────────
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
                st.markdown(msg["content"])

                # Sources expander
                if msg.get("sources"):
                    sources = msg["sources"]
                    conf = msg.get("confidence", 0)
                    cols = st.columns([3, 1])
                    with cols[0]:
                        with st.expander(f"📎 View {len(sources)} source(s)"):
                            for src in sources:
                                st.markdown(
                                    f"**{src['file']}** — Page {src['page']} "
                                    f"*(relevance: {src.get('similarity_score', 0):.0%})*"
                                )
                                st.caption(f"> {src.get('text_excerpt','')[:250]}…")
                                st.divider()
                    with cols[1]:
                        # Confidence meter
                        st.metric("Confidence", f"{conf:.0f}%")
                        bar_color = (
                            "#22c55e" if conf >= 70
                            else "#f59e0b" if conf >= 40
                            else "#ef4444"
                        )
                        st.markdown(
                            f'<div style="background:#e2e8f0;border-radius:4px;height:8px;">'
                            f'<div style="background:{bar_color};width:{conf}%;'
                            f'height:8px;border-radius:4px;"></div></div>',
                            unsafe_allow_html=True,
                        )

    # ── Chat input ────────────────────────────────────────────────────────────
    # Handle pending question from suggestion buttons
    pending = st.session_state.pop("_pending_question", None)
    prompt = st.chat_input("Ask a question about your documents…") or pending

    if prompt:
        # Show user message immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        # Get answer
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking…"):
                t_start = time.perf_counter()
                result = ask_question(prompt)
                elapsed = time.perf_counter() - t_start

            if result:
                answer = result["answer"]
                sources = result.get("sources", [])
                confidence = result.get("confidence", 0.0)

                st.markdown(answer)

                if sources:
                    cols = st.columns([3, 1])
                    with cols[0]:
                        with st.expander(f"📎 View {len(sources)} source(s)"):
                            for src in sources:
                                st.markdown(
                                    f"**{src['file']}** — Page {src['page']} "
                                    f"*(relevance: {src.get('similarity_score', 0):.0%})*"
                                )
                                st.caption(f"> {src.get('text_excerpt','')[:250]}…")
                                st.divider()
                    with cols[1]:
                        st.metric("Confidence", f"{confidence:.0f}%")
                        bar_color = (
                            "#22c55e" if confidence >= 70
                            else "#f59e0b" if confidence >= 40
                            else "#ef4444"
                        )
                        st.markdown(
                            f'<div style="background:#e2e8f0;border-radius:4px;height:8px;">'
                            f'<div style="background:{bar_color};width:{confidence}%;'
                            f'height:8px;border-radius:4px;"></div></div>',
                            unsafe_allow_html=True,
                        )

                st.caption(f"⏱ Answered in {elapsed:.1f}s")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "confidence": confidence,
                })
            else:
                fallback = "Sorry, I encountered an error. Please try again."
                st.error(fallback)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": fallback,
                })
