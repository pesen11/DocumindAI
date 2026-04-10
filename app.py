"""
DocuMind AI — Unified Streamlit app for Streamlit Cloud deployment.

All backend services are called directly (no separate FastAPI server needed).
Secrets are read from st.secrets (set via Streamlit Cloud dashboard or
.streamlit/secrets.toml locally).

Run locally:
    streamlit run app.py
"""

from __future__ import annotations
import os
import sys
import time
from datetime import datetime
from typing import Any

# ── Inject secrets into environment BEFORE importing backend ──────────────────
# This must happen before any backend module is imported (they read env at load time).
import streamlit as st

def _load_secrets() -> None:
    """Push Streamlit secrets into os.environ so backend config can read them."""
    secret_keys = [
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "LLM_PROVIDER",
        "LLM_MODEL",
        "LLM_TEMPERATURE",
        "MAX_TOKENS",
        "LLM_TIMEOUT",
        "EMBEDDING_MODEL",
        "CHUNK_SIZE",
        "CHUNK_OVERLAP",
        "TOP_K",
        "MAX_FILE_SIZE",
    ]
    try:
        for key in secret_keys:
            if key in st.secrets and key not in os.environ:
                os.environ[key] = str(st.secrets[key])
    except Exception:
        pass  # st.secrets not available (e.g. running tests)

_load_secrets()

# Force in-memory ChromaDB — no filesystem persistence needed for cloud
os.environ.setdefault("CHROMA_USE_EPHEMERAL", "true")

# Ensure backend package is importable when running from repo root
sys.path.insert(0, os.path.dirname(__file__))

# ── Backend imports ───────────────────────────────────────────────────────────
from backend.services.vector_store import VectorStore
from backend.services.document_processor import DocumentProcessor
from backend.services.rag_engine import RAGEngine, generate_query_suggestions
from backend.models.query import ConversationTurn
from backend.config import MAX_FILE_SIZE

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
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Cached service singletons ─────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI models…")
def get_services() -> tuple[VectorStore, DocumentProcessor, RAGEngine]:
    """
    Initialise heavy services once per app process.
    st.cache_resource keeps these alive across all user sessions and reruns.
    """
    vs = VectorStore()
    dp = DocumentProcessor(vs)
    rag = RAGEngine(vs)
    return vs, dp, rag


vector_store, doc_processor, rag_engine = get_services()

# ── Session state initialisation ──────────────────────────────────────────────
def _init_state() -> None:
    defaults: dict[str, Any] = {
        "collection_id": None,
        "messages": [],
        "uploaded_docs": [],
        "top_k": 5,
        "temperature": 0.3,
        "suggested_questions": [],
        "processing": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ── Direct service helpers (replace HTTP API calls) ───────────────────────────
def upload_files(files: list) -> bool:
    """Process uploaded files directly via the document processor."""
    file_tuples = [
        (f.name, f.getvalue()) for f in files
        if f.name.lower().endswith(".pdf")
    ]
    if not file_tuples:
        st.error("Please upload PDF files only.")
        return False

    # Validate file sizes
    for name, data in file_tuples:
        if len(data) > MAX_FILE_SIZE:
            size_mb = MAX_FILE_SIZE // (1024 * 1024)
            st.error(f"'{name}' exceeds the {size_mb} MB limit.")
            return False
        if not data.startswith(b"%PDF"):
            st.error(f"'{name}' does not appear to be a valid PDF.")
            return False

    try:
        info = doc_processor.process_pdfs(
            file_tuples,
            collection_id=st.session_state.collection_id,
        )
    except ValueError as exc:
        st.error(str(exc))
        return False
    except Exception as exc:
        st.error(f"Processing failed: {exc}")
        return False

    st.session_state.collection_id = info.collection_id

    # Generate suggested questions (best-effort)
    summaries = [{"file_name": f.file_name, "excerpt": ""} for f in info.files]
    st.session_state.suggested_questions = generate_query_suggestions(summaries)

    existing = {d["name"] for d in st.session_state.uploaded_docs}
    for f in info.files:
        if f.file_name not in existing:
            st.session_state.uploaded_docs.append({
                "name": f.file_name,
                "chunks": f.chunk_count,
            })

    return True


def ask_question(question: str) -> dict[str, Any] | None:
    """Run the RAG pipeline directly."""
    if not st.session_state.collection_id:
        st.error("No documents loaded.")
        return None

    history = [
        ConversationTurn(role=m["role"], content=m["content"])
        for m in st.session_state.messages[-20:]
    ]

    try:
        response = rag_engine.query(
            collection_id=st.session_state.collection_id,
            question=question,
            conversation_history=history,
            top_k=st.session_state.top_k,
            temperature=st.session_state.temperature,
        )
    except Exception as exc:
        st.error(f"Query failed: {exc}")
        return None

    return {
        "answer": response.answer,
        "sources": [s.model_dump() for s in response.sources],
        "confidence": response.confidence,
        "model_used": response.model_used,
    }


def delete_collection() -> None:
    """Delete the current collection from vector store."""
    if st.session_state.collection_id:
        vector_store.delete_collection(st.session_state.collection_id)


# ── Export helper ─────────────────────────────────────────────────────────────
def _export_as_markdown() -> str:
    lines = [
        "# DocuMind AI — Conversation Export",
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
    st.image("https://img.icons8.com/fluency/96/books.png", width=60)
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
            st.markdown(f"📄 `{doc['name']}`")
        if st.button("🗑 Remove All Documents", use_container_width=True):
            delete_collection()
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
        st.rerun()

    if st.session_state.messages:
        st.download_button(
            "⬇️ Export as Markdown",
            data=_export_as_markdown(),
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

    # ── Config check ──────────────────────────────────────────────────────────
    st.divider()
    from backend.config import LLM_PROVIDER, LLM_MODEL
    if LLM_PROVIDER != "none":
        st.success(f"LLM: {LLM_MODEL}", icon="✅")
    else:
        st.warning("No API key configured. Set ANTHROPIC_API_KEY or OPENAI_API_KEY in Streamlit secrets.", icon="⚠️")


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
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])

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
    pending = st.session_state.pop("_pending_question", None)
    prompt = st.chat_input("Ask a question about your documents…") or pending

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

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
