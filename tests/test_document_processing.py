"""
Tests for PDF parsing and text chunking.
"""

import io
import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────
def _make_minimal_pdf(text: str = "Hello world. This is a test PDF.") -> bytes:
    """
    Build a minimal valid PDF in memory without any third-party library.
    Suitable only for unit-test purposes.
    """
    stream = text.encode()
    stream_len = len(stream)

    body = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n"
        b"2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1>>\nendobj\n"
        b"3 0 obj\n<</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources <</Font <</F1 5 0 R>>>>>>\nendobj\n"
    )
    content_stream = (
        b"4 0 obj\n<</Length " + str(stream_len + 30).encode() + b">>\nstream\n"
        b"BT /F1 12 Tf 72 720 Td (" + stream + b") Tj ET\nendstream\nendobj\n"
        b"5 0 obj\n<</Type /Font /Subtype /Type1 /BaseFont /Helvetica>>\nendobj\n"
    )
    xref_offset = len(body) + len(content_stream)
    xref = (
        b"xref\n0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000274 00000 n \n"
        b"0000000400 00000 n \n"
    )
    trailer = (
        b"trailer\n<</Size 6 /Root 1 0 R>>\n"
        b"startxref\n" + str(xref_offset).encode() + b"\n%%EOF"
    )
    return body + content_stream + xref + trailer


# ── Text splitter tests ───────────────────────────────────────────────────────
class TestTextSplitter:
    def test_basic_split(self):
        from backend.utils.text_splitter import split_text

        text = "A" * 2500
        chunks = split_text(text, chunk_size=1000, chunk_overlap=200)
        assert len(chunks) >= 2, "Should produce multiple chunks for long text"
        for chunk in chunks:
            assert len(chunk) <= 1100, "Chunk should not vastly exceed chunk_size"

    def test_overlap_maintained(self):
        from backend.utils.text_splitter import split_text

        # Create text with clear sentence breaks every ~200 chars
        sentences = [f"Sentence number {i} ends here. " for i in range(20)]
        text = " ".join(sentences)
        chunks = split_text(text, chunk_size=300, chunk_overlap=100)
        # Adjacent chunks should share some content (overlap)
        if len(chunks) >= 2:
            # Find common words between adjacent chunks
            words_a = set(chunks[0].split())
            words_b = set(chunks[1].split())
            assert len(words_a & words_b) > 0, "Adjacent chunks should share overlapping content"

    def test_empty_text(self):
        from backend.utils.text_splitter import split_text

        chunks = split_text("", chunk_size=1000, chunk_overlap=200)
        assert chunks == [], "Empty text should return empty list"

    def test_short_text_not_split(self):
        from backend.utils.text_splitter import split_text

        text = "Short text that fits in one chunk."
        chunks = split_text(text, chunk_size=1000, chunk_overlap=200)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_no_mid_sentence_split_preference(self):
        from backend.utils.text_splitter import split_text

        # Make a text where ". " boundary falls well before chunk_size
        text = "First sentence is here. " * 30
        chunks = split_text(text, chunk_size=200, chunk_overlap=50)
        for chunk in chunks:
            # Chunks should not end mid-word
            assert chunk == chunk.strip() or chunk.endswith(".")


# ── PDF parser tests ──────────────────────────────────────────────────────────
class TestPDFParser:
    def test_invalid_bytes_raises(self):
        from backend.utils.pdf_parser import extract_pages

        with pytest.raises(ValueError):
            extract_pages(b"not a pdf at all", "fake.pdf")

    def test_clean_text_removes_extra_whitespace(self):
        from backend.utils.pdf_parser import _clean_text

        messy = "Hello   world\n\n\n\nGoodbye   world"
        clean = _clean_text(messy)
        assert "   " not in clean, "Triple spaces should be collapsed"
        assert clean.count("\n\n\n") == 0, "Triple newlines should be collapsed"

    def test_clean_text_preserves_paragraphs(self):
        from backend.utils.pdf_parser import _clean_text

        text = "Para one.\n\nPara two."
        clean = _clean_text(text)
        assert "\n\n" in clean, "Double newline (paragraph break) should be preserved"

    def test_non_breaking_space_removed(self):
        from backend.utils.pdf_parser import _clean_text

        text = "Hello\xa0World"
        clean = _clean_text(text)
        assert "\xa0" not in clean


# ── RAG engine helpers ────────────────────────────────────────────────────────
class TestCitationExtraction:
    def test_extracts_citations(self):
        from backend.services.rag_engine import _extract_citations

        text = (
            "The sky is blue [Source: report.pdf, Page 3]. "
            "Water is wet [Source: science.pdf, Page 12]."
        )
        citations = _extract_citations(text)
        assert len(citations) == 2
        assert citations[0] == ("report.pdf", 3)
        assert citations[1] == ("science.pdf", 12)

    def test_no_citations_returns_empty(self):
        from backend.services.rag_engine import _extract_citations

        assert _extract_citations("No citations here.") == []

    def test_citation_with_spaces_in_filename(self):
        from backend.services.rag_engine import _extract_citations

        text = "See [Source: my document.pdf, Page 7]."
        citations = _extract_citations(text)
        assert len(citations) == 1
        assert citations[0] == ("my document.pdf", 7)


class TestConfidenceScoring:
    def test_high_similarity_high_confidence(self):
        from backend.services.rag_engine import _calculate_confidence

        conf = _calculate_confidence([0.95, 0.92, 0.90], 3)
        assert conf > 80, "High similarity scores should yield high confidence"

    def test_low_similarity_low_confidence(self):
        from backend.services.rag_engine import _calculate_confidence

        conf = _calculate_confidence([0.2, 0.15, 0.1], 1)
        assert conf < 30, "Low similarity scores should yield low confidence"

    def test_empty_scores_returns_zero(self):
        from backend.services.rag_engine import _calculate_confidence

        assert _calculate_confidence([], 0) == 0.0

    def test_confidence_capped_at_100(self):
        from backend.services.rag_engine import _calculate_confidence

        conf = _calculate_confidence([1.0, 1.0, 1.0], 10)
        assert conf <= 100.0


# ── API endpoint tests (uses TestClient) ─────────────────────────────────────
class TestAPIEndpoints:
    @pytest.fixture(autouse=True)
    def client(self):
        """Create a FastAPI TestClient with services initialised."""
        from fastapi.testclient import TestClient
        from backend.main import app
        from backend.api.routes import init_services
        init_services()
        self.client = TestClient(app)

    def test_health_endpoint(self):
        resp = self.client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_upload_rejects_non_pdf(self):
        resp = self.client.post(
            "/api/upload",
            files=[("files", ("test.txt", b"plain text", "text/plain"))],
        )
        assert resp.status_code == 400

    def test_upload_rejects_fake_pdf(self):
        """A file with .pdf extension but invalid content."""
        resp = self.client.post(
            "/api/upload",
            files=[("files", ("test.pdf", b"definitely not pdf bytes", "application/pdf"))],
        )
        assert resp.status_code == 400

    def test_query_unknown_collection(self):
        resp = self.client.post(
            "/api/query",
            json={
                "collection_id": "nonexistent-collection-id",
                "question": "What is this about?",
            },
        )
        assert resp.status_code == 404

    def test_delete_unknown_collection(self):
        resp = self.client.delete("/api/collections/nonexistent-id")
        assert resp.status_code == 404

    def test_get_unknown_collection(self):
        resp = self.client.get("/api/collections/nonexistent-id")
        assert resp.status_code == 404
