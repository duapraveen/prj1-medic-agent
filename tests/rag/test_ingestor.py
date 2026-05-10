import io
from unittest.mock import MagicMock, patch

import pytest

from medic_agent.rag.ingestor import CHUNK_OVERLAP, CHUNK_SIZE, load_pdf, load_text


# --- load_text ---

def test_load_text_returns_chunks():
    text = "A" * 2500
    chunks = load_text(text, "test.txt")
    assert len(chunks) > 1


def test_load_text_chunk_fields():
    chunks = load_text("Hello world " * 10, "note.txt")
    for chunk in chunks:
        assert "text" in chunk
        assert chunk["source_filename"] == "note.txt"
        assert "chunk_index" in chunk


def test_load_text_chunk_indices_sequential():
    text = "B" * 3000
    chunks = load_text(text, "test.txt")
    for i, chunk in enumerate(chunks):
        assert chunk["chunk_index"] == i


def test_load_text_chunk_size_at_most_chunk_size():
    text = "C" * 5000
    chunks = load_text(text, "test.txt")
    for chunk in chunks:
        assert len(chunk["text"]) <= CHUNK_SIZE


def test_load_text_overlap_present():
    # The end of chunk N should appear at the start of chunk N+1
    text = "D" * 3000
    chunks = load_text(text, "test.txt")
    assert len(chunks) >= 2
    tail = chunks[0]["text"][-CHUNK_OVERLAP:]
    head = chunks[1]["text"][:CHUNK_OVERLAP]
    assert tail == head


def test_load_text_empty_string_returns_empty():
    chunks = load_text("", "empty.txt")
    assert chunks == []


def test_load_text_short_text_is_single_chunk():
    text = "Short clinical note."
    chunks = load_text(text, "short.txt")
    assert len(chunks) == 1
    assert chunks[0]["text"] == text


# --- load_pdf ---

def _make_mock_reader(pages_text: list[str]):
    mock_pages = []
    for t in pages_text:
        page = MagicMock()
        page.extract_text.return_value = t
        mock_pages.append(page)
    mock_reader = MagicMock()
    mock_reader.pages = mock_pages
    return mock_reader


def test_load_pdf_returns_chunks(mocker):
    mocker.patch(
        "medic_agent.rag.ingestor.PdfReader",
        return_value=_make_mock_reader(["A" * 1200, "B" * 1200]),
    )
    chunks = load_pdf(b"fake-pdf-bytes", "report.pdf")
    assert len(chunks) >= 2


def test_load_pdf_chunk_has_correct_filename(mocker):
    mocker.patch(
        "medic_agent.rag.ingestor.PdfReader",
        return_value=_make_mock_reader(["Some clinical text " * 20]),
    )
    chunks = load_pdf(b"fake-pdf-bytes", "encounter.pdf")
    for chunk in chunks:
        assert chunk["source_filename"] == "encounter.pdf"


def test_load_pdf_handles_empty_page(mocker):
    mocker.patch(
        "medic_agent.rag.ingestor.PdfReader",
        return_value=_make_mock_reader(["", "Actual content " * 10, ""]),
    )
    chunks = load_pdf(b"fake-pdf-bytes", "mixed.pdf")
    assert len(chunks) >= 1
    assert all(chunk["text"].strip() for chunk in chunks)
