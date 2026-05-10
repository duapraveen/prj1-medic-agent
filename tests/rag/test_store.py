import uuid

import chromadb
import pytest

import medic_agent.rag.store as store_module
from medic_agent.rag.store import (
    add_document,
    delete_document,
    document_exists,
    get_document_info,
    list_documents,
)

HIDDEN_DIM = 768

# Single shared ephemeral client for the test session — avoids ChromaDB's
# shared in-memory singleton issue by giving each test its own collection name.
_ephemeral_client = chromadb.EphemeralClient()


@pytest.fixture(autouse=True)
def ephemeral_collection(mocker):
    """Give each test its own isolated in-memory ChromaDB collection."""
    collection = _ephemeral_client.get_or_create_collection(
        name=f"test_{uuid.uuid4().hex}",
        metadata={"hnsw:space": "cosine"},
    )
    mocker.patch.object(store_module, "_get_collection", return_value=collection)
    return collection


def _make_chunks(filename: str, n: int) -> list[dict]:
    return [
        {"text": f"chunk {i} text", "source_filename": filename, "chunk_index": i}
        for i in range(n)
    ]


def _make_embeddings(n: int) -> list[list[float]]:
    return [[0.1] * HIDDEN_DIM for _ in range(n)]


# --- add_document + document_exists ---

def test_document_exists_false_before_add():
    assert document_exists("report.pdf") is False


def test_document_exists_true_after_add():
    chunks = _make_chunks("report.pdf", 3)
    add_document("report.pdf", chunks, _make_embeddings(3))
    assert document_exists("report.pdf") is True


def test_document_exists_false_for_different_doc():
    chunks = _make_chunks("a.pdf", 2)
    add_document("a.pdf", chunks, _make_embeddings(2))
    assert document_exists("b.pdf") is False


# --- list_documents ---

def test_list_documents_empty_initially():
    assert list_documents() == []


def test_list_documents_returns_added_filenames():
    add_document("alpha.pdf", _make_chunks("alpha.pdf", 2), _make_embeddings(2))
    add_document("beta.txt", _make_chunks("beta.txt", 3), _make_embeddings(3))
    docs = list_documents()
    assert set(docs) == {"alpha.pdf", "beta.txt"}


def test_list_documents_no_duplicates():
    # Multiple chunks from same doc should appear only once
    add_document("report.pdf", _make_chunks("report.pdf", 5), _make_embeddings(5))
    docs = list_documents()
    assert docs.count("report.pdf") == 1


# --- delete_document ---

def test_delete_document_removes_it():
    add_document("note.pdf", _make_chunks("note.pdf", 2), _make_embeddings(2))
    delete_document("note.pdf")
    assert document_exists("note.pdf") is False


def test_delete_document_leaves_others_intact():
    add_document("keep.pdf", _make_chunks("keep.pdf", 2), _make_embeddings(2))
    add_document("remove.pdf", _make_chunks("remove.pdf", 2), _make_embeddings(2))
    delete_document("remove.pdf")
    assert document_exists("keep.pdf") is True
    assert document_exists("remove.pdf") is False


def test_delete_nonexistent_document_does_not_raise():
    delete_document("ghost.pdf")  # should not raise


# --- get_document_info ---

def test_get_document_info_empty_initially():
    assert get_document_info() == []


def test_get_document_info_returns_filename_and_chunk_count():
    add_document("info_test.pdf", _make_chunks("info_test.pdf", 4), _make_embeddings(4))
    info = get_document_info()
    assert len(info) == 1
    assert info[0]["filename"] == "info_test.pdf"
    assert info[0]["chunk_count"] == 4


def test_get_document_info_includes_upload_date():
    add_document("dated.pdf", _make_chunks("dated.pdf", 2), _make_embeddings(2))
    info = get_document_info()
    assert info[0]["upload_date"] != "Unknown"


def test_get_document_info_multiple_documents():
    add_document("a.pdf", _make_chunks("a.pdf", 2), _make_embeddings(2))
    add_document("b.pdf", _make_chunks("b.pdf", 3), _make_embeddings(3))
    info = get_document_info()
    assert len(info) == 2
    names = [d["filename"] for d in info]
    assert "a.pdf" in names and "b.pdf" in names
