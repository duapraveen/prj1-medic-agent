import uuid

import chromadb
import pytest

import medic_agent.rag.retriever as retriever_module
from medic_agent.rag.retriever import graph_retrieve, retrieve

HIDDEN_DIM = 768
FAKE_EMBEDDING = [0.1] * HIDDEN_DIM
RETRIEVER_GET_COLLECTION = "medic_agent.rag.retriever._get_collection"
RETRIEVER_EMBED_QUERY = "medic_agent.rag.retriever.embed_query"

_ephemeral_client = chromadb.EphemeralClient()


def _populated_collection(n_chunks: int):
    collection = _ephemeral_client.get_or_create_collection(
        name=f"test_{uuid.uuid4().hex}",
        metadata={"hnsw:space": "cosine"},
    )
    if n_chunks > 0:
        collection.add(
            ids=[str(i) for i in range(n_chunks)],
            embeddings=[[0.1] * HIDDEN_DIM for _ in range(n_chunks)],
            documents=[f"Clinical text chunk {i}" for i in range(n_chunks)],
            metadatas=[
                {"source_filename": "encounter.pdf", "chunk_index": i}
                for i in range(n_chunks)
            ],
        )
    return collection


@pytest.fixture(autouse=True)
def mock_embed(mocker):
    return mocker.patch(RETRIEVER_EMBED_QUERY, return_value=FAKE_EMBEDDING)


# --- basic retrieval ---

def test_retrieve_returns_list_of_dicts(mocker):
    mocker.patch(RETRIEVER_GET_COLLECTION, return_value=_populated_collection(3))
    results = retrieve("what codes apply?")
    assert isinstance(results, list)
    assert all(isinstance(r, dict) for r in results)


def test_retrieve_chunk_has_required_fields(mocker):
    mocker.patch(RETRIEVER_GET_COLLECTION, return_value=_populated_collection(3))
    results = retrieve("diagnosis codes")
    for chunk in results:
        assert "text" in chunk
        assert "source_filename" in chunk
        assert "chunk_index" in chunk


def test_retrieve_default_k_returns_at_most_five(mocker):
    mocker.patch(RETRIEVER_GET_COLLECTION, return_value=_populated_collection(8))
    results = retrieve("query")
    assert len(results) <= 5


def test_retrieve_k_parameter_respected(mocker):
    mocker.patch(RETRIEVER_GET_COLLECTION, return_value=_populated_collection(8))
    results = retrieve("query", k=3)
    assert len(results) == 3


def test_retrieve_returns_fewer_than_k_when_collection_is_small(mocker):
    mocker.patch(RETRIEVER_GET_COLLECTION, return_value=_populated_collection(2))
    results = retrieve("query", k=5)
    assert len(results) == 2


# --- empty collection ---

def test_retrieve_empty_collection_returns_empty_list(mocker):
    mocker.patch(RETRIEVER_GET_COLLECTION, return_value=_populated_collection(0))
    results = retrieve("any query")
    assert results == []


# --- embedder is called with the right query ---

def test_retrieve_passes_query_to_embedder(mocker, mock_embed):
    mocker.patch(RETRIEVER_GET_COLLECTION, return_value=_populated_collection(3))
    retrieve("ICD-10 codes for hypertension")
    mock_embed.assert_called_once_with("ICD-10 codes for hypertension")


# --- graph_retrieve ---

def test_graph_retrieve_returns_empty_for_empty_list():
    result = graph_retrieve([])
    assert result == []


def test_graph_retrieve_formats_entity_context_as_chunk(mocker):
    mocker.patch.object(
        retriever_module.graph_store,
        "get_related_entities",
        return_value=[{"text": "Type 2 diabetes", "entity_type": "Diagnosis", "filename": "enc.pdf"}],
    )
    result = graph_retrieve(["Type 2 diabetes"])
    assert len(result) == 1
    assert result[0]["source_filename"] == "[knowledge-graph]"
    assert result[0]["chunk_index"] == -1
    assert "Type 2 diabetes" in result[0]["text"]
    assert "enc.pdf" in result[0]["text"]


def test_graph_retrieve_returns_empty_when_no_graph_matches(mocker):
    mocker.patch.object(
        retriever_module.graph_store, "get_related_entities", return_value=[]
    )
    result = graph_retrieve(["unknown entity"])
    assert result == []


def test_graph_retrieve_groups_multiple_docs_per_entity(mocker):
    mocker.patch.object(
        retriever_module.graph_store,
        "get_related_entities",
        return_value=[
            {"text": "Hypertension", "entity_type": "Diagnosis", "filename": "a.pdf"},
            {"text": "Hypertension", "entity_type": "Diagnosis", "filename": "b.pdf"},
        ],
    )
    result = graph_retrieve(["Hypertension"])
    assert len(result) == 1
    assert "a.pdf" in result[0]["text"]
    assert "b.pdf" in result[0]["text"]
