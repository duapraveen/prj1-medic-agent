import numpy as np
import pytest

from medic_agent.rag.embedder import embed_query, embed_texts

HIDDEN_DIM = 768

def _fake_token_embeddings(seq_len: int = 6) -> list:
    # Shape [seq_len, HIDDEN_DIM] — simulates HuggingFace feature_extraction output
    rng = np.random.default_rng(42)
    return rng.random((seq_len, HIDDEN_DIM)).tolist()


# --- embed_query ---

def test_embed_query_returns_list_of_floats(mocker):
    mocker.patch(
        "medic_agent.rag.embedder._client.feature_extraction",
        return_value=_fake_token_embeddings(),
    )
    result = embed_query("myocardial infarction")
    assert isinstance(result, list)
    assert all(isinstance(v, float) for v in result)


def test_embed_query_returns_correct_dimension(mocker):
    mocker.patch(
        "medic_agent.rag.embedder._client.feature_extraction",
        return_value=_fake_token_embeddings(),
    )
    result = embed_query("myocardial infarction")
    assert len(result) == HIDDEN_DIM


def test_embed_query_mean_pools_tokens(mocker):
    # Use a 2-token sequence with known values; verify output is their mean
    token_embeddings = [[1.0] * HIDDEN_DIM, [3.0] * HIDDEN_DIM]
    mocker.patch(
        "medic_agent.rag.embedder._client.feature_extraction",
        return_value=token_embeddings,
    )
    result = embed_query("test")
    assert abs(result[0] - 2.0) < 1e-5


def test_embed_query_handles_batch_dim(mocker):
    # Some HuggingFace endpoints wrap output in an extra batch dimension [1, seq_len, hidden]
    token_embeddings = [_fake_token_embeddings(seq_len=4)]  # shape [1, 4, 768]
    mocker.patch(
        "medic_agent.rag.embedder._client.feature_extraction",
        return_value=token_embeddings,
    )
    result = embed_query("test")
    assert len(result) == HIDDEN_DIM


def test_embed_query_calls_correct_model(mocker):
    mock_fe = mocker.patch(
        "medic_agent.rag.embedder._client.feature_extraction",
        return_value=_fake_token_embeddings(),
    )
    embed_query("I21.9")
    _, kwargs = mock_fe.call_args
    assert kwargs["model"] == "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"


# --- embed_texts ---

def test_embed_texts_returns_one_vector_per_text(mocker):
    mocker.patch(
        "medic_agent.rag.embedder._client.feature_extraction",
        return_value=_fake_token_embeddings(),
    )
    results = embed_texts(["text one", "text two", "text three"])
    assert len(results) == 3


def test_embed_texts_each_vector_correct_dimension(mocker):
    mocker.patch(
        "medic_agent.rag.embedder._client.feature_extraction",
        return_value=_fake_token_embeddings(),
    )
    results = embed_texts(["a", "b"])
    for vec in results:
        assert len(vec) == HIDDEN_DIM


def test_embed_texts_empty_list_returns_empty(mocker):
    mocker.patch("medic_agent.rag.embedder._client.feature_extraction")
    results = embed_texts([])
    assert results == []
