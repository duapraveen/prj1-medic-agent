import numpy as np
from huggingface_hub import InferenceClient

from medic_agent.config.settings import EMBEDDING_MODEL, HUGGINGFACE_API_KEY

_client = InferenceClient(token=HUGGINGFACE_API_KEY)


def _mean_pool(token_embeddings) -> list[float]:
    # HuggingFace feature_extraction returns [seq_len, hidden] for a single text.
    # Mean pool across the token dimension → one 768-dim sentence vector.
    arr = np.array(token_embeddings, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]  # strip batch dim if present
    return arr.mean(axis=0).tolist()


def embed_query(query: str) -> list[float]:
    result = _client.feature_extraction(query, model=EMBEDDING_MODEL)
    return _mean_pool(result)


def embed_texts(texts: list[str]) -> list[list[float]]:
    return [embed_query(t) for t in texts]
