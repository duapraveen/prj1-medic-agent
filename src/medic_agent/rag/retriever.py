from medic_agent.rag.embedder import embed_query
from medic_agent.rag.store import _get_collection


def retrieve(query: str, k: int = 5) -> list[dict]:
    collection = _get_collection()
    count = collection.count()
    if count == 0:
        return []

    embedding = embed_query(query)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(k, count),
        include=["documents", "metadatas"],
    )

    chunks = []
    for text, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append(
            {
                "text": text,
                "source_filename": meta["source_filename"],
                "chunk_index": meta["chunk_index"],
            }
        )
    return chunks
