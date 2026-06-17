from medic_agent.rag import graph_store
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


def graph_retrieve(entity_texts: list[str]) -> list[dict]:
    if not entity_texts:
        return []
    rows = graph_store.get_related_entities(entity_texts)
    if not rows:
        return []
    by_entity: dict[str, list[str]] = {}
    for row in rows:
        key = f"{row['text']} ({row['entity_type']})"
        by_entity.setdefault(key, []).append(row["filename"])
    lines = [
        f"- {entity} → documented in: {', '.join(sorted(set(filenames)))}"
        for entity, filenames in by_entity.items()
    ]
    return [{
        "text": "Knowledge graph entity context:\n" + "\n".join(lines),
        "source_filename": "[knowledge-graph]",
        "chunk_index": -1,
    }]
