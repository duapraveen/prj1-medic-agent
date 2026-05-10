import chromadb
from datetime import datetime, timezone

from medic_agent.config.settings import CHROMA_PERSIST_DIR

COLLECTION_NAME = "medic_documents"

_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def add_document(doc_id: str, chunks: list[dict], embeddings: list[list[float]]) -> None:
    collection = _get_collection()
    upload_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    ids = [f"{doc_id}::{chunk['chunk_index']}" for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [
        {
            "source_filename": chunk["source_filename"],
            "chunk_index": chunk["chunk_index"],
            "upload_date": upload_date,
        }
        for chunk in chunks
    ]
    collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)


def get_document_info() -> list[dict]:
    """Returns [{filename, chunk_count, upload_date}] for every stored document."""
    collection = _get_collection()
    result = collection.get(include=["metadatas"])
    docs: dict[str, dict] = {}
    for meta in result["metadatas"]:
        name = meta["source_filename"]
        if name not in docs:
            docs[name] = {
                "filename": name,
                "chunk_count": 0,
                "upload_date": meta.get("upload_date", "Unknown"),
            }
        docs[name]["chunk_count"] += 1
    return sorted(docs.values(), key=lambda d: d["filename"])


def document_exists(doc_id: str) -> bool:
    collection = _get_collection()
    result = collection.get(
        where={"source_filename": {"$eq": doc_id}},
        limit=1,
        include=[],
    )
    return len(result["ids"]) > 0


def list_documents() -> list[str]:
    collection = _get_collection()
    result = collection.get(include=["metadatas"])
    seen = set()
    docs = []
    for meta in result["metadatas"]:
        name = meta["source_filename"]
        if name not in seen:
            seen.add(name)
            docs.append(name)
    return docs


def delete_document(doc_id: str) -> None:
    collection = _get_collection()
    collection.delete(where={"source_filename": {"$eq": doc_id}})
