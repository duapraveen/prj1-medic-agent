import hashlib
from datetime import datetime, timezone

import kuzu

from medic_agent.config.settings import KUZU_PERSIST_DIR

_db: kuzu.Database | None = None
_conn: kuzu.Connection | None = None


def _get_conn() -> kuzu.Connection:
    global _db, _conn
    if _conn is None:
        _db = kuzu.Database(KUZU_PERSIST_DIR)
        _conn = kuzu.Connection(_db)
        _init_schema(_conn)
    return _conn


def _init_schema(conn: kuzu.Connection) -> None:
    conn.execute(
        "CREATE NODE TABLE IF NOT EXISTS Document("
        "id STRING, filename STRING, ingested_at STRING, PRIMARY KEY(id))"
    )
    conn.execute(
        "CREATE NODE TABLE IF NOT EXISTS Entity("
        "id STRING, text STRING, entity_type STRING, PRIMARY KEY(id))"
    )
    conn.execute(
        "CREATE REL TABLE IF NOT EXISTS APPEARS_IN("
        "FROM Entity TO Document, chunk_id STRING, chunk_index INT64)"
    )


def _entity_id(text: str) -> str:
    return hashlib.md5(text.lower().strip().encode()).hexdigest()


def upsert_document(doc_id: str, filename: str) -> None:
    conn = _get_conn()
    ts = datetime.now(timezone.utc).isoformat()
    result = conn.execute(
        "MATCH (d:Document {id: $id}) RETURN COUNT(*) AS cnt",
        parameters={"id": doc_id},
    )
    if result.get_next()[0] == 0:
        conn.execute(
            "CREATE (:Document {id: $id, filename: $filename, ingested_at: $ts})",
            parameters={"id": doc_id, "filename": filename, "ts": ts},
        )


def upsert_entities(
    doc_id: str, entities: list[dict], chunk_id: str, chunk_index: int
) -> None:
    conn = _get_conn()
    for entity in entities:
        eid = _entity_id(entity["text"])
        result = conn.execute(
            "MATCH (e:Entity {id: $id}) RETURN COUNT(*) AS cnt",
            parameters={"id": eid},
        )
        if result.get_next()[0] == 0:
            conn.execute(
                "CREATE (:Entity {id: $id, text: $text, entity_type: $etype})",
                parameters={"id": eid, "text": entity["text"], "etype": entity["entity_type"]},
            )
        conn.execute(
            "MATCH (e:Entity {id: $eid}), (d:Document {id: $did})"
            " CREATE (e)-[:APPEARS_IN {chunk_id: $chunk_id, chunk_index: $cidx}]->(d)",
            parameters={
                "eid": eid, "did": doc_id, "chunk_id": chunk_id, "cidx": chunk_index,
            },
        )


def get_related_entities(entity_texts: list[str]) -> list[dict]:
    if not entity_texts:
        return []
    conn = _get_conn()
    conditions = " OR ".join(f"e.text = $t{i}" for i in range(len(entity_texts)))
    params = {f"t{i}": t for i, t in enumerate(entity_texts)}
    result = conn.execute(
        f"MATCH (e:Entity)-[:APPEARS_IN]->(d:Document)"
        f" WHERE {conditions}"
        f" RETURN e.text, e.entity_type, d.filename LIMIT 50",
        parameters=params,
    )
    rows = []
    while result.has_next():
        row = result.get_next()
        rows.append({"text": row[0], "entity_type": row[1], "filename": row[2]})
    return rows


def delete_document_entities(doc_id: str) -> None:
    conn = _get_conn()
    result = conn.execute(
        "MATCH (e:Entity)-[:APPEARS_IN]->(d:Document {id: $id}) RETURN DISTINCT e.id",
        parameters={"id": doc_id},
    )
    entity_ids = []
    while result.has_next():
        entity_ids.append(result.get_next()[0])

    conn.execute(
        "MATCH (:Entity)-[r:APPEARS_IN]->(d:Document {id: $id}) DELETE r",
        parameters={"id": doc_id},
    )
    for eid in entity_ids:
        r = conn.execute(
            "MATCH (e:Entity {id: $id})-[:APPEARS_IN]->() RETURN COUNT(*) AS cnt",
            parameters={"id": eid},
        )
        if r.get_next()[0] == 0:
            conn.execute(
                "MATCH (e:Entity {id: $id}) DELETE e",
                parameters={"id": eid},
            )
    conn.execute(
        "MATCH (d:Document {id: $id}) DELETE d",
        parameters={"id": doc_id},
    )
