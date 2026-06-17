# Knowledge Graph Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Kuzu embedded knowledge graph alongside ChromaDB so that entities and their cross-document relationships are indexed at ingest time and used as a second retrieval path (alongside vector similarity) during agent runs.

**Architecture:** Every chunk ingested into ChromaDB also runs through an entity extractor (Haiku LLM call) that produces structured `{text, entity_type}` records; those records are written into a Kuzu embedded graph as `Entity` and `Document` nodes connected by `APPEARS_IN` edges. At query time each agent's `retrieve_node` calls both `retrieve()` (vector) and `graph_retrieve()` (graph), then concatenates the results as context for the LLM. Deleting a document from ChromaDB also removes its graph nodes and prunes any orphaned entities.

**Tech Stack:** Kuzu ≥ 0.6 (embedded graph DB, openCypher, Python bindings), existing LiteLLM `complete()` for entity extraction, pytest-mock for unit tests.

---

## File Map

| Action | File | Responsibility |
|---|---|---|
| Create | `src/medic_agent/rag/graph_store.py` | Kuzu schema init, node/edge CRUD, entity query |
| Create | `src/medic_agent/rag/entity_extractor.py` | LLM → JSON entity list from chunk text |
| Create | `tests/rag/test_graph_store.py` | graph_store unit tests (real Kuzu, temp dir) |
| Create | `tests/rag/test_entity_extractor.py` | entity_extractor unit tests (mocked LLM) |
| Modify | `pyproject.toml` | Add `kuzu` dependency |
| Modify | `src/medic_agent/config/settings.py` | Add `KUZU_PERSIST_DIR`, `ENTITY_EXTRACTOR_MODEL_ID` |
| Modify | `src/medic_agent/rag/store.py` | Call graph_store + entity_extractor in add/delete |
| Modify | `src/medic_agent/rag/retriever.py` | Add `graph_retrieve()` |
| Modify | `src/medic_agent/agents/coding_agent.py` | Hybrid retrieve_node (vector + graph) |
| Modify | `src/medic_agent/agents/ambient_agent.py` | Hybrid retrieve_node (vector + graph) |
| Modify | `tests/rag/test_store.py` | Mock graph_store + entity_extractor in autouse fixture |
| Modify | `tests/agents/test_coding_agent.py` | Mock graph_retrieve |
| Modify | `tests/agents/test_ambient_agent.py` | Mock graph_retrieve + extract_entities |

---

## Task 1: Add Kuzu dependency and config constants

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/medic_agent/config/settings.py`

- [ ] **Step 1: Add kuzu to pyproject.toml**

In `pyproject.toml`, add `"kuzu>=0.6",` to the `dependencies` list (after `langgraph`):

```toml
dependencies = [
    "chromadb>=1.5.9",
    "huggingface-hub>=1.14.0",
    "kuzu>=0.6",
    "langfuse>=4.6.1",
    "langgraph>=1.1.10",
    "litellm>=1.83.14",
    "pypdf>=6.11.0",
    "python-dotenv>=1.2.2",
    "ragas>=0.4.3",
    "streamlit>=1.57.0",
]
```

- [ ] **Step 2: Install the dependency**

```bash
uv add kuzu
uv run python -c "import kuzu; print(kuzu.__version__)"
```

Expected: prints a version string like `0.6.x`, no import error.

- [ ] **Step 3: Add settings constants**

In `src/medic_agent/config/settings.py`, add after the `CHROMA_PERSIST_DIR` line in the `# --- Paths ---` block:

```python
KUZU_PERSIST_DIR = str(DATA_DIR / "kuzu")
```

And add after `ROUTER_MODEL_ID = ...` in the `# --- V2 ---` block:

```python
ENTITY_EXTRACTOR_MODEL_ID = AVAILABLE_MODELS["Claude Haiku (Fast)"]
```

- [ ] **Step 4: Verify import**

```bash
uv run python -c "from medic_agent.config.settings import KUZU_PERSIST_DIR, ENTITY_EXTRACTOR_MODEL_ID; print(KUZU_PERSIST_DIR, ENTITY_EXTRACTOR_MODEL_ID)"
```

Expected: `data/kuzu claude-haiku-4-5-20251001`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock src/medic_agent/config/settings.py
git commit -m "feat(kg): add kuzu dependency and graph config constants"
```

---

## Task 2: graph_store.py + tests

**Files:**
- Create: `src/medic_agent/rag/graph_store.py`
- Create: `tests/rag/test_graph_store.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/rag/test_graph_store.py`:

```python
import pytest
import medic_agent.rag.graph_store as gs


@pytest.fixture(autouse=True)
def fresh_graph(tmp_path, mocker):
    """Each test gets an isolated Kuzu DB in a temp dir."""
    import medic_agent.rag.graph_store as gs_mod
    gs_mod._db = None
    gs_mod._conn = None
    mocker.patch("medic_agent.rag.graph_store.KUZU_PERSIST_DIR", str(tmp_path / "kuzu"))
    yield
    gs_mod._db = None
    gs_mod._conn = None


def test_upsert_document_does_not_raise():
    gs.upsert_document("note.pdf", "note.pdf")


def test_upsert_entities_and_get_related():
    gs.upsert_document("note.pdf", "note.pdf")
    entities = [{"text": "Type 2 diabetes", "entity_type": "Diagnosis"}]
    gs.upsert_entities("note.pdf", entities, "note.pdf::0", 0)
    rows = gs.get_related_entities(["Type 2 diabetes"])
    assert len(rows) == 1
    assert rows[0]["text"] == "Type 2 diabetes"
    assert rows[0]["entity_type"] == "Diagnosis"
    assert rows[0]["filename"] == "note.pdf"


def test_get_related_entities_empty_list_returns_empty():
    result = gs.get_related_entities([])
    assert result == []


def test_get_related_entities_no_match_returns_empty():
    gs.upsert_document("note.pdf", "note.pdf")
    result = gs.get_related_entities(["nonexistent entity"])
    assert result == []


def test_delete_document_entities_removes_entity_and_doc():
    gs.upsert_document("note.pdf", "note.pdf")
    gs.upsert_entities(
        "note.pdf", [{"text": "Hypertension", "entity_type": "Diagnosis"}], "note.pdf::0", 0
    )
    gs.delete_document_entities("note.pdf")
    result = gs.get_related_entities(["Hypertension"])
    assert result == []


def test_same_entity_links_to_multiple_docs():
    gs.upsert_document("a.pdf", "a.pdf")
    gs.upsert_document("b.pdf", "b.pdf")
    entity = {"text": "Type 2 diabetes", "entity_type": "Diagnosis"}
    gs.upsert_entities("a.pdf", [entity], "a.pdf::0", 0)
    gs.upsert_entities("b.pdf", [entity], "b.pdf::0", 0)
    rows = gs.get_related_entities(["Type 2 diabetes"])
    filenames = {r["filename"] for r in rows}
    assert "a.pdf" in filenames and "b.pdf" in filenames


def test_delete_one_doc_keeps_entity_linked_to_other():
    gs.upsert_document("a.pdf", "a.pdf")
    gs.upsert_document("b.pdf", "b.pdf")
    entity = {"text": "Hypertension", "entity_type": "Diagnosis"}
    gs.upsert_entities("a.pdf", [entity], "a.pdf::0", 0)
    gs.upsert_entities("b.pdf", [entity], "b.pdf::0", 0)
    gs.delete_document_entities("a.pdf")
    rows = gs.get_related_entities(["Hypertension"])
    assert any(r["filename"] == "b.pdf" for r in rows)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/rag/test_graph_store.py -v
```

Expected: `ImportError: cannot import name 'graph_store'` or `ModuleNotFoundError`.

- [ ] **Step 3: Implement graph_store.py**

Create `src/medic_agent/rag/graph_store.py`:

```python
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
                "eid": eid, "did": doc_id, "chunk_id": chunk_id, "cidx": chunk_index
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/rag/test_graph_store.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/medic_agent/rag/graph_store.py tests/rag/test_graph_store.py
git commit -m "feat(kg): add graph_store with Kuzu schema, CRUD, and entity query"
```

---

## Task 3: entity_extractor.py + tests

**Files:**
- Create: `src/medic_agent/rag/entity_extractor.py`
- Create: `tests/rag/test_entity_extractor.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/rag/test_entity_extractor.py`:

```python
import medic_agent.rag.entity_extractor as extractor_module


def test_extract_entities_parses_valid_json(mocker):
    mocker.patch.object(
        extractor_module,
        "complete",
        return_value=(
            '{"entities": [{"text": "Type 2 diabetes", "entity_type": "Diagnosis"}]}',
            {},
        ),
    )
    result = extractor_module.extract_entities("Patient has type 2 diabetes.")
    assert result == [{"text": "Type 2 diabetes", "entity_type": "Diagnosis"}]


def test_extract_entities_strips_markdown_fences(mocker):
    mocker.patch.object(
        extractor_module,
        "complete",
        return_value=(
            '```json\n{"entities": [{"text": "Metformin", "entity_type": "Medication"}]}\n```',
            {},
        ),
    )
    result = extractor_module.extract_entities("Patient takes Metformin.")
    assert result == [{"text": "Metformin", "entity_type": "Medication"}]


def test_extract_entities_returns_empty_on_bad_json(mocker):
    mocker.patch.object(extractor_module, "complete", return_value=("not valid json", {}))
    result = extractor_module.extract_entities("some clinical text")
    assert result == []


def test_extract_entities_filters_items_with_non_string_fields(mocker):
    mocker.patch.object(
        extractor_module,
        "complete",
        return_value=(
            '{"entities": [{"text": 999, "entity_type": "Diagnosis"}, '
            '{"text": "Hypertension", "entity_type": "Diagnosis"}]}',
            {},
        ),
    )
    result = extractor_module.extract_entities("...")
    assert result == [{"text": "Hypertension", "entity_type": "Diagnosis"}]


def test_extract_entities_uses_entity_extractor_model(mocker):
    from medic_agent.config.settings import ENTITY_EXTRACTOR_MODEL_ID

    mock_complete = mocker.patch.object(
        extractor_module, "complete", return_value=('{"entities": []}', {})
    )
    extractor_module.extract_entities("some text")
    model_used = mock_complete.call_args[0][0]
    assert model_used == ENTITY_EXTRACTOR_MODEL_ID


def test_extract_entities_truncates_long_input(mocker):
    mock_complete = mocker.patch.object(
        extractor_module, "complete", return_value=('{"entities": []}', {})
    )
    long_text = "x" * 5000
    extractor_module.extract_entities(long_text)
    user_input_sent = mock_complete.call_args[0][2]
    assert len(user_input_sent) <= 2000
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/rag/test_entity_extractor.py -v
```

Expected: `ModuleNotFoundError: No module named 'medic_agent.rag.entity_extractor'`

- [ ] **Step 3: Implement entity_extractor.py**

Create `src/medic_agent/rag/entity_extractor.py`:

```python
import json
import re

from medic_agent.config.settings import ENTITY_EXTRACTOR_MODEL_ID
from medic_agent.llm.client import complete

_SYSTEM_PROMPT = """\
Extract medical entities from clinical text.
Return ONLY valid JSON (no markdown):
{"entities": [{"text": "...", "entity_type": "Diagnosis|Medication|Procedure|Finding|Anatomy|Provider"}]}
Extract only explicitly documented entities. Do not infer. Return {"entities": []} if none found.\
"""

_MAX_INPUT_CHARS = 2000


def extract_entities(chunk_text: str) -> list[dict]:
    text, _ = complete(ENTITY_EXTRACTOR_MODEL_ID, _SYSTEM_PROMPT, chunk_text[:_MAX_INPUT_CHARS])
    clean = re.sub(r"```(?:json)?\n?|```", "", text).strip()
    try:
        data = json.loads(clean)
        entities = data.get("entities", [])
        return [
            e for e in entities
            if isinstance(e.get("text"), str) and isinstance(e.get("entity_type"), str)
        ]
    except (json.JSONDecodeError, AttributeError):
        return []
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/rag/test_entity_extractor.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/medic_agent/rag/entity_extractor.py tests/rag/test_entity_extractor.py
git commit -m "feat(kg): add entity_extractor — Haiku LLM call → JSON entity list per chunk"
```

---

## Task 4: Wire graph into store.add_document() and delete_document()

**Files:**
- Modify: `src/medic_agent/rag/store.py`
- Modify: `tests/rag/test_store.py`

- [ ] **Step 1: Update test_store.py autouse fixture to mock graph calls**

In `tests/rag/test_store.py`, update the `ephemeral_collection` fixture to also mock the graph_store and entity_extractor calls that `add_document` and `delete_document` will now make. Replace the existing fixture:

```python
@pytest.fixture(autouse=True)
def ephemeral_collection(mocker):
    """Give each test its own isolated in-memory ChromaDB collection."""
    collection = _ephemeral_client.get_or_create_collection(
        name=f"test_{uuid.uuid4().hex}",
        metadata={"hnsw:space": "cosine"},
    )
    mocker.patch.object(store_module, "_get_collection", return_value=collection)
    mocker.patch("medic_agent.rag.store.graph_store.upsert_document")
    mocker.patch("medic_agent.rag.store.graph_store.upsert_entities")
    mocker.patch("medic_agent.rag.store.graph_store.delete_document_entities")
    mocker.patch(
        "medic_agent.rag.store.entity_extractor.extract_entities", return_value=[]
    )
    return collection
```

Also add `import medic_agent.rag.store as store_module` at the top if it doesn't already exist (the file already has `import medic_agent.rag.store as store_module` at line 6).

- [ ] **Step 2: Run existing store tests to confirm they still pass before the change**

```bash
uv run pytest tests/rag/test_store.py -v
```

Expected: all tests PASS (mocks are in place before the implementation changes).

- [ ] **Step 3: Update store.py to wire in graph calls**

In `src/medic_agent/rag/store.py`, add two imports at the top after the existing imports:

```python
from medic_agent.rag import entity_extractor, graph_store
```

Then replace `add_document` with:

```python
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

    graph_store.upsert_document(doc_id, doc_id)
    for chunk in chunks:
        chunk_id = f"{doc_id}::{chunk['chunk_index']}"
        entities = entity_extractor.extract_entities(chunk["text"])
        graph_store.upsert_entities(doc_id, entities, chunk_id, chunk["chunk_index"])
```

And replace `delete_document` with:

```python
def delete_document(doc_id: str) -> None:
    collection = _get_collection()
    collection.delete(where={"source_filename": {"$eq": doc_id}})
    graph_store.delete_document_entities(doc_id)
```

- [ ] **Step 4: Run all store tests to verify they still pass**

```bash
uv run pytest tests/rag/test_store.py -v
```

Expected: all tests PASS. The graph calls are mocked so no real Kuzu or LLM calls are made.

- [ ] **Step 5: Commit**

```bash
git add src/medic_agent/rag/store.py tests/rag/test_store.py
git commit -m "feat(kg): wire entity extraction and graph upsert into store.add_document"
```

---

## Task 5: graph_retrieve() in retriever.py + tests

**Files:**
- Modify: `src/medic_agent/rag/retriever.py`
- Modify: `tests/rag/test_retriever.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/rag/test_retriever.py`:

```python
import medic_agent.rag.retriever as retriever_module
from medic_agent.rag.retriever import graph_retrieve


def test_graph_retrieve_returns_empty_for_empty_list():
    result = graph_retrieve([])
    assert result == []


def test_graph_retrieve_formats_entity_context_as_chunk(mocker):
    mocker.patch.object(
        retriever_module.graph_store,
        "get_related_entities",
        return_value=[
            {"text": "Type 2 diabetes", "entity_type": "Diagnosis", "filename": "enc.pdf"},
        ],
    )
    result = graph_retrieve(["Type 2 diabetes"])
    assert len(result) == 1
    chunk = result[0]
    assert chunk["source_filename"] == "[knowledge-graph]"
    assert chunk["chunk_index"] == -1
    assert "Type 2 diabetes" in chunk["text"]
    assert "enc.pdf" in chunk["text"]


def test_graph_retrieve_returns_empty_when_no_graph_matches(mocker):
    mocker.patch.object(
        retriever_module.graph_store,
        "get_related_entities",
        return_value=[],
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
```

- [ ] **Step 2: Run new tests to verify they fail**

```bash
uv run pytest tests/rag/test_retriever.py::test_graph_retrieve_returns_empty_for_empty_list -v
```

Expected: `ImportError: cannot import name 'graph_retrieve' from 'medic_agent.rag.retriever'`

- [ ] **Step 3: Implement graph_retrieve in retriever.py**

In `src/medic_agent/rag/retriever.py`, add an import at the top:

```python
from medic_agent.rag import graph_store
```

Then append `graph_retrieve` after the existing `retrieve` function:

```python
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
    return [
        {
            "text": "Knowledge graph entity context:\n" + "\n".join(lines),
            "source_filename": "[knowledge-graph]",
            "chunk_index": -1,
        }
    ]
```

- [ ] **Step 4: Run all retriever tests to verify they pass**

```bash
uv run pytest tests/rag/test_retriever.py -v
```

Expected: all tests PASS (both original and new graph_retrieve tests).

- [ ] **Step 5: Commit**

```bash
git add src/medic_agent/rag/retriever.py tests/rag/test_retriever.py
git commit -m "feat(kg): add graph_retrieve — entity lookup path for hybrid retrieval"
```

---

## Task 6: Hybrid retrieve_node in coding_agent and ambient_agent

**Files:**
- Modify: `src/medic_agent/agents/coding_agent.py`
- Modify: `src/medic_agent/agents/ambient_agent.py`
- Modify: `tests/agents/test_coding_agent.py`
- Modify: `tests/agents/test_ambient_agent.py`

- [ ] **Step 1: Update test_coding_agent.py to mock graph_retrieve**

In `tests/agents/test_coding_agent.py`, add `graph_retrieve` to the mock list:

```python
import medic_agent.agents.coding_agent as coding_module
from medic_agent.agents.coding_agent import build_coding_agent


def test_coding_agent_runs_all_steps_and_produces_response(mocker):
    mocker.patch.object(
        coding_module,
        "llm_step",
        side_effect=[
            ("Diagnoses:\n- DM2", {"name": "extract", "type": "generation"}),
            ("DRAFT CODES E11.9", {"name": "code", "type": "generation"}),
            ("FINAL E11.9 99213", {"name": "verify", "type": "generation"}),
        ],
    )
    mocker.patch.object(
        coding_module,
        "retrieve",
        return_value=[{"text": "ctx", "source_filename": "f", "chunk_index": 0}],
    )
    mocker.patch.object(coding_module, "graph_retrieve", return_value=[])

    graph = build_coding_agent()
    out = graph.invoke(
        {"query": "code this", "model_id": "claude-haiku-4-5-20251001", "scratch": {}}
    )
    assert out["response"] == "FINAL E11.9 99213"
    assert len(out["chunks"]) == 1
    step_names = [s["name"] for s in out["agent_steps"]]
    assert step_names == ["extract", "retrieval", "code", "verify"]
```

- [ ] **Step 2: Update test_ambient_agent.py to mock graph_retrieve and extract_entities**

In `tests/agents/test_ambient_agent.py`:

```python
import medic_agent.agents.ambient_agent as ambient_module
from medic_agent.agents.ambient_agent import build_ambient_agent


def test_ambient_agent_runs_all_steps_and_produces_response(mocker):
    mocker.patch.object(
        ambient_module,
        "llm_step",
        side_effect=[
            ("S O A P NOTE", {"name": "soap", "type": "generation"}),
            ("BILLING CODES E11.9", {"name": "code", "type": "generation"}),
            ("FINAL NOTE + CODES + FLAGS", {"name": "verify", "type": "generation"}),
        ],
    )
    mocker.patch.object(ambient_module, "retrieve", return_value=[])
    mocker.patch.object(ambient_module, "extract_entities", return_value=[])
    mocker.patch.object(ambient_module, "graph_retrieve", return_value=[])

    graph = build_ambient_agent()
    out = graph.invoke(
        {
            "query": "Doctor: ...\nPatient: ...",
            "model_id": "claude-haiku-4-5-20251001",
            "scratch": {},
        }
    )
    assert out["response"] == "FINAL NOTE + CODES + FLAGS"
    step_names = [s["name"] for s in out["agent_steps"]]
    assert step_names == ["retrieval", "soap", "code", "verify"]
```

- [ ] **Step 3: Run tests to confirm they fail before implementation**

```bash
uv run pytest tests/agents/test_coding_agent.py tests/agents/test_ambient_agent.py -v
```

Expected: tests fail because `graph_retrieve` and `extract_entities` don't exist in those modules yet.

- [ ] **Step 4: Update coding_agent.py with hybrid retrieve_node**

In `src/medic_agent/agents/coding_agent.py`, add imports:

```python
import re
import time

from langgraph.graph import END, START, StateGraph

from medic_agent.agents.state import AgentState, llm_step
from medic_agent.config.prompts import get_prompt
from medic_agent.rag.retriever import graph_retrieve, retrieve
```

Add the helper and update `retrieve_node`:

```python
def _parse_entity_texts(entities_text: str) -> list[str]:
    return [
        m.group(1).strip()
        for m in re.finditer(r"^[-]\s+(.+)$", entities_text, re.MULTILINE)
    ]


def retrieve_node(state: AgentState) -> dict:
    entities_text = state.get("scratch", {}).get("entities", "")
    start = time.monotonic()
    vector_chunks = retrieve(f"{state['query']}\n{entities_text}", k=5)
    entity_texts = _parse_entity_texts(entities_text)
    graph_chunks = graph_retrieve(entity_texts)
    chunks = vector_chunks + graph_chunks
    step = {
        "name": "retrieval",
        "type": "retriever",
        "model_id": "",
        "token_usage": {},
        "latency_ms": round((time.monotonic() - start) * 1000, 1),
        "input_summary": state["query"][:200],
        "output_summary": f"{len(vector_chunks)} vector + {len(graph_chunks)} graph chunks",
    }
    return {"chunks": chunks, "agent_steps": [step]}
```

- [ ] **Step 5: Update ambient_agent.py with hybrid retrieve_node**

In `src/medic_agent/agents/ambient_agent.py`, update imports:

```python
import time

from langgraph.graph import END, START, StateGraph

from medic_agent.agents.state import AgentState, llm_step
from medic_agent.config.prompts import get_prompt
from medic_agent.rag.entity_extractor import extract_entities
from medic_agent.rag.retriever import graph_retrieve, retrieve
```

Update `retrieve_node`:

```python
def retrieve_node(state: AgentState) -> dict:
    start = time.monotonic()
    vector_chunks = retrieve(state["query"], k=5)
    entities = extract_entities(state["query"][:2000])
    entity_texts = [e["text"] for e in entities]
    graph_chunks = graph_retrieve(entity_texts)
    chunks = vector_chunks + graph_chunks
    step = {
        "name": "retrieval",
        "type": "retriever",
        "model_id": "",
        "token_usage": {},
        "latency_ms": round((time.monotonic() - start) * 1000, 1),
        "input_summary": state["query"][:200],
        "output_summary": f"{len(vector_chunks)} vector + {len(graph_chunks)} graph chunks",
    }
    return {"chunks": chunks, "agent_steps": [step]}
```

- [ ] **Step 6: Run all agent tests to verify they pass**

```bash
uv run pytest tests/agents/ -v
```

Expected: all tests PASS.

- [ ] **Step 7: Run the full test suite to check for regressions**

```bash
uv run pytest -v
```

Expected: all tests PASS. No regressions in rag, agents, observability, or eval tests.

- [ ] **Step 8: Commit**

```bash
git add src/medic_agent/agents/coding_agent.py src/medic_agent/agents/ambient_agent.py \
        tests/agents/test_coding_agent.py tests/agents/test_ambient_agent.py
git commit -m "feat(kg): hybrid retrieve_node — merge vector + graph context in both agents"
```

---

## Post-implementation: verify end-to-end and review the graph

After all tasks complete:

**Run the app:**
```bash
uv run streamlit run src/medic_agent/ui/app.py
```
Upload a document in Tab 2 → it should ingest into both ChromaDB and Kuzu. The entity extraction Haiku call will run per chunk. Submit a query → retrieval step output summary in the session log should show `N vector + M graph chunks`.

**Inspect the graph with Kuzu Explorer:**
```bash
uvx kuzu-explorer data/kuzu/
```
Open `http://localhost:8000` in a browser. Run a Cypher query:
```cypher
MATCH (e:Entity)-[:APPEARS_IN]->(d:Document) RETURN e, d LIMIT 50
```
This renders the entity→document graph visually with clickable nodes.

**Quick terminal inspection:**
```bash
uv run python -c "
from medic_agent.rag.graph_store import _get_conn
conn = _get_conn()
r = conn.execute('MATCH (e:Entity)-[:APPEARS_IN]->(d:Document) RETURN e.text, e.entity_type, d.filename LIMIT 20')
while r.has_next():
    print(r.get_next())
"
```
