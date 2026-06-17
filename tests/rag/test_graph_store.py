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
