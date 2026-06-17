# Development Plan — medic-agent

**Version:** 0.9  
**Last Updated:** 2026-06-16  

---

## Guiding Principle

Build the thinnest possible working vertical slice first.
Each phase must produce something that runs and can be tested end-to-end.
Never build infrastructure before you have something working that needs it.

---

## Phase Overview

```
Phase 0 (V0): Core Loop              ← COMPLETE ✅
Phase 1 (V1): Context Layer          ← COMPLETE ✅
Phase 2 (V2): Multi-Agent Orchestr.  ← COMPLETE ✅
Phase 3 (V3): Knowledge Graph Layer  ← YOU ARE HERE (planning approved 2026-06-16)
Phase 4 (V4): Production-Ready       ← Future
```

> **Phase 2 redefinition (2026-06-16):** V2 was originally scoped as
> "Multi-Persona" (clinician / admin / patient). It was redefined as
> **Multi-Agent Orchestration**: each use case became its own multi-step
> agent with a hybrid orchestrator routing each query automatically.
> V2 is complete as of 2026-06-16.

---

## Phase 0 — Core Loop (V0)

**Goal:** User types a query → LLM answers → response displayed.
**Success:** You can ask a healthcare question and get a useful Claude response in a browser.

### Step 0.1 — Environment Setup
- [x] Install `uv` (modern Python package manager)
- [x] Create virtual environment with Python 3.11
- [x] Install dependencies: `streamlit`, `litellm`, `python-dotenv`
- [x] Create `src/medic_agent/__init__.py` and all subfolder `__init__.py` files
- [x] Create `.env` with `ANTHROPIC_API_KEY`
- [x] Create `.env.example` (no real keys)
- [x] Create `.gitignore`
- [x] Initialize git repo
- [x] Verify: `python -c "import streamlit; import litellm"` runs without error

### Step 0.2 — LLM Client
- [x] Create `src/medic_agent/llm/client.py`
- [x] Implement `ask(model_id, system_prompt, user_query) -> str`
- [x] Test with a hardcoded query in `__main__` block
- [x] Verify: `python src/medic_agent/llm/client.py` returns a response from Claude

### Step 0.3 — Config
- [x] Create `src/medic_agent/config/settings.py`
- [x] Load `ANTHROPIC_API_KEY` from `.env`
- [x] Define `AVAILABLE_MODELS` dict (display name → LiteLLM model ID)
- [x] Define `DEFAULT_SYSTEM_PROMPT`
- [x] Validate API key on import (raise clear error if missing)

### Step 0.4 — Streamlit UI
- [x] Create `src/medic_agent/ui/app.py`
- [x] Add model selector dropdown (from `config.settings.AVAILABLE_MODELS`)
- [x] Add query text area
- [x] Add "Ask" button
- [x] Wire button to `llm.client.ask()`
- [x] Display response below input
- [x] Add loading spinner during API call
- [x] Add basic error display
- [x] Verify: `streamlit run src/medic_agent/ui/app.py` opens in browser and works end-to-end

### Step 0.5 — Polish and Baseline Test
- [x] 10 unit tests passing (happy path, validation, all error mappings, cache_control)
- [x] API key verified absent from all source and test files
- [x] Prompt caching enabled on system prompt (V1 document context hook in place)
- [x] All code committed to git with descriptive messages

**V0 Complete ✅ — 2026-05-09**

---

## Phase 1 — Context Layer (V1)

**Goal:** User uploads documents once; agent answers questions grounded in those documents forever after.
**Success:** Upload a clinical PDF, ask a question about it, get a cited answer.

### Step 1.1 — Dependencies and Storage Setup
- [x] `uv add chromadb pypdf huggingface_hub langfuse ragas`
- [x] Add `HUGGINGFACE_API_KEY` and `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` to `.env` and `.env.example`
- [x] Update `config/settings.py`:
  - Validate `HUGGINGFACE_API_KEY` at startup
  - Add `CHROMA_PERSIST_DIR = "data/chroma"`
  - Add `EMBEDDING_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"`
  - Add `CODING_SYSTEM_PROMPT` and `AMBIENT_SYSTEM_PROMPT` constants
  - Add `USE_CASES` dict mapping display names to system prompts
- [x] Create `data/chroma/` directory
- [x] Add `data/` to `.gitignore`
- [x] Create `src/medic_agent/rag/__init__.py` and `tests/rag/__init__.py`
- [x] Verify: `uv run python -c "import chromadb; import pypdf; import huggingface_hub; print('OK')"`

### Step 1.2 — Document Ingestor (`rag/ingestor.py`)
- [x] `load_pdf(file_bytes: bytes, filename: str) -> list[dict]`
- [x] `load_text(text: str, filename: str) -> list[dict]`
- [x] Each chunk dict: `{text, source_filename, chunk_index}`
- [x] Chunking: 1000-char window, 200-char overlap
- [x] Verify: parse a sample PDF and print chunk count

### Step 1.3 — Embedder (`rag/embedder.py`)
- [x] `embed_texts(texts: list[str]) -> list[list[float]]` — batch, for ingestion
- [x] `embed_query(query: str) -> list[float]` — single, for retrieval
- [x] Model: `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` via `huggingface_hub.InferenceClient`
- [x] Apply mean pooling across token dimension to get sentence-level vector (768 dims)
- [x] Verify: embed "myocardial infarction" and "I21.9" — confirm vectors are close in cosine similarity

### Step 1.4 — Vector Store (`rag/store.py`)
- [x] `add_document(doc_id, chunks, embeddings) -> None`
- [x] `document_exists(doc_id) -> bool`
- [x] `list_documents() -> list[str]`
- [x] `delete_document(doc_id) -> None`
- [x] ChromaDB `PersistentClient` at `CHROMA_PERSIST_DIR`
- [x] Verify: add a document, restart Python, confirm it persists

### Step 1.5 — Retriever (`rag/retriever.py`)
- [x] `retrieve(query: str, k: int = 5) -> list[dict]`
- [x] Returns list of `{text, source_filename, chunk_index}` dicts
- [x] Embeds query, queries ChromaDB by cosine similarity
- [x] Verify: retrieve chunks from an uploaded document with a relevant query

### Step 1.6 — Wire Context into LLM Client (`llm/client.py`)
- [x] Update `ask()` signature: add `context: list[dict] | None = None`
- [x] Update `_build_messages()`: inject context as second `cache_control: ephemeral` block
- [x] Context block format: `[Source: filename | Chunk N]\n<text>\n\n...`
- [x] System prompt instructs LLM to cite sources and prefer document context
- [x] Verify: call `ask()` with context, confirm response references the document

### Step 1.7 — Observability Hooks (`observability/tracer.py`)
- [x] Create `src/medic_agent/observability/__init__.py`
- [x] Create `src/medic_agent/observability/tracer.py`:
  - `Session` dataclass: `session_id, timestamp, use_case, model_id, query, chunks_retrieved, system_prompt_version, response, latency_ms, token_usage, error`
  - `log_session(session: Session) -> None` — writes JSON to `data/sessions/` AND sends to LangFuse
  - LangFuse trace wraps the full pipeline: retrieval span + LLM span
- [x] Create `data/sessions/` directory; add to `.gitignore`
- [x] Wire `log_session()` call into `llm/client.py` after each `ask()` completes
- [x] Verify: submit a query → check LangFuse dashboard → see trace with retrieval + LLM spans

### Step 1.8 — Update Streamlit UI (`ui/app.py`)

Restructure into four tabs:
`st.tabs(["🤖 Agent", "📚 Knowledge Base & Prompts", "📊 Observability", "🧪 Evaluation"])`

**Tab 1 — Agent:**
- [x] Sidebar: use case selector only (Medical Coding / Ambient Note Taking)
- [x] Sidebar: KB status summary — "📄 N documents (M chunks) → Manage in Tab 2"
- [x] Use case selector drives system prompt (loaded from `data/prompts.json`), input label, default query
- [x] On query: retrieve top-5 → inject context → call `ask()` → display response + citations
- [x] Verify: full coding flow and full ambient flow work end-to-end

**Tab 2 — Knowledge Base & System Prompts:**
- [x] *Section: Knowledge Base*
  - [x] File uploader (PDF, TXT); on upload: ingest → embed → store; show success/duplicate/error
  - [x] Document table: filename, chunk count, upload date, delete button
  - [x] Summary footer: total docs and total chunks
- [x] *Section: System Prompts*
  - [x] Load current prompts from `data/prompts.json` (fallback to `settings.py` defaults)
  - [x] `st.text_area("Medical Coding Prompt", ...)` — full editable text, tall height
  - [x] `st.text_area("Ambient Note Taking Prompt", ...)` — full editable text, tall height
  - [x] "💾 Save Prompts" button → write `data/prompts.json` + push new version to LangFuse + show active version number
  - [x] "↩ Reset to Defaults" button → reload defaults into text areas without saving
- [x] Verify: edit coding prompt → save → run a query in Tab 1 → LangFuse trace shows new prompt version

**Tab 3 — Observability:**
- [x] Read `data/sessions/` JSONL files into a DataFrame
- [x] Render summary stats bar: total sessions, avg latency, avg cost, error count
- [x] Render session log table with filters (use case, date range)
- [x] Expandable row: full query, response, retrieved chunks, prompt version, token breakdown
- [x] "Export CSV" button
- [x] "Open in LangFuse" button → `st.link_button()` to LangFuse project URL (from config)
- [x] Verify: run 3 queries in Tab 1 → Tab 3 shows all 3 with correct metadata

**Tab 4 — Evaluation:**
- [x] Render golden cases table with checkboxes (select which cases to run)
- [x] Layer selector checkboxes: Layer 1 (free/instant), Layer 2 (RAGAS), Layer 3 (LLM-as-judge)
- [x] Show estimated time and cost warning before running Layer 2/3
- [x] "Run Evaluation" button → calls `EvalRunner.run_all()` with spinner per layer
- [x] Results table: case ID, L1 pass/fail, RAGAS faithfulness, judge overall score
- [x] Delta column vs baseline (green ✅ / amber ⚠️ / red 🔴)
- [x] "Set as Baseline" button → writes scores to `tests/eval/baseline.json`
- [ ] Score history chart (scores over time, keyed by timestamp)
- [ ] "Export Results" button + "Open in LangFuse" eval traces link
- [x] Verify: run Layer 1 only → all 5 cases show results instantly

### Step 1.9 — Unit Tests
- [x] `tests/rag/test_ingestor.py`: chunk count, overlap, metadata fields
- [x] `tests/rag/test_embedder.py`: mocked HuggingFace calls, correct dimensions
- [x] `tests/rag/test_store.py`: in-memory ChromaDB (not persistent) for test isolation
- [x] `tests/rag/test_retriever.py`: mocked embedder + in-memory store
- [x] `tests/observability/test_tracer.py`: mocked LangFuse, verify session fields captured
- [x] All tests pass with `uv run pytest`

### Step 1.10 — Evaluation Runner + Tests
- [x] Create `src/medic_agent/evaluation/__init__.py`
- [x] Create `src/medic_agent/evaluation/runner.py`:
  - `EvalResult` dataclass: `{case_id, layer1_pass, ragas_scores, judge_scores, timestamp}`
  - `run_layer1(case) -> dict` — deterministic checks (ICD-10 regex, CPT format, SOAP sections)
  - `run_layer2(case, retrieved_chunks) -> dict` — RAGAS faithfulness + context precision
  - `run_layer3(case, response) -> dict` — Claude Sonnet judges on rubric, returns JSON scores
  - `run_all(cases, layers) -> list[EvalResult]` — called by both UI and pytest
  - Scores written to LangFuse as evaluation traces
- [x] Create `tests/eval/test_eval.py`:
  - Loads `golden_cases.json`
  - Calls `EvalRunner.run_all()` 
  - Asserts Layer 1 passes for all cases
  - Asserts no judge score regresses >0.5 vs `baseline.json` (if baseline exists)
- [x] Run `uv run pytest tests/eval/ -v -m eval` — separate pytest mark so it doesn't run with unit tests
- [x] After first passing run: click "Set as Baseline" in Tab 3 → `baseline.json` created

**V1 Complete ✅ — 2026-05-10**

---

## Phase 2 — Multi-Agent Orchestration (V2)

> Do not begin V2 until V1 is complete and tested. (V1 complete ✅ 2026-05-10)

**Goal:** Turn the two use cases into two specialized **multi-step agents** and
put a **hybrid orchestrator** in front of them that reads the raw user query and
routes to the correct agent automatically. The UI no longer asks the user to pick
the use case; instead it displays which agent the system selected, why, and how
confidently. Every response is scored online by an LLM-as-judge, and all of it —
routing, each agent step, and judge scores — is captured as nested LangFuse spans.

**Success:** Paste a coding encounter → system routes to the Coding agent, runs
extract→retrieve→code→verify, shows the determined agent + reasoning + judge
scores. Paste a conversation transcript → system routes to the Ambient agent,
runs retrieve→SOAP→code→verify. A manual override forces a specific agent.

### Design Decisions (locked 2026-06-16)

| Decision | Choice | Reason |
|---|---|---|
| Orchestration framework | LangGraph | User-approved; reverses the earlier "no LangChain ecosystem" rule for the orchestration layer only |
| Router | Hybrid: heuristics first, LLM (Haiku) fallback | Cheap and instant in the common case; robust when ambiguous |
| User control | Auto routing + manual override (Auto / Coding / Ambient) | Escape hatch when the router is wrong |
| Agent shape | Full multi-step subgraphs (extract/retrieve/code/verify etc.) | Genuinely agentic; "as needed" — lean, not maximal |
| Online eval | LLM-as-judge (Sonnet) scores every response, default on (toggleable) | User wants judge metrics visible in observability traces |
| Tracing | Manual nested spans, populated then emitted by tracer.py | Matches existing pattern; captures custom router + judge data |
| Prompt editing | ALL prompts editable in Tab 2, grouped by agent (router / coding / ambient / judge) | Full transparency and tuning; settings.py holds defaults, data/prompts.json persists |

### Architecture (summary — see ARCHITECTURE.md §14 for full detail)

```
START → router → (conditional edge) → coding_agent ─┐
                                    → ambient_agent ─┴→ judge → END
```

New package `src/medic_agent/agents/`:
- `state.py` — `AgentState` (LangGraph TypedDict) shared across nodes
- `router.py` — hybrid router → `RouteDecision{use_case, method, confidence, reasoning}`
- `coding_agent.py` — nodes: extract → retrieve → code → verify
- `ambient_agent.py` — nodes: retrieve → soap → code → verify
- `judge.py` — online LLM-as-judge (shares rubric with `evaluation/runner.py`)
- `orchestrator.py` — builds the graph; single entry point `run(query, model_id, override)`

### Step 2.1 — Dependencies & Scaffolding
- [x] `uv add langgraph` (pulls in `langchain-core`)
- [x] Create `src/medic_agent/agents/__init__.py` and `tests/agents/__init__.py`
- [x] Verify: `uv run python -c "import langgraph; print('OK')"`

### Step 2.2 — Agent State & Non-Logging LLM Primitive
- [x] `agents/state.py`: `AgentState` TypedDict — `query, model_id, override, route, use_case, chunks, scratch (per-step intermediates), response, judge_scores, agent_steps`
- [x] `llm/client.py`: add `complete(model_id, system_prompt, user_query, context=None) -> tuple[str, dict]` — returns `(text, token_usage)`, **does not** log a Session (agent nodes call this; the orchestrator logs one Session per query). `ask()` stays unchanged for V0 tests.
- [x] Verify: `complete()` returns text + usage without writing to `data/sessions/`

### Step 2.3 — Hybrid Router (`agents/router.py`)
- [x] `RouteDecision` dataclass: `{use_case, method, confidence, reasoning}`
- [x] `heuristic_route(query) -> RouteDecision | None` — dialogue markers / coding keywords; `None` when ambiguous
- [x] `llm_route(query, model_id) -> RouteDecision` — Haiku classifier returning use_case + confidence + reasoning
- [x] `route(query, model_id, override) -> RouteDecision` — override → `method="manual"`; else heuristic, then LLM fallback
- [x] Add `ROUTER_SYSTEM_PROMPT` to `settings.py`
- [x] Tests: heuristic hits both ways, ambiguous → None, mocked LLM fallback, override path

### Step 2.4 — Coding Agent Subgraph (`agents/coding_agent.py`)
- [x] Nodes: `extract` (diagnoses/procedures) → `retrieve` (entity-aware top-5) → `code` (ICD-10/CPT with citations + gaps; uses editable coding prompt) → `verify` (self-check every code is doc-supported; finalize)
- [x] Add `CODING_EXTRACT_PROMPT`, `CODING_VERIFY_PROMPT` to `settings.py`
- [x] Each node appends a step record to `state["agent_steps"]`
- [x] Tests: each node with mocked `complete()` + mocked `retrieve()`

### Step 2.5 — Ambient Agent Subgraph (`agents/ambient_agent.py`)
- [x] Nodes: `retrieve` (coding references) → `soap` (draft SOAP from transcript; uses editable ambient prompt) → `code` (billing codes from Assessment) → `verify` (faithfulness/section self-check + documentation flags; finalize)
- [x] Add `AMBIENT_CODE_PROMPT`, `AMBIENT_VERIFY_PROMPT` to `settings.py`
- [x] Tests: each node with mocked `complete()` + mocked `retrieve()`

### Step 2.6 — Online Judge (`agents/judge.py`) + Shared Rubric
- [x] Extract Layer-3 judge logic from `evaluation/runner.py` into a shared function (rubric per use case, JSON parsing)
- [x] `judge_output(use_case, response, model_id) -> dict` — per-dimension + overall + reasoning
- [x] `evaluation/runner.py` Layer 3 now calls the shared function (no duplication)
- [x] Tests: mocked judge call, JSON parse, both rubrics

### Step 2.7 — Orchestrator Graph (`agents/orchestrator.py`)
- [x] Build `StateGraph`: `router → agent → judge → END`
- [x] `run(query, model_id, override="Auto", judge_on=True) -> dict` — returns `{route, response, chunks, judge_scores, agent_steps}`
- [x] Populate the enriched `Session` and call `log_session()` once
- [x] Tests: routing dispatches to correct agent; manual override respected; judge toggle off skips judge

### Step 2.8 — Enhanced Multi-Agent Tracing (`observability/tracer.py`)
- [x] `Session` new fields: `route_decision: dict`, `agent_steps: list[dict]`, `judge_scores: dict`
- [x] Emit nested spans: `router` span, `agent:{use_case}` span containing one span per agent step (generation/retriever), `judge` span; attach judge `overall` as a LangFuse score
- [x] Local JSONL captures route_decision + judge_scores for Tab 3
- [x] Tests: mocked LangFuse, assert nested structure + judge score attached

### Step 2.9 — UI Updates (`ui/app.py`)
- [x] Sidebar: replace use-case radio with **routing mode** selectbox (`Auto / Medical Coding / Ambient Note Taking`) + judge on/off toggle
- [x] Tab 1: single generic input area; on Submit call `orchestrator.run(...)`
- [x] **Routing panel** after Submit: *Determined agent · method (heuristic/llm/manual) · confidence · one-line reasoning*
- [x] Show response + sources + judge scores inline
- [x] **Tab 2: expand to all prompts**, grouped by agent (Router; Coding: extract/code/verify; Ambient: soap/code/verify; Judge: coding/ambient)
- [x] Tab 3 (Observability): add route decision + judge overall to session rows and summary
- [x] Verify: coding flow, ambient flow, a forced override, and editing a step prompt then re-running all work end-to-end

### Step 2.10 — Evaluation Integration
- [x] `EvalRunner` runs cases through `orchestrator.run()` (full pipeline under eval)
- [x] Layer 1: add **router-accuracy** check (golden cases carry known `use_case`)
- [x] Verify: `uv run pytest tests/eval/ -v -m eval`

### Step 2.11 — Tests & Acceptance
- [x] All unit tests pass with `uv run pytest` (external calls mocked)
- [x] V2 acceptance criteria in `CLAUDE.md` all checked
- [x] All steps committed with descriptive messages

**V2 Complete ✅ — 2026-06-16**

---

## Phase 3 — Knowledge Graph Layer (V3)

> Do not begin V3 until V2 is complete and tested. (V2 complete ✅ 2026-06-16)

**Goal:** Add a Kuzu embedded knowledge graph alongside ChromaDB so that entities
and their cross-document relationships are indexed at ingest time and used as a
second retrieval path during agent runs.

**Success:** Upload a document → entities are extracted and stored in Kuzu. Submit
a coding or ambient query → the retrieve step merges vector chunks with entity
context from the graph. The session log shows `N vector + M graph chunks`. Kuzu
Explorer (`uvx kuzu-explorer data/kuzu/`) renders the entity graph visually.

### Design Decisions (locked 2026-06-16)

| Decision | Choice | Reason |
|---|---|---|
| Graph DB | Kuzu (embedded) | In-process like ChromaDB; openCypher; no server; MIT license |
| Store strategy | Dual-store (Kuzu + ChromaDB) | Each answers different questions; neither replaces the other |
| Entity extraction timing | At ingest, Haiku per chunk | Front-load cost at upload; query path pays nothing extra |
| Entity ID | MD5(text.lower()) | Same entity across docs shares one node → cross-document linking is automatic |
| Graph schema V3 | Entity-to-Document only (no RELATED_TO edges) | Minimal first pass; relation extraction is a V4 extension |

### Architecture (summary — see ARCHITECTURE.md §17 for full detail)

```
INGEST:  chunk → ChromaDB (vector)
                → entity_extractor (Haiku) → graph_store (Kuzu)

RETRIEVE: query → Path A: ChromaDB vector similarity (existing)
                → Path B: entity_texts → Kuzu traversal (new)
                → merge A + B → LLM context
```

New files: `rag/graph_store.py`, `rag/entity_extractor.py`  
Modified: `rag/store.py`, `rag/retriever.py`, `agents/coding_agent.py`, `agents/ambient_agent.py`

### Step 3.1 — Kuzu Dependency + Config
- [ ] `uv add kuzu`
- [ ] Add `KUZU_PERSIST_DIR = str(DATA_DIR / "kuzu")` to `config/settings.py`
- [ ] Add `ENTITY_EXTRACTOR_MODEL_ID = AVAILABLE_MODELS["Claude Haiku (Fast)"]` to `config/settings.py`
- [ ] Add `data/kuzu/` to `.gitignore`
- [ ] Verify: `uv run python -c "import kuzu; print(kuzu.__version__)"`

### Step 3.2 — Graph Store (`rag/graph_store.py`)
- [ ] Module-level `_db` / `_conn` singletons; lazy init on first call via `_get_conn()`
- [ ] `_init_schema(conn)` — `CREATE NODE TABLE IF NOT EXISTS` for Document + Entity; `CREATE REL TABLE IF NOT EXISTS` for APPEARS_IN
- [ ] `upsert_document(doc_id, filename)` — check-then-insert Document node
- [ ] `upsert_entities(doc_id, entities, chunk_id, chunk_index)` — check-then-insert Entity nodes; create APPEARS_IN edges
- [ ] `get_related_entities(entity_texts)` — Cypher MATCH returning `{text, entity_type, filename}`
- [ ] `delete_document_entities(doc_id)` — delete APPEARS_IN edges, prune orphan Entity nodes, delete Document node
- [ ] Unit tests in `tests/rag/test_graph_store.py` using real Kuzu in `tmp_path` fixture (no mocking of Kuzu itself)
- [ ] Tests cover: upsert + query, empty input returns empty, delete removes entity, same entity links to two docs, delete one preserves the other

### Step 3.3 — Entity Extractor (`rag/entity_extractor.py`)
- [ ] `extract_entities(chunk_text: str) -> list[dict]` — Haiku LLM call, JSON parse, returns `[{text, entity_type}]`
- [ ] Strip markdown fences before JSON parse; return `[]` on any parse failure
- [ ] Truncate input to 2000 chars before sending (avoids token spikes on large chunks)
- [ ] `ENTITY_EXTRACTOR_MODEL_ID` used for all calls (not caller-supplied; consistent cost control)
- [ ] Unit tests in `tests/rag/test_entity_extractor.py` with mocked `complete()` — valid JSON, markdown fences, bad JSON, non-string fields filtered, truncation applied

### Step 3.4 — Wire Graph into `rag/store.py`
- [ ] `add_document()`: after ChromaDB write, call `graph_store.upsert_document()` then for each chunk: `entity_extractor.extract_entities()` → `graph_store.upsert_entities()`
- [ ] `delete_document()`: after ChromaDB delete, call `graph_store.delete_document_entities()`
- [ ] Update `tests/rag/test_store.py` autouse fixture: add `mocker.patch` for `graph_store.upsert_document`, `graph_store.upsert_entities`, `graph_store.delete_document_entities`, `entity_extractor.extract_entities` (returns `[]`)
- [ ] Verify: all existing store tests still pass

### Step 3.5 — Graph Retrieval Path (`rag/retriever.py`)
- [ ] `graph_retrieve(entity_texts: list[str]) -> list[dict]` — calls `graph_store.get_related_entities()`, formats results as one synthetic chunk with `source_filename="[knowledge-graph]"` and `chunk_index=-1`
- [ ] Returns `[]` immediately for empty entity list (no graph call)
- [ ] Unit tests in `tests/rag/test_retriever.py`: formatted output structure, grouping multiple docs per entity, empty-list short-circuit, no-match returns empty

### Step 3.6 — Hybrid `retrieve_node` in Both Agents
- [ ] `agents/coding_agent.py`: `retrieve_node` parses entity bullets from `scratch["entities"]` with regex → calls both `retrieve()` and `graph_retrieve()` → concatenates; `output_summary` shows `N vector + M graph chunks`
- [ ] `agents/ambient_agent.py`: `retrieve_node` calls `extract_entities(state["query"][:2000])` inline → calls both `retrieve()` and `graph_retrieve()` → concatenates
- [ ] Update `tests/agents/test_coding_agent.py`: add `mocker.patch.object(coding_module, "graph_retrieve", return_value=[])`
- [ ] Update `tests/agents/test_ambient_agent.py`: add mocks for `graph_retrieve` and `extract_entities`
- [ ] All agent tests pass

### Step 3.7 — End-to-End Verification
- [ ] Run full test suite: `uv run pytest -v` — all pass
- [ ] Run app: `uv run streamlit run src/medic_agent/ui/app.py`
- [ ] Upload a document in Tab 2 — verify no errors; session log retrieve step shows `N vector + M graph chunks`
- [ ] Inspect graph: `uvx kuzu-explorer data/kuzu/` → run `MATCH (e:Entity)-[:APPEARS_IN]->(d:Document) RETURN e, d LIMIT 50`
- [ ] V3 acceptance criteria in `CLAUDE.md` all checked
- [ ] Commit each step with descriptive messages

---

## Phase 4 — Production-Ready (V4)

> Do not begin V4 until V3 is complete and tested.

**Goal:** App is deployable, has auth, is shareable.

### Planned Steps (to be detailed when V4 begins)
- [ ] Add user authentication
- [ ] Containerize with Docker
- [ ] Deploy to cloud (provider TBD)
- [ ] HIPAA compliance review if real patient data is involved

---

## Dependency Map — V0

```
0.1 Environment Setup
    ↓
0.2 LLM Client  ←─── depends on 0.1
    ↓
0.3 Config      ←─── depends on 0.1
    ↓
0.4 Streamlit UI ←── depends on 0.2 + 0.3
    ↓
0.5 Polish      ←─── depends on 0.4
```

---

## Revision History

| Version | Date | Change |
|---|---|---|
| 0.1 | 2026-05-09 | Initial scaffold |
| 0.2 | 2026-05-09 | Updated file paths to src/medic_agent/ structure |
| 0.3 | 2026-05-09 | V1 Phase 1 fleshed out with concrete steps |
| 0.4 | 2026-05-09 | Specialized to coding + ambient; SapBERT replaces OpenAI embeddings; use case selector added to Step 1.7 |
| 0.5 | 2026-05-10 | Three-tab UI in Step 1.8; EvalRunner module in Step 1.10; user workflows documented |
| 0.6 | 2026-05-10 | Four-tab UI; KB & Prompts tab in Step 1.8; prompt persistence design |
| 0.7 | 2026-05-10 | V1 marked complete; all Step 1.1–1.10 items checked off |
| 0.8 | 2026-06-16 | Phase 2 redefined: Multi-Persona → Multi-Agent Orchestration; LangGraph, hybrid router, multi-step agents, online judge; Steps 2.1–2.11 detailed |
| 0.9 | 2026-06-16 | Phase 2 marked complete; Phase 3 = Knowledge Graph (Kuzu + entity extractor + hybrid retrieval, Steps 3.1–3.7); old Phase 3 → Phase 4 Production-Ready |
