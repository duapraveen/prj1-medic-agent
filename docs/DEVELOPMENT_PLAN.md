# Development Plan — medic-agent

**Version:** 0.1  
**Last Updated:** 2026-05-09  

---

## Guiding Principle

Build the thinnest possible working vertical slice first.
Each phase must produce something that runs and can be tested end-to-end.
Never build infrastructure before you have something working that needs it.

---

## Phase Overview

```
Phase 0 (V0): Core Loop        ← COMPLETE ✅
Phase 1 (V1): Context Layer    ← YOU ARE HERE
Phase 2 (V2): Multi-Persona    ← Future
Phase 3 (V3): Production-Ready ← Future
```

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
- [ ] `uv add chromadb pypdf huggingface_hub langfuse ragas`
- [ ] Add `HUGGINGFACE_API_KEY` and `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` to `.env` and `.env.example`
- [ ] Update `config/settings.py`:
  - Validate `HUGGINGFACE_API_KEY` at startup
  - Add `CHROMA_PERSIST_DIR = "data/chroma"`
  - Add `EMBEDDING_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"`
  - Add `CODING_SYSTEM_PROMPT` and `AMBIENT_SYSTEM_PROMPT` constants
  - Add `USE_CASES` dict mapping display names to system prompts
- [ ] Create `data/chroma/` directory
- [ ] Add `data/` to `.gitignore`
- [ ] Create `src/medic_agent/rag/__init__.py` and `tests/rag/__init__.py`
- [ ] Verify: `uv run python -c "import chromadb; import pypdf; import huggingface_hub; print('OK')"`

### Step 1.2 — Document Ingestor (`rag/ingestor.py`)
- [ ] `load_pdf(file_bytes: bytes, filename: str) -> list[dict]`
- [ ] `load_text(text: str, filename: str) -> list[dict]`
- [ ] Each chunk dict: `{text, source_filename, chunk_index}`
- [ ] Chunking: 1000-char window, 200-char overlap
- [ ] Verify: parse a sample PDF and print chunk count

### Step 1.3 — Embedder (`rag/embedder.py`)
- [ ] `embed_texts(texts: list[str]) -> list[list[float]]` — batch, for ingestion
- [ ] `embed_query(query: str) -> list[float]` — single, for retrieval
- [ ] Model: `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` via `huggingface_hub.InferenceClient`
- [ ] Apply mean pooling across token dimension to get sentence-level vector (768 dims)
- [ ] Verify: embed "myocardial infarction" and "I21.9" — confirm vectors are close in cosine similarity

### Step 1.4 — Vector Store (`rag/store.py`)
- [ ] `add_document(doc_id, chunks, embeddings) -> None`
- [ ] `document_exists(doc_id) -> bool`
- [ ] `list_documents() -> list[str]`
- [ ] `delete_document(doc_id) -> None`
- [ ] ChromaDB `PersistentClient` at `CHROMA_PERSIST_DIR`
- [ ] Verify: add a document, restart Python, confirm it persists

### Step 1.5 — Retriever (`rag/retriever.py`)
- [ ] `retrieve(query: str, k: int = 5) -> list[dict]`
- [ ] Returns list of `{text, source_filename, chunk_index}` dicts
- [ ] Embeds query, queries ChromaDB by cosine similarity
- [ ] Verify: retrieve chunks from an uploaded document with a relevant query

### Step 1.6 — Wire Context into LLM Client (`llm/client.py`)
- [ ] Update `ask()` signature: add `context: list[dict] | None = None`
- [ ] Update `_build_messages()`: inject context as second `cache_control: ephemeral` block
- [ ] Context block format: `[Source: filename | Chunk N]\n<text>\n\n...`
- [ ] System prompt instructs LLM to cite sources and prefer document context
- [ ] Verify: call `ask()` with context, confirm response references the document

### Step 1.7 — Observability Hooks (`observability/tracer.py`)
- [ ] Create `src/medic_agent/observability/__init__.py`
- [ ] Create `src/medic_agent/observability/tracer.py`:
  - `Session` dataclass: `session_id, timestamp, use_case, model_id, query, chunks_retrieved, system_prompt_version, response, latency_ms, token_usage, error`
  - `log_session(session: Session) -> None` — writes JSON to `data/sessions/` AND sends to LangFuse
  - LangFuse trace wraps the full pipeline: retrieval span + LLM span
- [ ] Create `data/sessions/` directory; add to `.gitignore`
- [ ] Wire `log_session()` call into `llm/client.py` after each `ask()` completes
- [ ] Verify: submit a query → check LangFuse dashboard → see trace with retrieval + LLM spans

### Step 1.8 — Update Streamlit UI (`ui/app.py`)

Restructure into three tabs: `st.tabs(["🤖 Agent", "📊 Observability", "🧪 Evaluation"])`

**Tab 1 — Agent:**
- [ ] Sidebar: use case selector (Medical Coding / Ambient Note Taking)
- [ ] Sidebar: file uploader (PDF + TXT), document list with delete buttons
- [ ] Use case selector drives system prompt, input label, and default query text
- [ ] On upload: ingest → embed → store pipeline; show success/duplicate/error message
- [ ] On query: retrieve top-5 → inject context → call `ask()` → display response + citations
- [ ] Verify: full coding flow and full ambient flow work end-to-end

**Tab 2 — Observability:**
- [ ] Read `data/sessions/` JSONL files into a DataFrame
- [ ] Render summary stats bar: total sessions, avg latency, avg cost, error count
- [ ] Render session log table with filters (use case, date range)
- [ ] Expandable row: full query, response, retrieved chunks, prompt version, token breakdown
- [ ] "Export CSV" button
- [ ] "Open in LangFuse" button → `st.link_button()` to user's LangFuse project URL (from config)
- [ ] Verify: run 3 queries in Tab 1 → Tab 2 shows all 3 with correct metadata

**Tab 3 — Evaluation:**
- [ ] Render golden cases table with checkboxes (select which cases to run)
- [ ] Layer selector checkboxes: Layer 1 (free), Layer 2 (RAGAS), Layer 3 (LLM-as-judge)
- [ ] Show estimated time and cost warning before running Layer 2/3
- [ ] "Run Evaluation" button → calls `EvalRunner.run_all()` with spinner
- [ ] Results table: case ID, L1 pass/fail, RAGAS faithfulness, judge overall score
- [ ] Delta column vs baseline (green/amber/red)
- [ ] "Set as Baseline" button → writes scores to `tests/eval/baseline.json`
- [ ] Score history chart (scores over time, keyed by timestamp)
- [ ] "Export Results" button
- [ ] "Open in LangFuse" → eval traces in cloud dashboard
- [ ] Verify: run Layer 1 only → all 5 cases show results instantly

### Step 1.9 — Unit Tests
- [ ] `tests/rag/test_ingestor.py`: chunk count, overlap, metadata fields
- [ ] `tests/rag/test_embedder.py`: mocked HuggingFace calls, correct dimensions
- [ ] `tests/rag/test_store.py`: in-memory ChromaDB (not persistent) for test isolation
- [ ] `tests/rag/test_retriever.py`: mocked embedder + in-memory store
- [ ] `tests/observability/test_tracer.py`: mocked LangFuse, verify session fields captured
- [ ] All tests pass with `uv run pytest`

### Step 1.10 — Evaluation Runner + Tests
- [ ] Create `src/medic_agent/evaluation/__init__.py`
- [ ] Create `src/medic_agent/evaluation/runner.py`:
  - `EvalResult` dataclass: `{case_id, layer1_pass, ragas_scores, judge_scores, timestamp}`
  - `run_layer1(case) -> dict` — deterministic checks (ICD-10 regex, CPT format, SOAP sections)
  - `run_layer2(case, retrieved_chunks) -> dict` — RAGAS faithfulness + context precision
  - `run_layer3(case, response) -> dict` — Claude Sonnet judges on rubric, returns JSON scores
  - `run_all(cases, layers) -> list[EvalResult]` — called by both UI and pytest
  - Scores written to LangFuse as evaluation traces
- [ ] Create `tests/eval/test_eval.py`:
  - Loads `golden_cases.json`
  - Calls `EvalRunner.run_all()` 
  - Asserts Layer 1 passes for all cases
  - Asserts no judge score regresses >0.5 vs `baseline.json` (if baseline exists)
- [ ] Run `uv run pytest tests/eval/ -v -m eval` — separate pytest mark so it doesn't run with unit tests
- [ ] After first passing run: click "Set as Baseline" in Tab 3 → `baseline.json` created

**V1 Complete when:** All items above are checked off.

---

## Phase 2 — Multi-Persona (V2)

> Do not begin V2 until V1 is complete and tested.

**Goal:** Agent adapts behavior based on who is asking (clinician / admin / patient).

### Planned Steps (to be detailed when V2 begins)
- [ ] Define persona profiles and their system prompts
- [ ] Add persona selector to UI
- [ ] Route queries through persona-appropriate context
- [ ] [PLACEHOLDER]

---

## Phase 3 — Production-Ready (V3)

> Do not begin V3 until V2 is complete and tested.

**Goal:** App is deployable, has auth, is shareable.

### Planned Steps (to be detailed when V3 begins)
- [ ] Add user authentication
- [ ] Containerize with Docker
- [ ] Deploy to cloud (provider TBD)
- [ ] HIPAA compliance review if real patient data is involved
- [ ] [PLACEHOLDER]

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
