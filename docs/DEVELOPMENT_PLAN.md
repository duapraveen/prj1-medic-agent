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
Phase 1 (V1): Context Layer    ← COMPLETE ✅
Phase 2 (V2): Multi-Persona    ← YOU ARE HERE (not started)
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
| 0.6 | 2026-05-10 | Four-tab UI; KB & Prompts tab in Step 1.8; prompt persistence design |
| 0.7 | 2026-05-10 | V1 marked complete; all Step 1.1–1.10 items checked off |
