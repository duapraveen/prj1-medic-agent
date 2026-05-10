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
- [ ] Add sidebar: use case selector (Medical Coding / Ambient Note Taking)
- [ ] Use case selector drives system prompt and input label/placeholder
- [ ] Add sidebar: file uploader (PDF + TXT), document list with delete buttons
- [ ] On upload: ingest → embed → store pipeline; show success/duplicate/error message
- [ ] On query: retrieve top-5 chunks → inject context → call `ask()` with selected system prompt
- [ ] Show source citations below response
- [ ] Verify: upload encounter doc → select "Medical Coding" → submit → get coded output with citations
- [ ] Verify: paste transcript → select "Ambient Note Taking" → submit → get SOAP note + codes

### Step 1.9 — Unit Tests
- [ ] `tests/rag/test_ingestor.py`: chunk count, overlap, metadata fields
- [ ] `tests/rag/test_embedder.py`: mocked HuggingFace calls, correct dimensions
- [ ] `tests/rag/test_store.py`: in-memory ChromaDB (not persistent) for test isolation
- [ ] `tests/rag/test_retriever.py`: mocked embedder + in-memory store
- [ ] `tests/observability/test_tracer.py`: mocked LangFuse, verify session fields captured
- [ ] All tests pass with `uv run pytest`

### Step 1.10 — Evaluation Golden Dataset + LLM-as-Judge
- [ ] Golden dataset lives at `tests/eval/golden_cases.json` (already created — see below)
- [ ] Create `tests/eval/test_eval.py`:
  - Layer 1 (deterministic): ICD-10 format regex, CPT format, SOAP section presence
  - Layer 2 (RAGAS): faithfulness, context precision, answer relevancy — run against RAG pipeline
  - Layer 3 (LLM-as-judge): Claude Sonnet judges Haiku output on structured rubric (1–5 per dimension)
  - Judge model: `claude-sonnet-4-6` (smarter than the system being judged)
- [ ] Scores written back to LangFuse as evaluation traces
- [ ] Run: `uv run pytest tests/eval/ -v` produces a score report
- [ ] Baseline scores recorded — this is the regression benchmark

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
