# Architecture Document — medic-agent

**Version:** 0.2  
**Status:** Active  
**Last Updated:** 2026-05-09  

---

## 1. System Overview

medic-agent is a locally-run healthcare AI assistant.
V0 was a simple request-response loop: user query → LLM → displayed response.
V1 adds a RAG (Retrieval-Augmented Generation) layer: users upload documents once, and the agent grounds its answers in those documents on every subsequent query.

---

## 2. V0 Architecture — Component Diagram

```
┌─────────────────────────────────────────────────────┐
│                    User's Browser                   │
└───────────────────────┬─────────────────────────────┘
                        │ HTTP (localhost)
┌───────────────────────▼─────────────────────────────┐
│        Streamlit Web UI (src/medic_agent/ui/app.py)  │
│  - Model selector dropdown                          │
│  - Query input (text area)                          │
│  - Submit button                                    │
│  - Response display area                            │
└───────────────────────┬─────────────────────────────┘
                        │ Python function call
┌───────────────────────▼─────────────────────────────┐
│    LLM Client Layer (src/medic_agent/llm/client.py)  │
│  - Accepts: model_id, system_prompt, user_query     │
│  - Returns: response text                           │
│  - Wraps: LiteLLM                                   │
└───────────────────────┬─────────────────────────────┘
                        │ HTTPS API call
┌───────────────────────▼─────────────────────────────┐
│              LiteLLM (third-party library)           │
│  - Unified interface for 100+ LLM providers         │
│  - Routes to correct provider API                   │
└───────────────────────┬─────────────────────────────┘
                        │
            ┌───────────┴───────────┐
            ▼                       ▼
    ┌───────────────┐     ┌──────────────────┐
    │  Anthropic    │     │  OpenAI (future) │
    │  Claude API   │     │  GPT models      │
    └───────────────┘     └──────────────────┘
```

---

## 3. V1 Architecture — RAG Layer (Current)

Two distinct flows: **upload** (document ingestion) and **query** (retrieval + generation).

```
UPLOAD FLOW
──────────────────────────────────────────────────────────────
User uploads file (PDF / TXT)
        │
        ▼
┌───────────────────┐    ┌───────────────────┐    ┌──────────────────────┐
│  rag/ingestor.py  │───►│  rag/embedder.py  │───►│  rag/store.py        │
│  Parse file       │    │  OpenAI           │    │  ChromaDB            │
│  Split into chunks│    │  text-embedding-  │    │  Persistent on disk  │
│                   │    │  3-small          │    │  data/chroma/        │
└───────────────────┘    └───────────────────┘    └──────────────────────┘

QUERY FLOW
──────────────────────────────────────────────────────────────
User types query
        │
        ▼
┌───────────────────┐    ┌──────────────────────┐
│  rag/embedder.py  │───►│  rag/retriever.py    │
│  Embed the query  │    │  Similarity search   │
│                   │    │  Top-k chunks        │
└───────────────────┘    └──────────┬───────────┘
                                    │ context chunks
                                    ▼
                         ┌──────────────────────┐
                         │  llm/client.py       │
                         │  _build_messages():  │
                         │  [system ⚡cached]   │
                         │  [context ⚡cached]  │
                         │  [user query]        │
                         └──────────┬───────────┘
                                    │
                                    ▼
                              ┌──────────┐
                              │ LiteLLM  │
                              └────┬─────┘
                                   │
                              ┌────▼─────┐
                              │ Claude   │
                              │ API      │
                              └──────────┘
                                   │
                                   ▼
                    Response + source citations shown in UI
```

---

## 4. Component Descriptions

### 4.1 `src/medic_agent/ui/app.py` — Streamlit UI
- Entry point: run with `streamlit run src/medic_agent/ui/app.py`
- Handles all UI rendering and user interaction
- Calls `llm.client` functions — no business logic here
- No direct LLM or API calls — UI layer only

### 4.2 `src/medic_agent/llm/client.py` — LLM Abstraction Layer
- Single public function: `ask(model_id, system_prompt, user_query) → str`
- All LLM interaction goes through this layer
- Isolates the rest of the app from LiteLLM specifics
- Extended in V1 with context injection before the LLM call

### 4.3 `src/medic_agent/rag/ingestor.py` — Document Ingestion (V1)
- Accepts PDF (bytes) or plain text (str), returns list of chunk dicts
- Each chunk: `{text, source_filename, chunk_index}`
- Chunking: 1000-character windows with 200-character overlap
- Supported types: `.pdf`, `.txt`

### 4.4 `src/medic_agent/rag/embedder.py` — Embedding (V1)
- `embed_texts(texts) -> list[list[float]]` — batch embed for ingestion
- `embed_query(query) -> list[float]` — single embed for retrieval
- Model: OpenAI `text-embedding-3-small` (1536 dimensions)
- API key: `OPENAI_API_KEY` from `.env`

### 4.5 `src/medic_agent/rag/store.py` — Vector Store (V1)
- Wraps ChromaDB `PersistentClient` pointed at `data/chroma/`
- `add_document(doc_id, chunks, embeddings)` — stores chunks + vectors
- `document_exists(doc_id)` — prevents duplicate ingestion
- `list_documents()` — returns all stored document names
- `delete_document(doc_id)` — removes document and its chunks

### 4.6 `src/medic_agent/rag/retriever.py` — Retrieval (V1)
- `retrieve(query, k=5) -> list[dict]` — returns top-k relevant chunks
- Embeds the query, queries ChromaDB, returns chunks with metadata
- Strategy: cosine similarity, top-5

### 4.7 `src/medic_agent/api/routes.py` — FastAPI Layer (optional)
- HTTP API layer for when a REST interface is needed
- Thin layer: validates request, delegates to `llm.client`, returns response

### 4.8 `src/medic_agent/config/settings.py` — Configuration
- Loads and validates all environment variables on startup
- Defines available models, embedding model, ChromaDB path
- Raises clear errors if required keys are missing

### 4.9 `.env` — Secrets
- `ANTHROPIC_API_KEY` — required (LLM calls)
- `OPENAI_API_KEY` — required from V1 (embeddings)
- Never committed to git

---

## 5. Tech Stack — Decisions and Rationale

| Technology | Version | Purpose | Why This, Not That |
|---|---|---|---|
| Python | 3.11+ | Core language | User's primary language; 3.11 has better performance and error messages than 3.9 |
| Streamlit | latest | Web UI | Python-native UI with zero HTML/CSS/JS; perfect for data/AI apps; local dev server built-in |
| LiteLLM | latest | LLM abstraction | One API for all providers; drop-in Claude→GPT switching; active project |
| uv | latest | Package manager | 10-100x faster than pip; built-in virtual env management; modern standard |
| python-dotenv | latest | Secret management | Industry standard for .env loading; simple |
| FastAPI | latest | API layer | Approved for HTTP interface alongside Streamlit |
| ChromaDB | latest | Vector database | Local, persistent, no server required; purpose-built for embeddings |
| pypdf | latest | PDF parsing | Lightweight pure-Python PDF reader; no system dependencies |
| openai | latest | Embeddings API | text-embedding-3-small: high quality, cheap ($0.02/M tokens), simple |
| pytest + pytest-mock | latest | Testing | Standard Python test framework with mocking support |

**Not used and why:**
- **LangChain / LlamaIndex** — we understand the RAG fundamentals now; still adding unnecessary abstraction for our use case
- **Flask** — FastAPI approved instead
- **Ollama / local embeddings** — API embeddings are cheaper per token and simpler to operate; revisit if offline requirement emerges
- **Docker** — unnecessary overhead for local app
- **Postgres / SQLite** — ChromaDB covers our persistence needs

---

## 6. LLM Model Configuration — V0

| Display Name | LiteLLM Model ID | Use Case |
|---|---|---|
| Claude Haiku (Fast) | `claude-haiku-4-5-20251001` | **Default** — fast, low cost |
| Claude Sonnet (Balanced) | `claude-sonnet-4-6` | Higher quality when needed |
| Claude Opus (Powerful) | `claude-opus-4-6` | Complex clinical reasoning |

---

## 7. System Prompt Strategy


The system prompt is what tells the LLM to behave as a healthcare assistant.
This is a key product decision — the quality of the system prompt directly impacts response quality.

**V0 Approach:** Single hardcoded system prompt loaded from `config.py`

```
System Prompt:
"You are a knowledgeable healthcare assistant. You help clinicians, 
administrators, and patients with healthcare-related questions. 
You provide accurate, evidence-based information. You always recommend 
consulting a licensed healthcare professional for medical decisions. 
You do not provide diagnoses."
```

**V1 Approach:** System prompt varies by user persona (clinician vs. patient vs. admin)

---

## 8. Prompt Caching Strategy

Anthropic prompt caching is enabled on all requests via `cache_control: {"type": "ephemeral"}` markers in the message content blocks. Cached content is reused for 5 minutes at ~10% of the normal token cost.

| Content Block | Cached? | Notes |
|---|---|---|
| System prompt | Yes | Marked in `_build_messages()` in `llm/client.py` |
| Retrieved documents (V1) | Yes | Will be injected as a second marked block in the same helper |
| User query | No | Changes every request — never cached |

**Minimum token threshold for caching:**
- Claude Haiku: 2048 tokens
- Claude Sonnet / Opus: 1024 tokens

The V0 system prompt (~60 tokens) falls below the threshold and won't cache in practice.
V1 document context will comfortably exceed it.

All caching logic is isolated to `_build_messages()` in `llm/client.py`. The UI and config layers are unaware of it.

---

## 9. Error Handling Strategy

| Error Type | Handling |
|---|---|
| Missing API key | Fail at startup with clear message |
| API rate limit | Show user-friendly message, do not crash |
| API timeout | Show user-friendly message with retry suggestion |
| Invalid model selection | Should not be possible (controlled dropdown) |
| Empty query | Validate before API call, show inline error |

---

## 10. Security Considerations

- API keys loaded from `.env` only — never in source code
- `.env` in `.gitignore` — never committed
- No user data persisted in V0
- No authentication needed in V0 (local single-user)
- **V1 consideration:** If real patient data is involved, HIPAA compliance review required before any cloud deployment

---

## 11. V1 RAG Layer — Decisions

| Decision | Choice | Reason |
|---|---|---|
| Document types | PDF, plain text (.txt) | Most common in healthcare; FHIR JSON deferred to V2 |
| Chunking | Recursive character split | Simple, effective; 1000 char window, 200 char overlap |
| Embedding model | OpenAI text-embedding-3-small | 1536 dims, cheap, no local setup required |
| Vector store | ChromaDB PersistentClient | Local, free, no server, persists to data/chroma/ |
| Retrieval strategy | Cosine similarity, top-5 | Simple and effective; MMR deferred to V2 |
| Storage model | Persistent — upload once, use always | Single user; re-upload is unnecessary friction |
| Context injection | Second cached block in _build_messages() | Hook already in place in llm/client.py |
| Duplicate prevention | document_exists() check before ingestion | Avoid re-embedding already stored files |

**Context block format injected into LLM:**
```
CONTEXT FROM DOCUMENTS:

[Source: clinical_guidelines.pdf | Chunk 3]
<chunk text>

[Source: discharge_notes.txt | Chunk 1]
<chunk text>
...
```

**Prompt caching in V1:**
Retrieved context is injected as a second `cache_control: ephemeral` block in `_build_messages()`.
This means queries against the same retrieved context are served at ~10% token cost after the first call within 5 minutes.

---

## 11. Revision History

| Version | Date | Change |
|---|---|---|
| 0.1 | 2026-05-09 | Initial scaffold |
| 0.2 | 2026-05-09 | src/ structure, FastAPI approved, file paths updated |
| 0.3 | 2026-05-09 | V1 RAG layer: decisions locked, diagrams updated, components added |
