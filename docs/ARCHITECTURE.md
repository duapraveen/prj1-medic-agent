# Architecture Document — medic-agent

**Version:** 0.1  
**Status:** Draft  
**Last Updated:** 2026-05-09  

---

## 1. System Overview

medic-agent is a locally-run healthcare AI assistant.
In V0, it is a simple request-response loop: user query → LLM → displayed response.
The architecture is designed to be extended in V1 with a context/RAG layer without rewriting the core.

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

## 3. V1 Architecture — Planned Extension (RAG Layer)

```
┌──────────────────────────────────────────────────────────┐
│                     Streamlit Web UI                      │
│  + Document upload widget                                 │
│  + Source citation display                                │
└─────────────────────┬────────────────────────────────────┘
                      │
         ┌────────────┴────────────┐
         ▼                         ▼
┌─────────────────┐     ┌──────────────────────┐
│  LLM Client     │     │  Context/RAG Layer    │
│  (unchanged)    │     │  (new in V1)          │
│                 │     │  - Document ingestion  │
│                 │     │  - Chunking            │
│                 │     │  - Embedding           │
│                 │     │  - Vector search       │
└────────┬────────┘     └──────────┬───────────┘
         │                         │
         └────────────┬────────────┘
                      ▼
              ┌───────────────┐
              │  LiteLLM      │
              └───────────────┘
                      │
              ┌───────┴──────────────────┐
              ▼                           ▼
     ┌─────────────┐          ┌──────────────────────┐
     │ Claude API  │          │  Vector DB (local)    │
     └─────────────┘          │  ChromaDB (planned)   │
                              └──────────────────────┘
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

### 4.3 `src/medic_agent/api/routes.py` — FastAPI Layer (optional)
- HTTP API layer for when a REST interface is needed
- Thin layer: validates request, delegates to `llm.client`, returns response
- Not required for V0 (Streamlit calls Python directly)

### 4.4 `src/medic_agent/config/settings.py` — Configuration
- Loads and validates environment variables on startup
- Defines available models and their display names
- Central place for all app settings
- Raises clear errors if required config is missing

### 4.5 `.env` — Secrets
- `ANTHROPIC_API_KEY` — required for V0
- `OPENAI_API_KEY` — optional, needed when GPT support is added
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
| pytest | latest | Testing | Standard Python test framework |

| FastAPI | latest | API layer | Approved; use when an HTTP interface is needed alongside Streamlit |

**Not used and why:**
- **LangChain / LlamaIndex** — adds heavy abstraction before fundamentals are understood; will evaluate for V1 RAG layer
- **Flask** — FastAPI is approved instead; Flask is excluded
- **Docker** — unnecessary overhead for local V0
- **SQLite / Postgres** — no persistence needed in V0

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

## 9. Security Considerations

- API keys loaded from `.env` only — never in source code
- `.env` in `.gitignore` — never committed
- No user data persisted in V0
- No authentication needed in V0 (local single-user)
- **V1 consideration:** If real patient data is involved, HIPAA compliance review required before any cloud deployment

---

## 10. V1 RAG Layer — Technical Notes (Future Reference)

<!-- TODO: Flesh this out when V1 planning begins -->

**Planned components:**
- **Document ingestion:** Accept PDF, plain text, FHIR JSON
- **Chunking strategy:** [PLACEHOLDER — recursive character splitting vs. semantic chunking]
- **Embedding model:** [PLACEHOLDER — local model (nomic-embed-text via Ollama) vs. API (OpenAI ada-002)]
- **Vector store:** ChromaDB (local, free, no server needed)
- **Retrieval strategy:** [PLACEHOLDER — top-k similarity vs. MMR]
- **Context injection:** Retrieved chunks injected into system prompt before LLM call

---

## 11. Revision History

| Version | Date | Change |
|---|---|---|
| 0.1 | 2026-05-09 | Initial scaffold |
| 0.2 | 2026-05-09 | src/ structure, FastAPI approved, file paths updated |
