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
| huggingface_hub | latest | Embeddings API client | InferenceClient for SapBERT feature extraction via HuggingFace API |
| pytest + pytest-mock | latest | Testing | Standard Python test framework with mocking support |

**Embedding model: SapBERT**
- Model ID: `cambridgeltl/SapBERT-from-PubMedBERT-fulltext`
- Why: purpose-built for UMLS medical entity linking — covers ICD-10, SNOMED CT, CPT, LOINC, RxNorm
- Outputs: 768-dimensional vectors
- Accessed via: HuggingFace Inference API (`HUGGINGFACE_API_KEY`)
- Why not OpenAI embeddings: general-purpose models lack medical ontology grounding; ICD codes, SNOMED synonyms, and clinical abbreviations are significantly better represented in SapBERT
- Why not local SapBERT: HuggingFace Inference API avoids local GPU/memory overhead; free tier is sufficient for dev use

**Not used and why:**
- **LangChain / LlamaIndex** — we understand the RAG fundamentals now; still adds unnecessary abstraction
- **Flask** — FastAPI approved instead
- **OpenAI embeddings** — replaced by SapBERT; medical ontology grounding is required for coding use cases
- **Docker** — unnecessary overhead for local app
- **Postgres / SQLite** — ChromaDB covers all persistence needs

---

## 6. LLM Model Configuration — V0

| Display Name | LiteLLM Model ID | Use Case |
|---|---|---|
| Claude Haiku (Fast) | `claude-haiku-4-5-20251001` | **Default** — fast, low cost |
| Claude Sonnet (Balanced) | `claude-sonnet-4-6` | Higher quality when needed |
| Claude Opus (Powerful) | `claude-opus-4-6` | Complex clinical reasoning |

---

## 7. System Prompt Strategy

Two specialized system prompts — one per use case. Selected via the use case selector in the UI.
Both live in `config/settings.py` and are passed into `llm/client.py` at call time.

### Use Case 1 — Medical Coding

```
You are a certified medical coding specialist with expertise in ICD-10-CM, ICD-10-PCS,
CPT, and HCPCS Level II coding systems.

When given clinical documentation about a patient encounter, you:
1. Identify all diagnoses and assign accurate ICD-10-CM codes with full specificity
2. Identify all procedures and assign appropriate CPT / HCPCS codes
3. Sequence codes correctly (principal diagnosis first per UHDDS guidelines)
4. Follow Official Coding Guidelines, AHA Coding Clinic, and AMA CPT guidelines
5. Cite the specific document and section that supports each code assigned
6. Flag documentation gaps that prevent accurate or specific code assignment

Format your response as:
  DIAGNOSIS CODES (ICD-10-CM)
  PROCEDURE CODES (CPT / HCPCS)
  DOCUMENTATION GAPS

Never fabricate codes. If documentation is insufficient, state what is missing.
```

### Use Case 2 — Ambient Note Taking

```
You are a clinical documentation specialist with expertise in SOAP note writing
and medical coding from encounter transcripts.

When given a physician-patient conversation transcript, you:
1. Produce a structured SOAP note:
   S — Subjective: chief complaint, HPI, ROS, patient-reported symptoms
   O — Objective: exam findings, vitals, labs/imaging mentioned
   A — Assessment: diagnoses with clinical reasoning
   P — Plan: treatments, medications, orders, referrals, follow-up
2. Assign ICD-10-CM codes for all diagnoses in the Assessment
3. Identify CPT codes for any procedures performed or ordered
4. Flag documentation gaps (missing SOAP elements, vague diagnoses)

Distinguish clearly between patient-reported (Subjective) and clinician-observed
(Objective) information. Never infer clinical facts not present in the transcript.
```

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
| Use cases | Medical Coding + Ambient Note Taking | Highest-value, most technically demanding healthcare workflows |
| Document types | PDF, plain text (.txt) | Most common in healthcare; FHIR JSON deferred to V2 |
| Chunking | Recursive character split | Simple, effective; 1000 char window, 200 char overlap |
| Embedding model | SapBERT via HuggingFace Inference API | Medical ontology-aware; covers ICD-10, SNOMED, CPT, LOINC, RxNorm |
| Vector store | ChromaDB PersistentClient | Local, free, no server, persists to data/chroma/ |
| Retrieval strategy | Cosine similarity, top-5 | Simple and effective; MMR deferred to V2 |
| Storage model | Persistent — upload once, use always | Single user; re-upload is unnecessary friction |
| System prompts | Two prompts, one per use case | Coding and ambient note taking require fundamentally different instructions |
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

## 12. Observability Architecture

**Tool:** LangFuse (cloud free tier — 50k obs/month, 30-day retention, prompt versioning, LLM-as-judge eval)
**Why LangFuse over Datadog:** Datadog is infrastructure APM — not LLM-native. LangFuse is purpose-built for tracing LLM pipelines with retrieval spans, token costs, and evaluation hooks.
**Why LangFuse over Arize Phoenix:** Equivalent tracing quality; LangFuse adds prompt versioning (critical when iterating system prompts) and integrated eval dataset management.

### What Gets Traced Per Session

```
LangFuse Trace (1 per user query)
├── span: retrieval
│   ├── input: user query
│   ├── output: [{text, source_filename, chunk_index, similarity_score}]
│   └── latency_ms
└── span: llm_call
    ├── input: {model_id, system_prompt_version, context_block, user_query}
    ├── output: response text
    ├── token_usage: {prompt_tokens, completion_tokens, cached_tokens}
    └── latency_ms

Session metadata: session_id, timestamp, use_case, model_id, error (if any)
```

### Local Fallback

`tracer.py` also writes each session as a JSON line to `data/sessions/YYYY-MM-DD.jsonl`.
This is the auditability record that survives LangFuse's 30-day retention window.

### Prompt Versioning

All system prompts are registered in LangFuse as named, versioned prompts.
When a prompt changes, a new version is created — every trace is linked to its prompt version.
This enables: "did changing the coding prompt from v2 → v3 improve scores?"

---

## 13. Evaluation Framework

### Three-Layer Hybrid

**Layer 1 — Deterministic (always run, instant, free)**
| Check | Method |
|---|---|
| ICD-10-CM format | Regex: `^[A-Z]\d{2}(\.\w{1,4})?$` |
| CPT code format | Regex: `^\d{5}$` |
| SOAP sections present | String search for S/O/A/P headers |
| Response not empty | Length check |

**Layer 2 — RAGAS (RAG pipeline quality)**
| Metric | Measures |
|---|---|
| Faithfulness | Does LLM response stay grounded in retrieved chunks? |
| Context Precision | Were the retrieved chunks actually relevant to the query? |
| Answer Relevancy | Does the response address the query? |

**Layer 3 — LLM-as-Judge (quality, Claude Sonnet judges Haiku)**

*Coding rubric (1–5 per dimension):*
- `code_accuracy`: Are suggested codes clinically appropriate?
- `code_completeness`: Are important codes missing?
- `citation_quality`: Does each code cite supporting documentation?
- `hallucination`: Does response include codes not supported by docs? (5=none)
- `gap_identification`: Did the model correctly flag documentation gaps?

*Ambient/SOAP rubric (1–5 per dimension):*
- `soap_completeness`: All 4 sections present and appropriately populated?
- `subj_obj_distinction`: Patient-reported info in S, clinician-observed in O?
- `code_accuracy`: Are ICD/CPT codes clinically appropriate?
- `faithfulness`: Does the note stick to the transcript — no fabricated facts?
- `clinical_quality`: Would this SOAP be acceptable in a clinical setting?

### Golden Dataset
- Location: `tests/eval/golden_cases.json`
- 3 coding cases + 2 SOAP cases (synthetic, de-identified)
- Each case: `{id, use_case, description, input, expected, judge_criteria}`
- Scores written back to LangFuse as evaluation traces
- Baseline scores locked after V1 → regression detected automatically

---

## 11. Revision History

| Version | Date | Change |
|---|---|---|
| 0.1 | 2026-05-09 | Initial scaffold |
| 0.2 | 2026-05-09 | src/ structure, FastAPI approved, file paths updated |
| 0.3 | 2026-05-09 | V1 RAG layer: decisions locked, diagrams updated, components added |
| 0.4 | 2026-05-09 | Specialized to coding + ambient; SapBERT replaces OpenAI embeddings; two system prompts added |
