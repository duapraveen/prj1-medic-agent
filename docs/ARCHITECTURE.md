# Architecture Document — medic-agent

**Version:** 0.7  
**Status:** Active  
**Last Updated:** 2026-06-16  

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

### Evaluation Runner Architecture

The eval logic lives in `src/medic_agent/evaluation/runner.py` — a callable Python module, NOT only in pytest. This allows both the Streamlit UI (Tab 3) and `pytest` to invoke the same evaluation logic.

```
src/medic_agent/evaluation/
├── __init__.py
└── runner.py          ← EvalRunner class: run_layer1(), run_layer2(), run_layer3(), run_all()
                          Returns: EvalResult dataclass with scores per case per layer

tests/eval/
├── golden_cases.json  ← ground truth
└── test_eval.py       ← pytest wrapper: calls runner.py, asserts no regressions vs baseline
```

`EvalRunner.run_all(cases, layers)` is the single entry point called by both UI and pytest.

---

### Prompt Persistence Design

System prompts are editable at runtime via Tab 2. Persistence works in two layers:

```
config/settings.py          ← DEFAULT prompts (factory reset source, never modified)
        │
        ▼ (on first run, or after Reset to Defaults)
data/prompts.json           ← user-edited prompts (persists across restarts)
        │
        ▼ (on every Save)
LangFuse Prompt Registry    ← versioned prompt history (v1, v2, v3...)
                               every trace links to the version that produced it
```

**Load order on startup:**
1. Check if `data/prompts.json` exists
2. If yes → load from `data/prompts.json`
3. If no → load from `config/settings.py` defaults

**On "Save Prompts":**
1. Write to `data/prompts.json`
2. Push new version to LangFuse (`langfuse.create_prompt()`)
3. Update in-memory prompt cache
4. Display new version number in Tab 2

**On "Reset to Defaults":**
1. Reload from `config/settings.py` into text areas (does NOT auto-save)
2. User must click Save to make it permanent

---

### User Workflow — Observability

```
[User submits query in Tab 1]
        ↓ automatic
[tracer.py captures session]
        ├── writes JSON line to data/sessions/YYYY-MM-DD.jsonl
        └── sends trace to LangFuse (retrieval span + LLM span)
        ↓
[User opens Tab 2]
        ├── reads data/sessions/ → renders session table + summary stats
        └── "Open in LangFuse" → browser link to cloud dashboard
```

### User Workflow — Evaluation

```
[User opens Tab 3]
        ↓
[Selects cases + layers]
        ↓
[Clicks "Run Evaluation"]
        ↓
[EvalRunner.run_all() executes]
        ├── Layer 1: regex checks, section presence (~2s)
        ├── Layer 2: RAGAS pipeline against live RAG (~3 min)
        └── Layer 3: Claude Sonnet judges each case (~5 min)
        ↓
[Results rendered in Tab 3 table]
        ├── scores written to LangFuse as eval traces
        └── delta vs baseline highlighted (green/amber/red)
        ↓
[User clicks "Set as Baseline" after a good run]
        └── baseline.json written to tests/eval/
```

---

## 14. V2 Architecture — Multi-Agent Orchestration

V2 replaces the user-selected use case with an **orchestrator** that reads the raw
query and routes it to one of two specialized **multi-step agents**. Built on
**LangGraph** (user-approved; see §15). The single-shot `ask()` path from V1 is
kept for V0 tests, but the Agent tab now runs through the orchestrator.

### 14.1 Graph Topology

```
                       ┌───────────────────────────────┐
   user query  ─────►  │   orchestrator.run(query,      │
   + model + override  │     model_id, override,        │
                       │     judge_on)                  │
                       └───────────────┬───────────────┘
                                       ▼
                              ┌─────────────────┐
                              │  router (node)  │  hybrid: heuristic → LLM fallback
                              └────────┬────────┘
                          use_case     │   (manual override short-circuits)
                   ┌──────────────────┴──────────────────┐
                   ▼                                      ▼
        ┌────────────────────┐                 ┌────────────────────┐
        │  coding_agent      │                 │  ambient_agent     │
        │  extract           │                 │  retrieve          │
        │   → retrieve       │                 │   → soap           │
        │   → code           │                 │   → code           │
        │   → verify         │                 │   → verify         │
        └─────────┬──────────┘                 └─────────┬──────────┘
                  └──────────────────┬──────────────────┘
                                     ▼
                            ┌─────────────────┐
                            │  judge (node)   │  LLM-as-judge, Sonnet, default on
                            └────────┬────────┘
                                     ▼
                                    END
                       (returns route, response, chunks,
                        judge_scores, agent_steps)
```

### 14.2 New Package — `src/medic_agent/agents/`

| Module | Responsibility | Depends on |
|---|---|---|
| `state.py` | `AgentState` TypedDict shared across all nodes | — |
| `router.py` | Hybrid routing → `RouteDecision{use_case, method, confidence, reasoning}` | `llm.client.complete`, `settings` |
| `coding_agent.py` | Coding subgraph nodes: extract → retrieve → code → verify | `llm.client.complete`, `rag.retriever`, `settings` |
| `ambient_agent.py` | Ambient subgraph nodes: retrieve → soap → code → verify | `llm.client.complete`, `rag.retriever`, `settings` |
| `judge.py` | Online LLM-as-judge; shares rubric with `evaluation/runner.py` | `llm.client.complete`, `settings` |
| `orchestrator.py` | Builds the LangGraph; entry point `run(...)`; logs one Session | all of the above, `observability.tracer` |

`AgentState` fields: `query, model_id, override, route, use_case, chunks,
scratch (per-step intermediates), response, judge_scores, agent_steps`.

### 14.3 Router (Hybrid)

```
route(query, model_id, override)
  ├─ override != "Auto"     → RouteDecision(method="manual",  confidence=1.0)
  ├─ heuristic_route(query) → RouteDecision(method="heuristic") if a strong signal
  │     • ambient signals: speaker markers (Doctor:/Patient:/Dr./PT:), multi-turn dialogue
  │     • coding signals:  "ICD", "CPT", "code this encounter", "billing codes for"
  └─ else llm_route(query)  → RouteDecision(method="llm") — Haiku classifier
                              returns {use_case, confidence, reasoning}
```

`reasoning` and `method`/`confidence` are surfaced in the Tab 1 routing panel and
logged to LangFuse.

### 14.4 Agents (Multi-Step Subgraphs)

**Coding agent** — `extract` pulls documented diagnoses/procedures; `retrieve`
does entity-aware top-5 retrieval; `code` assigns ICD-10-CM/CPT with citations and
documentation gaps (uses the editable coding prompt); `verify` self-checks that
every code is supported by the documentation and finalizes the formatted output.

**Ambient agent** — `retrieve` pulls coding-reference context; `soap` drafts the
SOAP note from the transcript (uses the editable ambient prompt); `code` assigns
billing codes from the Assessment; `verify` self-checks faithfulness/section
completeness, adds documentation flags, and finalizes.

Each node appends a step record `{name, type, model_id, token_usage, latency_ms,
input_summary, output_summary}` to `state["agent_steps"]` for tracing.

### 14.5 Online LLM-as-Judge

`judge.py` scores every response (default on; sidebar toggle for cost control)
against the rubrics in §13. Judge model is **Sonnet** (smarter than the system
under test, per the locked decision). The Layer-3 judge logic in
`evaluation/runner.py` is refactored into a shared function that both the online
judge and the eval runner call — single source of truth, no duplication. The
`overall` score is attached to the LangFuse trace as a score and shown in Tab 3.

### 14.6 Multi-Agent Tracing

Nodes write structured step records into the `Session`; `tracer.py` emits one
nested trace per query (populate-then-emit, matching V1):

```
LangFuse Trace: medic-agent-query
├── span: router            input=query, output=route_decision
├── span: agent:{use_case}
│   ├── span: extract        (generation)  ← coding only
│   ├── span: retrieval      (retriever)
│   ├── span: code | soap     (generation)
│   └── span: verify         (generation)
└── span: judge             (generation), output=judge_scores
    └── score: judge_overall attached to the trace

Session metadata: session_id, timestamp, use_case (determined),
route_decision{method,confidence,reasoning}, model_id, error
```

New `Session` fields: `route_decision: dict`, `agent_steps: list[dict]`,
`judge_scores: dict`. The local JSONL fallback also records routing + judge data
so Tab 3 can render them without LangFuse.

### 14.7 LLM Client Change

`llm/client.py` gains `complete(model_id, system_prompt, user_query, context=None)
-> tuple[str, dict]` — a non-logging primitive returning `(text, token_usage)`.
Agent nodes call `complete()` (so a multi-step query does not produce N Sessions);
the orchestrator logs exactly one Session. `ask()` is unchanged.

### 14.8 UI & Eval Impact

- **Sidebar:** use-case radio → routing-mode selectbox (`Auto / Medical Coding /
  Ambient Note Taking`) + judge on/off toggle.
- **Tab 1:** single generic input; routing panel shows determined agent + method +
  confidence + reasoning; response + sources + judge scores below.
- **Tab 2:** expanded — edits **all** prompts, grouped by agent. `settings.py`
  holds the defaults (factory reset); `data/prompts.json` persists edits; each Save
  versions them in LangFuse. Layout:
  - *Router:* `router`
  - *Medical Coding Agent:* `extract`, `code`, `verify`
  - *Ambient Agent:* `soap`, `code`, `verify`
  - *Judge:* `coding`, `ambient`

  `data/prompts.json` schema (V2):
  ```json
  {
    "version": 3,
    "saved_at": "2026-06-16T...",
    "router": "...",
    "coding":  { "extract": "...", "code": "...", "verify": "..." },
    "ambient": { "soap": "...", "code": "...", "verify": "..." },
    "judge":   { "coding": "...", "ambient": "..." }
  }
  ```
  Agent nodes and the router/judge load their prompt from `prompts.json`, falling
  back to the `settings.py` default. (This redefines the V1 flat
  `{coding, ambient}` schema; `prompts.json` is gitignored user config, so no
  back-compat shim per the project rules — a missing key falls back to defaults.)
- **Tab 3:** session rows gain route decision + judge overall.
- **Eval:** `EvalRunner` runs cases through `orchestrator.run()`; Layer 1 adds a
  router-accuracy check against each golden case's known `use_case`.

---

## 15. V2 Decisions — Rationale

| Decision | Choice | Why this, not that |
|---|---|---|
| Orchestration framework | **LangGraph** | User explicitly approved it for the orchestration layer. This is a deliberate, logged reversal of the project's "no LangChain ecosystem" rule (§5) — scoped to orchestration only; the RAG pipeline stays hand-built. |
| Routing strategy | **Hybrid** (heuristic → LLM fallback) | Free/instant when the query has a clear signal; Haiku fallback only when ambiguous. Pure-LLM wastes a call on obvious cases; pure-heuristic is brittle. |
| User control | **Auto + manual override** | Matches the "system decides" goal while keeping an escape hatch when the router is wrong. |
| Agent shape | **Full multi-step subgraphs** | User wants genuinely agentic behavior, not prompt-swapping. Kept lean ("as needed", ~4 nodes each) to bound latency/cost. |
| Online evaluation | **LLM-as-judge per query, Sonnet, default on** | User wants judge metrics visible in observability traces, not just in the Eval tab. Toggleable to control cost. |
| Tracing approach | **Manual nested spans** (not LangChain callback handler) | Keeps the V1 populate-then-emit pattern, captures custom router/judge fields, and avoids coupling traces to callback internals. |
| Judge/rubric reuse | **Shared function** between online judge and eval runner | Single source of truth for rubrics and scoring. |
| Prompt editing scope | **All prompts editable in Tab 2**, grouped by agent | Full transparency and tuning across router, agent steps, and judge; `settings.py` holds defaults, `data/prompts.json` persists edits. |

**Latency/cost note:** a query is now up to ~6 LLM calls (router fallback + 4
agent steps + judge) vs. 1 in V1. On Haiku this is cheap; on Opus it is material.
The heuristic-first router and the judge toggle are the mitigations.

**Synchronous compliance:** LangGraph runs via `.invoke()` synchronously — the
"no async/streaming" rule (CLAUDE.md) still holds.

---

## 16. Revision History

| Version | Date | Change |
|---|---|---|
| 0.1 | 2026-05-09 | Initial scaffold |
| 0.2 | 2026-05-09 | src/ structure, FastAPI approved, file paths updated |
| 0.3 | 2026-05-09 | V1 RAG layer: decisions locked, diagrams updated, components added |
| 0.4 | 2026-05-09 | Specialized to coding + ambient; SapBERT replaces OpenAI embeddings; two system prompts added |
| 0.5 | 2026-05-10 | Three-tab UI design; evaluation runner architecture; user workflows for obs+eval |
| 0.6 | 2026-05-10 | Four-tab UI; Knowledge Base & Prompts tab; prompt persistence design |
| 0.7 | 2026-06-16 | V2 multi-agent orchestration: LangGraph topology, agents package, hybrid router, online judge, multi-agent tracing (§14–15) |
