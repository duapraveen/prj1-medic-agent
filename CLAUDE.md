# medic-agent — AI Assistant Context

This file is read by Claude Code at the start of every session.
It defines the rules, constraints, and conventions for this project.
NEVER deviate from these instructions without explicit user approval.

---

## Project Overview

**medic-agent** is a healthcare AI agent specialized for two high-value clinical workflows:

1. **Medical Coding** — given encounter documents, produce ICD-10-CM, CPT, and HCPCS codes with citations and documentation gap analysis
2. **Ambient Note Taking** — given a physician-patient conversation transcript, produce a structured SOAP note and accurate billable code list

**Current phase:** V2 — Multi-agent orchestration (planning approved 2026-06-16): a LangGraph orchestrator routes each query (hybrid heuristic→LLM router, with manual override) to one of two specialized multi-step agents (Medical Coding / Ambient Note Taking), with an online LLM-as-judge scoring every response and nested multi-agent LangFuse tracing. V1 (RAG layer) is complete.

---

## Tech Stack (Locked)

| Layer | Choice | Reason |
|---|---|---|
| Language | Python 3.11+ | User's primary language |
| Web UI | Streamlit | Python-native, no frontend knowledge needed, local |
| API Layer | FastAPI | Approved for use when an HTTP API layer is needed |
| LLM Abstraction | LiteLLM | Unified interface for Claude, GPT, Gemini etc. |
| Package Manager | uv | Modern, fast, replaces pip+venv |
| Config / Secrets | python-dotenv + .env | Standard, safe, local |
| Vector Database | ChromaDB (PersistentClient) | Local, no server, persists to data/chroma/ |
| PDF Parsing | pypdf | Lightweight, pure Python, no system dependencies |
| Embeddings | SapBERT via HuggingFace Inference API | Medical ontology-aware; ICD-10, SNOMED, CPT, LOINC. Model: `cambridgeltl/SapBERT-from-PubMedBERT-fulltext`. 768 dims. Via HUGGINGFACE_API_KEY |
| Observability | LangFuse (cloud free tier) | LLM-native tracing; prompt versioning; LLM-as-judge eval; 50k obs/month free |
| Evaluation | RAGAS + LLM-as-judge (Claude Sonnet) + deterministic checks | Three-layer hybrid; scores written to LangFuse |
| Orchestration (V2) | LangGraph | Multi-agent routing + agent subgraphs. User-approved 2026-06-16; scoped to the orchestration layer only (RAG stays hand-built) |
| Version Control | Git | Standard |

Do NOT suggest or introduce Flask, LangChain, LlamaIndex, or other frameworks unless the user explicitly approves it.
**Exception (V2):** LangGraph is approved for the orchestration layer only. This does NOT reopen LangChain/LlamaIndex for the RAG pipeline — embeddings, retrieval, and the vector store remain hand-built.

---

## Project File Structure

```
prj1-medic-agent/
├── CLAUDE.md                  ← this file
├── README.md                  ← setup and run instructions
├── docs/
│   ├── PRD.md                 ← product requirements
│   ├── ARCHITECTURE.md        ← technical design
│   ├── DEVELOPMENT_PLAN.md    ← phased roadmap
│   └── CONVENTIONS.md         ← coding standards
├── src/
│   └── medic_agent/           ← main package
│       ├── __init__.py
│       ├── ui/                ← Streamlit UI layer
│       │   ├── __init__.py
│       │   └── app.py         ← Streamlit entry point
│       ├── llm/               ← LLM abstraction layer
│       │   ├── __init__.py
│       │   └── client.py      ← LiteLLM wrapper
│       ├── api/               ← FastAPI layer (if/when added)
│       │   ├── __init__.py
│       │   └── routes.py
│       └── config/            ← configuration and settings
│           ├── __init__.py
│           └── settings.py
│       ├── observability/     ← tracing + LangFuse integration
│       │   ├── __init__.py
│       │   └── tracer.py      ← Session dataclass, JSON log, LangFuse spans
│       ├── evaluation/        ← eval engine (called by UI Tab 3 AND pytest)
│       │   ├── __init__.py
│       │   └── runner.py      ← EvalRunner: run_layer1/2/3/all(); EvalResult dataclass
│       ├── rag/               ← RAG pipeline (V1)
│       │   ├── __init__.py
│       │   ├── ingestor.py    ← file loading + chunking
│       │   ├── embedder.py    ← OpenAI embedding calls
│       │   ├── store.py       ← ChromaDB persistence operations
│       │   └── retriever.py   ← query → top-k relevant chunks
│       └── agents/            ← multi-agent orchestration (V2)
│           ├── __init__.py
│           ├── state.py       ← AgentState (LangGraph TypedDict)
│           ├── router.py      ← hybrid router → RouteDecision
│           ├── coding_agent.py    ← extract → retrieve → code → verify
│           ├── ambient_agent.py   ← retrieve → soap → code → verify
│           ├── judge.py       ← online LLM-as-judge (shared rubric)
│           └── orchestrator.py    ← builds LangGraph; run() entry point
├── data/
│   └── chroma/                ← ChromaDB on-disk storage (gitignored)
├── data/
│   ├── chroma/                ← ChromaDB on-disk storage (gitignored)
│   ├── sessions/              ← JSON session logs for auditability (gitignored)
│   └── prompts.json           ← user-edited system prompts (gitignored — user-specific)
├── tests/
│   ├── llm/
│   │   └── test_client.py
│   ├── rag/
│   │   ├── test_ingestor.py
│   │   ├── test_embedder.py
│   │   ├── test_store.py
│   │   └── test_retriever.py
│   ├── agents/                ← V2 multi-agent tests
│   │   ├── test_router.py
│   │   ├── test_coding_agent.py
│   │   ├── test_ambient_agent.py
│   │   ├── test_judge.py
│   │   └── test_orchestrator.py
│   ├── observability/
│   │   └── test_tracer.py
│   ├── observability/
│   │   └── test_tracer.py
│   └── eval/
│       ├── golden_cases.json  ← synthetic test cases with expected outputs
│       ├── baseline.json      ← locked baseline scores (set after first good run)
│       └── test_eval.py       ← pytest wrapper calling evaluation/runner.py
├── .env                       ← secrets (never commit this)
├── .env.example               ← template for .env (safe to commit)
├── .gitignore
└── pyproject.toml             ← project metadata and dependencies
```

Each substantial new feature gets its own subfolder under `src/medic_agent/`.

---

## Coding Rules

1. **Minimum code.** Solve only what is asked. No speculative features.
2. **Touch only what you must.** Don't refactor or clean up code outside the task scope.
3. **No comments unless the logic is non-obvious.** Don't narrate what the code does.
4. **No docstrings** on simple functions unless explicitly requested.
5. **Explicit over implicit.** Favor readable variable names over cleverness.
6. **One responsibility per function.** Keep functions small and focused. Max 60 lines per function — if it's longer, split it. Exceptions require a comment explaining why.
7. **Fail loudly at startup.** Validate required config (API keys etc.) on app init, not at call time.
8. **Never hardcode secrets.** All API keys must come from environment variables.
9. **Type hints required** on all function signatures.
10. **No backwards-compatibility shims.** If something is removed, remove it cleanly.

---

## Conventions (Summary — see docs/CONVENTIONS.md for full detail)

- File names: `snake_case.py`
- Function names: `snake_case`
- Class names: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Max line length: 88 characters (Black default)
- Imports: stdlib → third-party → local, separated by blank lines

---

## What NOT to Do

- Do NOT use LangChain or LlamaIndex. We build the RAG pipeline ourselves to understand the fundamentals.
- Do NOT add user authentication. Single user, local only.
- Do NOT deploy to cloud. Local MacBook only.
- Do NOT add async/streaming. Keep it synchronous.
- Do NOT use a relational database. ChromaDB covers all persistence needs.
- Do NOT re-embed a document that already exists in the store — always check `document_exists()` first.
- Do NOT make real API calls in unit tests. Mock all external calls (HuggingFace, Anthropic, ChromaDB, LangFuse).
- Eval tests (`tests/eval/`) MAY make real LLM calls — they are integration tests, not unit tests. Run them explicitly, not as part of the default `pytest` suite.
- Do NOT create new files unless absolutely necessary.

---

## V0 Acceptance Criteria — COMPLETE ✅ (2026-05-09)

- [x] User can type a healthcare query in the web UI
- [x] User can select a Claude model (haiku / sonnet / opus)
- [x] App sends query to selected model via LiteLLM
- [x] Response is displayed in the web UI
- [x] API key is loaded from .env, never hardcoded
- [x] App runs locally with: `uv run streamlit run src/medic_agent/ui/app.py`

## Current V1 Acceptance Criteria

- [ ] Use case selector (Medical Coding / Ambient Note Taking) switches system prompt
- [ ] User can upload a PDF or TXT file via the sidebar
- [ ] Uploaded documents are parsed, chunked, embedded (SapBERT), and stored in ChromaDB
- [ ] Documents persist across app restarts — no re-upload needed
- [ ] User can see the list of uploaded documents in the sidebar
- [ ] User can delete a document from the store
- [ ] Duplicate uploads are detected and skipped
- [ ] On query, top-5 relevant chunks are retrieved and injected as context
- [ ] Coding response: ICD-10-CM + CPT codes with document citations + gap flags
- [ ] Ambient response: full SOAP note + code list + documentation flags
- [ ] Context injection uses prompt caching (cache_control: ephemeral)
- [ ] All new modules have unit tests with mocked external calls
- [ ] HUGGINGFACE_API_KEY and LANGFUSE keys loaded from .env, never hardcoded
- [ ] App has four tabs: Agent, Knowledge Base & Prompts, Observability, Evaluation
- [ ] Tab 2 (KB & Prompts): document upload, document list with chunk counts, delete
- [ ] Tab 2: editable text areas for Coding and Ambient system prompts with labels
- [ ] Tab 2: Save Prompts → writes data/prompts.json + pushes new version to LangFuse
- [ ] Tab 2: Reset to Defaults → restores settings.py defaults into text areas (no auto-save)
- [ ] Tab 3 reads data/sessions/ and renders session log table + summary stats
- [ ] Tab 3 has "Open in LangFuse" button linking to cloud dashboard
- [ ] Tab 4 renders golden cases, layer selector, Run button, results table, baseline delta
- [ ] Each query produces a LangFuse trace with retrieval span + LLM span
- [ ] Session logs written to data/sessions/ as JSON lines
- [ ] EvalRunner callable from both Tab 3 UI and pytest
- [ ] Eval golden dataset (5 cases) passes Layer 1 deterministic checks
- [ ] Baseline scores set and stored in tests/eval/baseline.json after first passing run

> V1 acceptance criteria above are carried from the completed V1 phase and remain
> the baseline behavior V2 must not regress.

---

## V2 Acceptance Criteria — Multi-Agent Orchestration

- [x] User no longer selects the use case; they paste a query and click Submit
- [x] A LangGraph orchestrator (`agents/orchestrator.py`) is the single entry point for the Agent tab
- [x] Hybrid router picks the agent: heuristics on clear signals, Haiku LLM fallback when ambiguous
- [x] Sidebar offers routing mode `Auto / Medical Coding / Ambient Note Taking` (manual override)
- [x] After Submit, the UI panel displays the determined agent + method + confidence + one-line reasoning
- [x] Coding agent runs multi-step: extract → retrieve → code → verify
- [x] Ambient agent runs multi-step: retrieve → soap → code → verify
- [x] Each agent node calls `llm.client.complete()` (non-logging); orchestrator logs exactly one Session per query
- [x] Online LLM-as-judge (Sonnet) scores every response by default; sidebar toggle can disable it
- [x] Judge logic is shared between `agents/judge.py` and `evaluation/runner.py` Layer 3 (no duplication)
- [x] LangFuse trace shows nested spans: router → agent:{use_case} → per-step spans → judge, with judge overall as a score (span structure asserted by tracer unit test)
- [x] `data/sessions/` JSONL records route_decision + judge_scores; Tab 3 renders them
- [x] Tab 2 edits ALL prompts, grouped by agent: router; coding (extract/code/verify); ambient (soap/code/verify); judge (coding/ambient). settings.py holds defaults, data/prompts.json persists edits, each Save pushes versions to LangFuse
- [x] `EvalRunner` runs golden cases through the orchestrator; Layer 1 adds a router-accuracy check
- [x] All new modules have unit tests with mocked external calls (router, both agents, judge, orchestrator, tracer)
- [x] LangGraph runs synchronously (`.invoke()`); no async/streaming introduced
- [x] App still runs with: `uv run streamlit run src/medic_agent/ui/app.py`

---

## Key Decisions Log

| Date | Decision | Reason |
|---|---|---|
| 2026-05-09 | Use LiteLLM over raw Anthropic SDK | Future multi-provider flexibility |
| 2026-05-09 | Use Streamlit over Flask | Zero frontend overhead for V0 |
| 2026-05-09 | FastAPI approved | Approved for HTTP API layer when needed; not Flask |
| 2026-05-09 | Use uv over pip | Modern standard, faster, better env isolation |
| 2026-05-09 | No LangChain | Adds abstraction before fundamentals are understood |
| 2026-05-09 | Prompt caching enabled | cache_control on system prompt + future doc context; isolated to _build_messages() in llm/client.py |
| 2026-05-09 | SapBERT over OpenAI embeddings | Medical ontology grounding required for ICD/SNOMED/CPT; SapBERT trained on UMLS |
| 2026-05-09 | HuggingFace Inference API over local | No GPU/memory overhead; free tier sufficient for dev |
| 2026-05-09 | ChromaDB persistent over in-memory | Single user; persist once, query always; stored at data/chroma/ |
| 2026-05-09 | PDF + TXT for V1; FHIR JSON deferred | FHIR needs special extraction logic; keep V1 scope tight |
| 2026-05-09 | Two system prompts, one per use case | Coding and ambient note taking have fundamentally different output requirements |
| 2026-05-09 | Specialized to coding + ambient note taking | Highest clinical value; requires deep medical ontology grounding |
| 2026-05-10 | LangFuse over Arize Phoenix | Cloud acceptable (no real patient data); LangFuse adds prompt versioning + integrated eval |
| 2026-05-10 | Three-layer eval: deterministic + RAGAS + LLM-as-judge | Codes are right/wrong (deterministic); RAG quality needs RAGAS; overall quality needs LLM judge |
| 2026-05-10 | Judge model: Claude Sonnet judges Claude Haiku | Judge must be smarter than the system being evaluated |
| 2026-05-10 | Three-tab Streamlit UI | Obs and Eval surfaced in-app; no external dashboard required for routine use |
| 2026-05-10 | EvalRunner in src/ not only in tests/ | UI (Tab 4) and pytest must share the same eval logic; runner.py is the single entry point |
| 2026-05-10 | Four-tab UI; KB & Prompts tab added | Separates configuration (Tab 2) from operation (Tab 1); doc upload moved out of sidebar |
| 2026-05-10 | Prompts editable at runtime via Tab 2 | Two-layer persistence: data/prompts.json (local) + LangFuse versions (cloud history) |
| 2026-05-10 | data/prompts.json gitignored | User prompt edits are personal config, not source code |
| 2026-06-16 | V2 = Multi-Agent Orchestration (was Multi-Persona) | Higher value than persona-switching; each use case becomes a real multi-step agent behind an orchestrator |
| 2026-06-16 | LangGraph approved for orchestration only | User-approved reversal of the no-LangChain-ecosystem rule, scoped to agents/; RAG pipeline stays hand-built |
| 2026-06-16 | Hybrid router (heuristic → LLM fallback) | Free/instant on clear signals; Haiku fallback only when ambiguous |
| 2026-06-16 | Auto routing + manual override | System decides the use case, but the user keeps an escape hatch |
| 2026-06-16 | Full multi-step agent subgraphs (~4 nodes each) | Genuinely agentic; "as needed" to bound latency/cost |
| 2026-06-16 | Online LLM-as-judge per query (Sonnet, default on, toggleable) | Judge metrics visible in observability traces, not only the Eval tab |
| 2026-06-16 | Manual nested LangFuse spans (not LangChain callback handler) | Keeps V1 populate-then-emit pattern; captures custom router/judge fields |
| 2026-06-16 | Shared judge/rubric between online judge and eval runner | Single source of truth; no duplication |
| 2026-06-16 | All prompts editable in Tab 2, grouped by agent | Full transparency and tuning: router, coding (extract/code/verify), ambient (soap/code/verify), and judge (coding/ambient) prompts all surfaced; settings.py holds defaults, data/prompts.json persists edits |
