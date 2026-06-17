# Product Requirements Document — medic-agent

**Version:** 0.5  
**Status:** Active  
**Last Updated:** 2026-06-16  

---

## 1. Problem Statement

Medical coding and clinical documentation are two of the most labor-intensive, error-prone workflows in US healthcare. Coding errors lead to claim denials, revenue loss, and compliance risk. Clinical documentation burden is a leading driver of physician burnout.

medic-agent addresses both:
1. **Medical Coding**: Given encounter documents, produce accurate ICD-10-CM, CPT, and HCPCS codes grounded in the actual documentation — with citations and gap identification.
2. **Ambient Note Taking**: Given a physician-patient conversation transcript, produce a structured SOAP note and an accurate billable code list — eliminating manual transcription and post-visit coding work.

Both use cases require deep medical ontology grounding (ICD-10, SNOMED CT, CPT, LOINC) that generic AI assistants cannot reliably provide.

---

## 2. Target Users

### V0–V2 (Complete)
- **Primary user:** Developer/PM — building, testing, and validating the agent

### V1–V3 (Current — local single user)
| Persona | Role | Use Case in this App |
|---|---|---|
| Medical Coder | Coding specialist, HIM professional | Upload encounter docs → get ICD-10/CPT suggestions with citations and gap flags |
| Physician / APP | Doctor, PA, NP | Paste encounter transcript → get SOAP note + billing codes |
| Clinical Documentation Specialist | CDI professional | Review and validate AI-generated SOAP notes and code assignments |

### Future (V4+)
| Persona | Use Case |
|---|---|
| RCM Manager | Denial analysis, prior auth prep |
| Patient | Symptom triage, visit prep (separate system prompt) |

---

## 3. Use Cases

### Use Case 1 — Medical Coding

**Trigger:** User uploads one or more encounter documents (clinical notes, operative reports, discharge summaries, lab results), then pastes a query into the Agent tab.

**Flow (V2 + V3):**
```
Upload encounter documents (Tab 2)
        ↓
Documents ingested → chunked → embedded (SapBERT) → stored in ChromaDB
                                                    → entities extracted (Haiku LLM)
                                                    → stored in Kuzu knowledge graph (V3)
        ↓
User pastes query → clicks Submit
        ↓
Orchestrator router (heuristic or LLM) determines: Medical Coding agent
UI shows: "Medical Coding · heuristic · 0.95 · Detected ICD/CPT coding keywords"
        ↓
Coding agent runs multi-step:
  extract → retrieve (vector + graph) → code → verify
        ↓
LLM (coding prompt) produces:
  - ICD-10-CM diagnosis codes (with sequencing)
  - CPT procedure codes
  - HCPCS codes (if applicable)
  - Citation: which document/section supports each code
  - Documentation gaps: what's missing that would affect coding
        ↓
Online LLM-as-judge (Sonnet) scores the response (code_accuracy, completeness, etc.)
```

**Output format:**
```
DIAGNOSIS CODES (ICD-10-CM)
  I10    — Essential hypertension
           Source: Progress Note, paragraph 2
  E11.65 — Type 2 diabetes with hyperglycemia
           Source: Lab Results, HbA1c 9.2%

PROCEDURE CODES (CPT)
  99214  — Office visit, moderate complexity
           Source: Progress Note

DOCUMENTATION GAPS
  - Duration of hypertension not documented (affects staging)
```

---

### Use Case 2 — Ambient Note Taking

**Trigger:** User pastes a physician-patient conversation transcript into the Agent tab.

**Flow (V2 + V3):**
```
User pastes transcript → clicks Submit
        ↓
Orchestrator router detects speaker-turn markers (Doctor:/Patient:)
→ routes to: Ambient Note Taking agent
UI shows: "Ambient Note Taking · heuristic · 0.90 · Detected multi-turn dialogue"
        ↓
Ambient agent runs multi-step:
  retrieve (vector + graph entity context) → soap → code → verify
        ↓
LLM (ambient prompt) produces:
  - Structured SOAP note (S/O/A/P)
  - ICD-10-CM codes for all diagnoses in Assessment
  - CPT E&M code based on documented complexity
  - Documentation flags (missing elements for complete SOAP)
        ↓
Online LLM-as-judge (Sonnet) scores the response (soap_completeness, faithfulness, etc.)
```

**Output format:**
```
SOAP NOTE
─────────────────────────────────────
S — SUBJECTIVE
  Chief Complaint: Fatigue and increased thirst x 3 weeks
  HPI: 54-year-old male presenting with...
  ...

O — OBJECTIVE
  Vitals: BP 142/88, HR 78, Wt 210 lbs
  Exam: ...

A — ASSESSMENT
  1. Type 2 diabetes mellitus with hyperglycemia (E11.65)
  2. Essential hypertension (I10)

P — PLAN
  1. Metformin 1000mg BID, continue
  2. Increase lisinopril to 10mg daily
  3. Repeat HbA1c in 3 months
  4. Follow up in 6 weeks

─────────────────────────────────────
BILLING CODES
  E11.65 — T2DM with hyperglycemia
  I10    — Essential hypertension
  99214  — Office visit E&M (moderate complexity)

DOCUMENTATION FLAGS
  - ROS not explicitly documented in transcript
```

---

## 4. Scope

### Completed (V0–V2)
- **Core LLM loop**: query → Claude → response; model selector (Haiku / Sonnet / Opus)
- **Document ingestion**: PDF and plain text (.txt); SapBERT embeddings; ChromaDB vector store
- **Persistent RAG**: upload once, retrieve always; duplicate detection; delete support
- **Multi-agent orchestration**: LangGraph orchestrator; hybrid router (heuristic → LLM fallback); manual override
- **Medical Coding agent**: extract → retrieve → code → verify (4 steps, ICD-10-CM + CPT + gaps)
- **Ambient Note Taking agent**: retrieve → soap → code → verify (4 steps, SOAP + billing codes + flags)
- **Online LLM-as-judge**: Sonnet scores every response on use-case-specific rubrics; toggleable
- **Observability**: LangFuse nested multi-agent traces; local JSONL fallback; Tab 3 session log
- **Evaluation framework**: 3-layer (deterministic + RAGAS + LLM-as-judge); golden dataset; baseline regression detection
- **Prompt management**: all 9 agent prompts editable in Tab 2; versioned in LangFuse; persisted in `data/prompts.json`

### V3 — In Scope (Current)
- **Knowledge graph (Kuzu)**: embedded graph DB alongside ChromaDB; no server required
- **Entity extraction at ingest**: Haiku LLM extracts `{text, entity_type}` from each chunk; stored as nodes with `APPEARS_IN` edges to documents
- **Hybrid retrieval**: both agents merge vector chunks + graph entity context per query
- **Cross-document entity linking**: same entity text across documents shares one graph node automatically
- **Graph inspection**: Kuzu Explorer (`uvx kuzu-explorer data/kuzu/`) for local visual traversal

### Permanent Out of Scope
- FHIR JSON ingestion — requires specialized extraction logic; deferred
- Real-time audio transcription — user pastes transcript manually
- Payer-specific rule engines — too narrow; prompt engineering addresses most cases
- Multi-turn conversation / memory — single-query model; history is in uploaded docs
- Claim submission or EHR integration — assistant only; submission is a separate workflow
- User authentication — single user, local only
- Cloud deployment — local MacBook only; no real patient data until HIPAA review

---

## 5. User Experience — V2 (Current)

### App Layout — Four Tabs

```
┌──────────────────────────────────────────────────────────────────────────┐
│  🤖 Agent  │  📚 Knowledge Base & Prompts  │  📊 Observability  │  🧪 Evaluation  │
├──────────────────────────────────────────────────────────────────────────┤
```

---

### Tab 1 — Agent

The user no longer selects a use case. They paste any clinical text and click Submit — the orchestrator determines the agent automatically. A manual override is available in the sidebar for edge cases.

```
┌─────────────────────────────────────────────────────────────┐
│  SIDEBAR                    │  MAIN AREA                    │
│  ─────────────              │  ────────────────────         │
│  Routing Mode:              │  [Model selector dropdown]    │
│  ○ Auto (recommended)       │                               │
│  ○ Medical Coding           │  [Query / Transcript input]   │
│  ○ Ambient Note Taking      │  (any clinical text)          │
│                             │                               │
│  ☑ Online Judge (Sonnet)    │  [Submit]                     │
│                             │                               │
│  Knowledge Base:            │  ──────────────────────────   │
│  📄 3 documents (42 chunks) │  ROUTING DECISION             │
│  → Manage in Tab 2          │  Medical Coding · heuristic   │
│                             │  Confidence: 0.95             │
│                             │  "Detected ICD/CPT keywords"  │
│                             │                               │
│                             │  ──────────────────────────   │
│                             │  RESPONSE                     │
│                             │  (code list or SOAP note)     │
│                             │                               │
│                             │  SOURCES USED                 │
│                             │  • encounter_note.pdf §3      │
│                             │  • [knowledge-graph] entities │
│                             │                               │
│                             │  JUDGE SCORES                 │
│                             │  code_accuracy: 4.5/5         │
│                             │  completeness: 4.0/5  ...     │
└─────────────────────────────────────────────────────────────┘
```

**Routing Behavior:**
- **Auto**: orchestrator classifies the query using heuristics (speaker-turn markers for ambient, coding keywords for coding) and falls back to a Haiku LLM call when ambiguous
- **Manual override**: forces a specific agent; routing panel shows `method: manual`
- Routing panel always visible after Submit: agent name, method, confidence score, one-line reasoning

---

### Tab 2 — Knowledge Base & Prompts

**Purpose:** All configuration in one place — uploaded documents and all 9 agent prompts (router + 3 coding + 3 ambient + 2 judge), each editable independently.

```
┌──────────────────────────────────────────────────────────────┐
│  KNOWLEDGE BASE                                              │
│  [📎 Upload PDF or TXT file]                                 │
│  Document              Chunks   Uploaded        Action       │
│  encounter_note.pdf      18     2026-06-16       [🗑 Delete] │
│  guidelines_2024.pdf     31     2026-06-15       [🗑 Delete] │
│  Total: 2 documents, 49 chunks                               │
├──────────────────────────────────────────────────────────────┤
│  AGENT PROMPTS              Active version: v5               │
│                                                              │
│  ▸ ROUTER                                                    │
│    [editable text area]                                      │
│                                                              │
│  ▸ MEDICAL CODING AGENT                                      │
│    Extract prompt:  [editable text area]                     │
│    Code prompt:     [editable text area]                     │
│    Verify prompt:   [editable text area]                     │
│                                                              │
│  ▸ AMBIENT NOTE TAKING AGENT                                 │
│    SOAP prompt:     [editable text area]                     │
│    Code prompt:     [editable text area]                     │
│    Verify prompt:   [editable text area]                     │
│                                                              │
│  ▸ JUDGE                                                     │
│    Coding rubric:   [editable text area]                     │
│    Ambient rubric:  [editable text area]                     │
│                                                              │
│  [💾 Save Prompts]          [↩ Reset to Defaults]           │
└──────────────────────────────────────────────────────────────┘
```

**Prompt Save Behavior:**
- All 9 prompts saved together to `data/prompts.json` (persists across restarts)
- Each save pushes a new version to LangFuse; all subsequent traces link to the new version
- "Reset to Defaults" restores all prompts from `config/settings.py` defaults (does not auto-save)

---

### Tab 3 — Observability

**Purpose:** Full visibility into every query — routing decision, agent steps, judge scores, latency, cost. Auditability record.

**How to use:** Passive — every query in Tab 1 appears here automatically. No user action required.

```
┌──────────────────────────────────────────────────────────────┐
│  SUMMARY BAR                                                 │
│  Sessions: 42  │  Avg latency: 8.2s  │  Errors: 1           │
│  Coding: 28    │  Ambient: 14        │  Tokens today: 210k   │
├──────────────────────────────────────────────────────────────┤
│  SESSION LOG TABLE                                           │
│  Time    │ Agent   │ Route  │ Judge │ Query preview   │ ms   │
│  06-16.. │ Coding  │ heur.  │  4.3  │ "code this..."  │8842  │
│  06-16.. │ Ambient │ llm    │  3.9  │ "Doctor: hi..." │12310 │
│  ▼ [expand row for full detail]                              │
├──────────────────────────────────────────────────────────────┤
│  [📤 Export CSV]          [🔗 Open in LangFuse →]           │
└──────────────────────────────────────────────────────────────┘
```

**Session detail (expandable row):**
- Full query text and full response
- Route decision: method (heuristic/llm/manual), confidence, reasoning
- Per-step breakdown: each agent step with name, latency, token usage
- Retrieved chunks (vector + graph) with source attribution
- Judge scores per dimension + overall
- Token breakdown (prompt / completion / cached)

**"Open in LangFuse"** → nested waterfall trace showing router → agent steps → judge spans with timing and token data.

---

### Tab 4 — Evaluation

**Purpose:** Assess quality of the AI pipeline. Run after any significant change (prompt, model, chunking, retrieval k).

**How to use:** Select which layers to run → click "Run Evaluation" → wait for results → optionally set as baseline.

```
┌──────────────────────────────────────────────────────────────┐
│  GOLDEN DATASET  (5 cases: 3 coding, 2 SOAP)                 │
│  ☑ coding_001  T2DM + HTN + CKD quarterly visit              │
│  ☑ coding_002  COPD acute exacerbation                       │
│  ☑ coding_003  Acute MI sparse documentation                 │
│  ☑ soap_001    Hypertension follow-up transcript             │
│  ☑ soap_002    Costochondritis — fabrication stress test     │
├──────────────────────────────────────────────────────────────┤
│  LAYERS TO RUN                                               │
│  ☑ Layer 1: Deterministic  (~2s, free)                       │
│  ☑ Layer 2: RAGAS           (~3 min, small LLM cost)         │
│  ☑ Layer 3: LLM-as-Judge    (~5 min, Sonnet cost ~$0.05)     │
│                                                              │
│  [▶ Run Evaluation]  [📌 Set as Baseline]                    │
├──────────────────────────────────────────────────────────────┤
│  RESULTS                             vs Baseline             │
│  Case        │ L1  │ Faithful │ Judge │ Δ Judge             │
│  coding_001  │ ✅  │  0.91    │  4.2  │ +0.3 ✅             │
│  coding_002  │ ✅  │  0.87    │  3.8  │ -0.2 ⚠️             │
│  coding_003  │ ✅  │  0.93    │  4.5  │ +0.1 ✅             │
│  soap_001    │ ✅  │  0.89    │  4.1  │  0.0 —              │
│  soap_002    │ ❌  │  0.72    │  3.3  │ -0.5 🔴             │
├──────────────────────────────────────────────────────────────┤
│  SCORE HISTORY (chart — scores over time)                    │
│  [📤 Export Results]   [🔗 View in LangFuse →]              │
└──────────────────────────────────────────────────────────────┘
```

**When to run evaluation:**
- After changing a system prompt → "did this improve coding accuracy?"
- After switching models (Haiku → Sonnet) → "is the quality gain worth the cost?"
- After changing chunk size or retrieval k → "did retrieval quality improve?"
- As a pre-commit regression check before significant changes

---

## 6. Success Criteria

### V0 — Complete ✅ (2026-05-09)
- [x] Query → LLM → response loop working
- [x] Model selector (Haiku / Sonnet / Opus)
- [x] API key management via .env
- [x] 10 unit tests passing

### V1 — Complete ✅ (2026-05-10)
- [x] PDF and TXT documents ingested, embedded (SapBERT), stored in ChromaDB persistently
- [x] Duplicate upload detection; delete support
- [x] Coding response: ICD-10-CM + CPT codes with document citations and gap flags
- [x] Ambient response: full SOAP note + code list + documentation flags
- [x] Four-tab UI: Agent, Knowledge Base & Prompts, Observability, Evaluation
- [x] LangFuse tracing per query; local JSONL fallback
- [x] Three-layer eval framework (deterministic + RAGAS + LLM-as-judge); golden dataset; baseline

### V2 — Complete ✅ (2026-06-16)
- [x] Orchestrator routes queries automatically; no use-case selection by user
- [x] Hybrid router (heuristic → Haiku LLM fallback); manual override available
- [x] Coding agent: extract → retrieve → code → verify (4-step multi-agent)
- [x] Ambient agent: retrieve → soap → code → verify (4-step multi-agent)
- [x] Online LLM-as-judge (Sonnet) scores every response; sidebar toggle
- [x] Routing panel displayed after Submit: agent, method, confidence, reasoning
- [x] All 9 prompts editable in Tab 2 grouped by agent; versioned in LangFuse
- [x] Nested LangFuse spans: router → agent steps → judge; judge score attached to trace
- [x] Eval runs through orchestrator; Layer 1 adds router-accuracy check

### V3 Definition of Done (Current)
- [ ] Kuzu knowledge graph indexed alongside ChromaDB at ingest time
- [ ] Entity extraction (Haiku) runs per chunk at upload; entities stored as nodes
- [ ] Same entity across documents shares one graph node (cross-document linking)
- [ ] Both agents use hybrid retrieval: vector chunks + graph entity context
- [ ] Deleting a document removes its graph nodes and prunes orphaned entities
- [ ] All new modules have unit tests; existing tests unaffected
- [ ] Kuzu Explorer works locally for graph inspection after a document upload

---

## 7. Non-Goals (Permanent)

- NOT a claim submission or EHR system — it is a coding and documentation *assistant*
- NOT a replacement for certified coder review — outputs must be reviewed before submission
- NOT HIPAA-compliant — do not enter real patient data until compliance review is done
- NOT a general-purpose chatbot — every response is grounded in medical coding standards

---

## 8. Open Questions

| # | Question | Status | Resolution |
|---|---|---|---|
| 1 | Should coding output follow a fixed structured format or free-form markdown? | Resolved | Fixed format: DIAGNOSIS CODES / PROCEDURE CODES / DOCUMENTATION GAPS sections |
| 2 | Should the system surface confidence scores per code? | Resolved | No per-code confidence. LLM-as-judge provides overall quality scores (code_accuracy, completeness, etc.) instead |
| 3 | Which payer guidelines should be prioritized as reference documents? | Open | User uploads their own reference PDFs; no bundled payer guidelines included |
| 4 | Should SOAP note sections be collapsible in the UI? | Resolved | No — displayed as inline text in the response area; no special rendering |
| 5 | Should the routing decision be overridable per-query or only via sidebar? | Resolved | Sidebar only (routing mode selectbox); no inline override per query |
| 6 | Should entity extraction failures block document ingest? | Resolved | No — extraction returns `[]` on parse failure; ChromaDB write always succeeds |
| 7 | Should RELATED_TO edges between entities be extracted in V3? | Resolved | No — deferred to V4; V3 only indexes Entity-to-Document links (APPEARS_IN) |

---

## 9. Revision History

| Version | Date | Change |
|---|---|---|
| 0.1 | 2026-05-09 | Initial scaffold |
| 0.2 | 2026-05-09 | Specialized to medical coding + ambient note taking use cases |
| 0.3 | 2026-05-10 | Three-tab UX: Agent, Observability, Evaluation |
| 0.4 | 2026-05-10 | Four-tab UX: added Knowledge Base & System Prompts tab; doc upload moved from sidebar |
| 0.5 | 2026-06-16 | V2 complete: UX updated for orchestrator routing panel, all-prompts Tab 2, judge scores, V2 success criteria; V3 knowledge graph scoped; open questions resolved; scope section rewritten to reflect current state |
