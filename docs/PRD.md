# Product Requirements Document — medic-agent

**Version:** 0.2  
**Status:** Active  
**Last Updated:** 2026-05-09  
**Author:** [YOUR NAME]  

---

## 1. Problem Statement

Medical coding and clinical documentation are two of the most labor-intensive, error-prone workflows in US healthcare. Coding errors lead to claim denials, revenue loss, and compliance risk. Clinical documentation burden is a leading driver of physician burnout.

medic-agent addresses both:
1. **Medical Coding**: Given encounter documents, produce accurate ICD-10-CM, CPT, and HCPCS codes grounded in the actual documentation — with citations and gap identification.
2. **Ambient Note Taking**: Given a physician-patient conversation transcript, produce a structured SOAP note and an accurate billable code list — eliminating manual transcription and post-visit coding work.

Both use cases require deep medical ontology grounding (ICD-10, SNOMED CT, CPT, LOINC) that generic AI assistants cannot reliably provide.

---

## 2. Target Users

### V0 (Complete)
- **Primary user:** Developer/PM — testing and validating the core loop

### V1 (Current)
| Persona | Role | Use Case in this App |
|---|---|---|
| Medical Coder | Coding specialist, HIM professional | Upload encounter docs → get code suggestions with citations |
| Physician / APP | Doctor, PA, NP | Paste encounter transcript → get SOAP note + codes |
| Clinical Documentation Specialist | CDI professional | Review and validate AI-generated SOAP and code suggestions |

### V2 and Beyond
| Persona | Use Case |
|---|---|
| RCM Manager | Denial analysis, prior auth prep |
| Patient | Symptom triage, visit prep (separate system prompt) |

---

## 3. Use Cases

### Use Case 1 — Medical Coding

**Trigger:** User uploads one or more encounter documents (clinical notes, operative reports, discharge summaries, lab results).

**Flow:**
```
Upload encounter documents
        ↓
Documents ingested → chunked → embedded (SapBERT) → stored in ChromaDB
        ↓
User types coding query (or uses default: "What are the appropriate codes for this encounter?")
        ↓
Relevant chunks retrieved from ChromaDB
        ↓
LLM (coding system prompt) produces:
  - ICD-10-CM diagnosis codes (with sequencing)
  - CPT procedure codes
  - HCPCS codes (if applicable)
  - Citation: which document/section supports each code
  - Documentation gaps: what's missing that would affect coding
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

**Trigger:** User pastes a physician-patient conversation transcript into the input field.

**Flow:**
```
User pastes transcript
        ↓
(Optional) Relevant coding guidelines retrieved from ChromaDB
        ↓
LLM (ambient system prompt) produces:
  - Structured SOAP note
  - ICD-10-CM codes for all diagnoses in Assessment
  - CPT codes for procedures mentioned
  - Documentation flags (missing elements for complete SOAP)
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

### V1 — In Scope
- **Use case selector**: Coding vs. Ambient Note Taking
- **Document ingestion**: PDF and plain text (.txt)
- **Persistent vector store**: ChromaDB (upload once, use always)
- **Medical ontology-aware embeddings**: SapBERT via HuggingFace Inference API
- **Two specialized system prompts**: one per use case
- **Retrieval-grounded responses**: top-5 relevant chunks injected as context
- **Source citation**: every code linked to supporting documentation
- **Documentation gap identification**: flags missing info

### V1 — Out of Scope
- FHIR JSON ingestion (V2)
- Real-time audio transcription (V2 — user pastes transcript manually)
- Payer-specific rule engines (V2)
- Multi-turn conversation / memory (V2)
- Claim submission or EHR integration (V3)
- User authentication (V3)
- Cloud deployment (V3)

---

## 5. User Experience — V1

### App Layout — Three Tabs

```
┌──────────────────────────────────────────────────────────────────┐
│  🤖 Agent  │  📊 Observability  │  🧪 Evaluation                │
├──────────────────────────────────────────────────────────────────┤
```

---

### Tab 1 — Agent

```
┌─────────────────────────────────────────────────────────────┐
│  SIDEBAR                    │  MAIN AREA                    │
│  ─────────────              │  ────────────────────         │
│  Use Case:                  │  [Model selector]             │
│  ○ Medical Coding           │                               │
│  ○ Ambient Note Taking      │  [Query / Transcript input]   │
│                             │                               │
│  Documents:                 │  [Submit button]              │
│  [Upload PDF or TXT]        │                               │
│  ─────────────              │  ─────────────────────────    │
│  • encounter_note.pdf  [x]  │  RESPONSE                     │
│  • guidelines_2024.pdf [x]  │  (SOAP note or code list)     │
│                             │                               │
│                             │  SOURCES USED                 │
│                             │  • encounter_note.pdf §3      │
│                             │  • guidelines_2024.pdf §7     │
└─────────────────────────────────────────────────────────────┘
```

**Use Case Selector Behavior:**
- **Medical Coding**: query pre-filled with "What are the appropriate codes for this encounter?"
- **Ambient Note Taking**: input label changes to "Paste encounter transcript here"
- System prompt switches automatically; prompt version logged to LangFuse

---

### Tab 2 — Observability

**Purpose:** Visibility into every query session — inputs, outputs, retrieved context, latency, cost. Auditability record.

**How to use:** This tab is passive — just open it. Every query you run in Tab 1 automatically appears here. No user action required.

```
┌──────────────────────────────────────────────────────────────┐
│  SUMMARY BAR                                                 │
│  Sessions: 42  │  Avg latency: 3.2s  │  Avg cost: $0.003    │
│  Errors: 1     │  Most used: Coding  │  Tokens today: 84k   │
├──────────────────────────────────────────────────────────────┤
│  FILTERS                                                     │
│  Use case: [All ▼]   Date: [Last 7 days ▼]                  │
├──────────────────────────────────────────────────────────────┤
│  SESSION LOG TABLE                                           │
│  Time       │ Use Case │ Model  │ Query (preview) │ ms │ err │
│  05-10 14:32│ Coding   │ Haiku  │ "What codes..." │1842│     │
│  05-10 14:28│ Ambient  │ Sonnet │ "Doctor: Good.."│3210│     │
│  ▼ [expand row for full detail]                              │
├──────────────────────────────────────────────────────────────┤
│  [📤 Export CSV]          [🔗 Open in LangFuse →]           │
└──────────────────────────────────────────────────────────────┘
```

**Session detail (expandable row):**
- Full query text
- Full response text
- Retrieved chunks with similarity scores
- System prompt version used
- Token breakdown (prompt / completion / cached)

**"Open in LangFuse" button** → opens the user's LangFuse project in a browser tab, where waterfall traces and full token-level inspection are available.

---

### Tab 3 — Evaluation

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

### V1 Definition of Done
- [ ] Use case selector (Coding / Ambient) controls system prompt
- [ ] PDF and TXT documents ingested, embedded with SapBERT, stored in ChromaDB
- [ ] Documents persist across app restarts
- [ ] Duplicate upload detection works
- [ ] Coding response: includes ICD-10-CM + CPT codes with document citations
- [ ] Ambient response: includes full SOAP note + code list + documentation flags
- [ ] Response cites source document and chunk for each code
- [ ] All new modules have unit tests with mocked external calls

---

## 7. Non-Goals (Permanent)

- NOT a claim submission or EHR system — it is a coding and documentation *assistant*
- NOT a replacement for certified coder review — outputs must be reviewed before submission
- NOT HIPAA-compliant — do not enter real patient data until compliance review is done
- NOT a general-purpose chatbot — every response is grounded in medical coding standards

---

## 8. Open Questions

| # | Question | Owner | Status |
|---|---|---|---|
| 1 | Should coding output follow a fixed structured format or free-form markdown? | [YOU] | Open |
| 2 | Should the system surface confidence scores per code? | [YOU] | Open |
| 3 | Which payer guidelines should be prioritized as reference documents in V1? | [YOU] | Open |
| 4 | Should SOAP note sections be collapsible in the UI? | [YOU] | Open |

---

## 9. Revision History

| Version | Date | Change |
|---|---|---|
| 0.1 | 2026-05-09 | Initial scaffold |
| 0.2 | 2026-05-09 | Specialized to medical coding + ambient note taking use cases |
| 0.3 | 2026-05-10 | Three-tab UX: Agent, Observability, Evaluation |
