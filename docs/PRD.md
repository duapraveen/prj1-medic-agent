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

### Layout
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
│                             │  SOURCES                      │
│                             │  • encounter_note.pdf, p.2    │
│                             │  • guidelines_2024.pdf, p.14  │
└─────────────────────────────────────────────────────────────┘
```

### Use Case Selector Behavior
- **Medical Coding**: default query pre-filled as "What are the appropriate codes for this encounter?" — user can override
- **Ambient Note Taking**: input label changes to "Paste encounter transcript here"
- System prompt switches automatically based on selection

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
