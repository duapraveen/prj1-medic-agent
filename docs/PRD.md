# Product Requirements Document — medic-agent

**Version:** 0.1  
**Status:** Draft  
**Last Updated:** 2026-05-09  
**Author:** [YOUR NAME]  

---

## 1. Problem Statement

Healthcare professionals and patients struggle to get accurate, contextual answers to healthcare questions quickly. 
Generic AI assistants lack healthcare-specific grounding and cannot be trusted with sensitive clinical or administrative queries.
medic-agent provides a focused, controllable AI interface tailored to healthcare workflows with appropriate guardrails relevant for healthcare usecase.


---

## 2. Target Users

### V0 (Current)
- **Primary user:** Developer/PM (you) — testing and validating the core loop

### V1 and Beyond
| Persona | Role | Primary Use Cases |
|---|---|---|
| Clinician | Doctor, nurse, PA | Clinical decision support, drug interactions, protocol lookup |
| Administrator | Billing, scheduling, RCM staff | Claim filing, prior auth, appointment workflows |
| Patient | End patient | Symptom triage, appointment prep, understanding diagnosis |

---

## 3. User Stories

### V0 — Core Loop
- As a user, I want to type a healthcare question so that I can get an AI-generated answer.
- As a user, I want to choose which LLM model answers my question so that I can control cost vs. quality.
- As a user, I want to see the response displayed clearly in the web UI so that I can read and act on it.

### V1 — Context Layer
<!-- TODO: Add user stories for document upload and RAG queries -->
- As a user, I want to upload a clinical document (PDF, FHIR JSON, etc.) so that the agent can answer questions grounded in that document.
- As a user, I want the agent to cite which document it used so that I can verify the answer.
- [PLACEHOLDER — add more as you define V1 scope]

### V2 — Multi-Agent / Specialized Agents
<!-- TODO: Define when you're ready to think about V2 -->
[PLACEHOLDER]

---

## 4. Scope

### V0 — In Scope
- Single text input → LLM → text response loop
- Model selector (Claude Haiku / Sonnet / Opus)
- Simple Streamlit web UI
- Local execution only
- API key management via .env

### V0 — Out of Scope
- User authentication
- Conversation history / memory
- Document upload or RAG
- Database persistence
- Cloud deployment
- Streaming responses
- Multi-turn conversations

### V1 — Planned (not yet designed)
- Document ingestion (PDF, FHIR JSON, plain text)
- Vector database for semantic search (local, e.g. ChromaDB)
- RAG-grounded responses with source citation
- Conversation memory (in-session)
- [PLACEHOLDER — add more as V1 takes shape]

---

## 5. User Experience — V0

### Flow
```
[User opens browser] → [Streamlit UI loads]
        ↓
[User selects model from dropdown]
        ↓
[User types healthcare query in text box]
        ↓
[User clicks "Ask" button]
        ↓
[Loading spinner shown]
        ↓
[Response displayed below input]
```

### UI Requirements
- Clean, minimal interface — no clutter
- Model selector: dropdown with at least 3 Claude options
- Input: multi-line text area (queries can be long)
- Submit: explicit button (not auto-submit on Enter)
- Output: clearly separated response area with model name shown
- Error states: show clear message if API call fails

---

## 6. Success Criteria

### V0 Definition of Done
- [ ] App runs locally with `streamlit run app.py`
- [ ] User can submit a query and receive a response
- [ ] Model selection works (Haiku / Sonnet / Opus respond differently)
- [ ] API key never appears in code or UI
- [ ] App handles API errors gracefully (shows message, doesn't crash)
- [ ] Response time is acceptable (< 30 seconds for Sonnet)

---

## 7. Non-Goals (Permanent)

- This is NOT a general-purpose chatbot — it is healthcare-focused
- This is NOT a replacement for clinical decision-making tools — it is an assistant
- This is NOT HIPAA-compliant in V0 — do not enter real patient data

---

## 8. Open Questions

<!-- TODO: Track unresolved decisions here. Delete when resolved. -->

| # | Question | Owner | Status |
|---|---|---|---|
| 1 | Should V0 have a system prompt that primes the LLM for healthcare context? | [YOU] | Open |
| 2 | What healthcare sub-domain should be the focus for the first real test query? | [YOU] | Open |
| 3 | Should model selection persist across sessions (via config file)? | [YOU] | Open |

---

## 9. Revision History

| Version | Date | Change |
|---|---|---|
| 0.1 | 2026-05-09 | Initial scaffold |
