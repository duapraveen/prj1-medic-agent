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

> Do not begin V1 until V0 is complete and tested.

**Goal:** User can upload documents and the agent answers questions grounded in those documents.

### Planned Steps (to be detailed when V1 begins)
- [ ] Select and integrate vector database (ChromaDB — local, free)
- [ ] Build document ingestion pipeline (PDF, text, FHIR JSON)
- [ ] Implement chunking and embedding strategy
- [ ] Implement retrieval and context injection
- [ ] Update UI with document upload widget
- [ ] Display source citations in response
- [ ] [PLACEHOLDER — add more detail when V1 planning begins]

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
