# medic-agent — AI Assistant Context

This file is read by Claude Code at the start of every session.
It defines the rules, constraints, and conventions for this project.
NEVER deviate from these instructions without explicit user approval.

---

## Project Overview

**medic-agent** is a healthcare-focused AI agent application.
It allows a user to submit natural language queries and receive answers from an LLM.
The app is designed for healthcare contexts: clinical, administrative (RCM, scheduling, claims), and patient-facing use cases.

**Current phase:** V0 — single-user, local web UI, Claude models only.

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
| Version Control | Git | Standard |

Do NOT suggest or introduce Flask, LangChain, or other frameworks unless the user explicitly approves it.

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
├── tests/
│   ├── test_llm_client.py
│   └── test_api.py
├── .env                       ← secrets (never commit this)
├── .env.example               ← template for .env (safe to commit)
├── .gitignore
└── pyproject.toml             ← project metadata and dependencies
```

Each substantial new feature gets its own subfolder under `src/medic_agent/`.
Example: a RAG feature in V1 would go in `src/medic_agent/rag/`.

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

- Do NOT use LangChain or LlamaIndex in V0. These add complexity before we understand the fundamentals.
- Do NOT add a database in V0. No SQLite, no Postgres, nothing persistent yet.
- Do NOT build authentication in V0. Single user, local only.
- Do NOT deploy to cloud in V0. Local MacBook only.
- Do NOT add async/streaming in V0. Keep it synchronous for now.
- Do NOT create new files unless absolutely necessary.

---

## Current V0 Acceptance Criteria

- [ ] User can type a healthcare query in the web UI
- [ ] User can select a Claude model (haiku / sonnet / opus)
- [ ] App sends query to selected model via LiteLLM
- [ ] Response is displayed in the web UI
- [ ] API key is loaded from .env, never hardcoded
- [ ] App runs locally with a single command: `uv run streamlit run src/medic_agent/ui/app.py`

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
