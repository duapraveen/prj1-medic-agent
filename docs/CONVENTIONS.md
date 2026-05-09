# Coding Conventions — medic-agent

**Version:** 0.1  
**Last Updated:** 2026-05-09  

These conventions apply to all code in this project.
When using AI-assisted coding, paste this file as context or reference CLAUDE.md.

---

## 1. Python Style

- Follow **PEP 8** as the baseline
- Line length: **88 characters** (Black formatter default)
- Use **Black** for auto-formatting (run before committing)
- Use **type hints** on all function signatures — no exceptions

```python
# Good
def ask(model_id: str, system_prompt: str, user_query: str) -> str:
    ...

# Bad — missing type hints
def ask(model_id, system_prompt, user_query):
    ...
```

---

## 2. Naming

| Thing | Convention | Example |
|---|---|---|
| Files | `snake_case.py` | `llm_client.py` |
| Functions | `snake_case` | `get_response()` |
| Variables | `snake_case` | `user_query` |
| Classes | `PascalCase` | `LLMClient` |
| Constants | `UPPER_SNAKE_CASE` | `DEFAULT_MODEL` |
| Private helpers | `_snake_case` | `_validate_query()` |

---

## 3. Imports

Order: stdlib → third-party → local. Separated by blank lines.

```python
# Good
import os
from pathlib import Path

import streamlit as st
from litellm import completion

from medic_agent.config.settings import AVAILABLE_MODELS
```

---

## 4. Functions

- One responsibility per function
- Max 60 lines per function. If longer, split it. Exceptions require a comment explaining why.
- No deeply nested logic (max 2-3 levels of indentation)
- Return early to avoid deep nesting

```python
# Good — return early
def ask(model_id: str, user_query: str) -> str:
    if not user_query.strip():
        raise ValueError("Query cannot be empty")
    # ... rest of logic

# Bad — nested
def ask(model_id: str, user_query: str) -> str:
    if user_query.strip():
        if model_id in AVAILABLE_MODELS:
            # ... deeply nested
```

---

## 5. Error Handling

- **Validate at boundaries** (user input, API calls) — not deep in internal logic
- **Fail loudly at startup** for missing config — don't fail silently at runtime
- **Catch specific exceptions**, not bare `except:`
- **Show user-friendly messages** in the UI, log the technical detail separately

```python
# Good
try:
    response = completion(model=model_id, messages=messages)
except AuthenticationError:
    raise RuntimeError("Invalid API key. Check your .env file.")
except RateLimitError:
    raise RuntimeError("Rate limit hit. Wait a moment and try again.")

# Bad
try:
    response = completion(...)
except Exception:
    pass  # silent failure
```

---

## 6. Configuration and Secrets

- **ALL secrets** come from environment variables, loaded via `python-dotenv`
- **Never hardcode** API keys, model names as magic strings, or URLs
- All configurable values live in `config.py`
- Reference constants from `config.py`, not inline strings

```python
# Good
from medic_agent.config.settings import DEFAULT_MODEL, SYSTEM_PROMPT
response = ask(model_id=DEFAULT_MODEL, system_prompt=SYSTEM_PROMPT, ...)

# Bad
response = ask(model_id="claude-sonnet-4-6", system_prompt="You are a ...", ...)
```

---

## 7. Comments

- **Don't comment what the code does** — write readable code instead
- **Do comment why** when the reason isn't obvious
- No TODO comments in committed code — use the DEVELOPMENT_PLAN.md instead

```python
# Good — explains WHY
# LiteLLM requires the "anthropic/" prefix for Claude models when using the raw model ID
model_id = f"anthropic/{model_id}"

# Bad — narrates the obvious
# Call the completion function with the model and messages
response = completion(model=model_id, messages=messages)
```

---

## 8. Git Commits

Use the following format for commit messages:

```
<type>: <short description>

Types:
  feat     — new feature
  fix      — bug fix
  chore    — setup, tooling, dependencies
  docs     — documentation only
  refactor — code restructure, no behavior change
  test     — adding or updating tests
```

Examples:
```
feat: add model selector dropdown to UI
fix: handle API rate limit error gracefully
chore: add uv lockfile and pyproject.toml
docs: update architecture diagram for V1 RAG layer
```

---

## 9. File Organization

All source code lives under `src/medic_agent/`. Each substantial feature gets its own subfolder.

| File | Responsibility | Must NOT contain |
|---|---|---|
| `ui/app.py` | UI rendering and user interaction | Business logic, direct LLM calls |
| `llm/client.py` | LLM interaction only | UI code, config loading |
| `config/settings.py` | Constants and env var loading | Logic, I/O |
| `api/routes.py` | HTTP request/response handling | Business logic (delegate to llm/) |

If you find yourself putting business logic in `ui/app.py`, stop and extract it to the appropriate subfolder.

---

## 10. Testing

- Tests live in `tests/` directory
- Tests live in `tests/` at the project root (mirrors `src/medic_agent/` structure)
- Test file names mirror source file names: `tests/llm/test_client.py` tests `src/medic_agent/llm/client.py`
- Use `pytest`
- V0 minimum: at least one test per public function in `llm/client.py`
- Mock external API calls in tests — never make real API calls in tests

```python
# Good — mock the API call
def test_ask_returns_string(mocker):
    mocker.patch("medic_agent.llm.client.completion", return_value=mock_response)
    result = ask("claude-sonnet-4-6", "You are helpful", "What is aspirin?")
    assert isinstance(result, str)
    assert len(result) > 0
```

---

## Revision History

| Version | Date | Change |
|---|---|---|
| 0.1 | 2026-05-09 | Initial scaffold |
| 0.2 | 2026-05-09 | Updated paths to src/medic_agent/ structure; 60-line function limit; FastAPI noted |
