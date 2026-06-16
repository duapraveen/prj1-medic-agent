# V2 Multi-Agent Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the user-selected use case with a LangGraph orchestrator that routes each query (hybrid heuristic→LLM router + manual override) to one of two multi-step agents (Medical Coding, Ambient Note Taking), scores every response with an online LLM-as-judge, and emits nested multi-agent LangFuse traces.

**Architecture:** A new `src/medic_agent/agents/` package holds a shared `AgentState`, a hybrid router, two agent subgraphs (each a compiled `StateGraph`), an online judge (sharing rubric logic with the eval runner), and an orchestrator that wires them: `START → router → {coding_agent | ambient_agent} → judge → END`. The orchestrator logs exactly one `Session` per query; agent nodes use a new non-logging `complete()` primitive. All prompts become editable in Tab 2 via a centralized `config/prompts.py`.

**Tech Stack:** Python 3.11, LangGraph, LiteLLM, ChromaDB, LangFuse, Streamlit, pytest + pytest-mock.

---

## Canonical naming (read first — used by every task)

- Use-case display names are the canonical values everywhere in `agents/`:
  - `USE_CASE_CODING = "Medical Coding"`
  - `USE_CASE_AMBIENT = "Ambient Note Taking"`
- Golden cases (`tests/eval/golden_cases.json`) use legacy keys `"medical_coding"` / `"ambient_note"`. The eval runner maps them via `EVAL_USE_CASE_MAP`.
- Router model is **always Haiku** (`ROUTER_MODEL_ID`); judge model is **always Sonnet** (`JUDGE_MODEL_ID`) regardless of the user's selected agent model.

## File structure (created/modified)

| File | Responsibility |
|---|---|
| `src/medic_agent/config/settings.py` (modify) | Add canonical names, router/judge model IDs, router prompt, agent step prompts, judge rubric prompts |
| `src/medic_agent/config/prompts.py` (create) | Centralized nested prompt load/save: `get_prompt()`, `load_all()`, `save_all()`, `PROMPT_DEFAULTS` |
| `src/medic_agent/llm/client.py` (modify) | Add `complete()` non-logging primitive |
| `src/medic_agent/agents/state.py` (create) | `AgentState` TypedDict + `llm_step()` helper |
| `src/medic_agent/agents/router.py` (create) | `RouteDecision`, `heuristic_route`, `llm_route`, `route` |
| `src/medic_agent/agents/judge.py` (create) | `build_judge_prompt`, `run_judge`, `judge_output` (shared with eval) |
| `src/medic_agent/agents/coding_agent.py` (create) | Nodes + `build_coding_agent()` compiled subgraph |
| `src/medic_agent/agents/ambient_agent.py` (create) | Nodes + `build_ambient_agent()` compiled subgraph |
| `src/medic_agent/agents/orchestrator.py` (create) | `build_graph()`, `run()` entry point, Session logging |
| `src/medic_agent/observability/tracer.py` (modify) | New Session fields + nested span emission |
| `src/medic_agent/evaluation/runner.py` (modify) | Route through orchestrator; share judge; router-accuracy check |
| `src/medic_agent/ui/app.py` (modify) | Routing-mode sidebar, routing panel, Tab 2 all-prompts, Tab 3 route/judge columns |
| `tests/agents/*` (create) | Unit tests for router, agents, judge, orchestrator |
| `tests/observability/test_tracer.py` (modify) | Add V2 nested-span tests |
| `pyproject.toml` (modify) | Add `langgraph` dependency |

---

## Task 1: Add LangGraph dependency & scaffolding

**Files:**
- Modify: `pyproject.toml`
- Create: `src/medic_agent/agents/__init__.py`, `tests/agents/__init__.py`

- [ ] **Step 1: Add the dependency**

Run: `uv add langgraph`

- [ ] **Step 2: Create package init files**

Create `src/medic_agent/agents/__init__.py` (empty) and `tests/agents/__init__.py` (empty).

- [ ] **Step 3: Verify import**

Run: `uv run python -c "import langgraph; from langgraph.graph import StateGraph, START, END; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock src/medic_agent/agents/__init__.py tests/agents/__init__.py
git commit -m "build(v2): add langgraph + agents package scaffolding"
```

---

## Task 2: Settings — canonical names, models, and prompts

**Files:**
- Modify: `src/medic_agent/config/settings.py`

- [ ] **Step 1: Append V2 constants to `settings.py`**

Add after the existing `USE_CASES` / `DEFAULT_USE_CASE` block:

```python
# --- V2: Multi-Agent Orchestration ---

USE_CASE_CODING = "Medical Coding"
USE_CASE_AMBIENT = "Ambient Note Taking"

ROUTER_MODEL_ID = AVAILABLE_MODELS["Claude Haiku (Fast)"]
JUDGE_MODEL_ID = "claude-sonnet-4-6"

ROUTER_SYSTEM_PROMPT = """\
You classify a single clinical AI request into exactly one agent.
- "Ambient Note Taking": the input is a physician-patient conversation transcript \
to convert into a SOAP note and billing codes.
- "Medical Coding": the input is clinical documentation (notes, reports, summaries) \
to assign billing codes for.
Return ONLY JSON, no markdown:
{"use_case": "Medical Coding" | "Ambient Note Taking", "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}\
"""

CODING_EXTRACT_PROMPT = """\
You are a clinical entity extractor. From the documentation provided, list every \
explicitly documented diagnosis and procedure as concise bullet points. Extract only \
what is documented — never infer. Output exactly:
Diagnoses:
- <diagnosis>
Procedures:
- <procedure>\
"""

CODING_VERIFY_PROMPT = """\
You are a medical coding auditor. You are given a DRAFT of assigned codes plus the \
source documentation context. For every code, confirm the documentation supports it; \
remove or down-flag any code not supported. Return the final corrected output, keeping \
the DIAGNOSIS CODES (ICD-10-CM), PROCEDURE CODES (CPT), and DOCUMENTATION GAPS sections.\
"""

AMBIENT_CODE_PROMPT = """\
You are a medical coder. Given a SOAP note, assign ICD-10-CM codes for each diagnosis in \
the Assessment and a CPT E&M code based on the documented complexity. Output ONLY a \
BILLING CODES section listing each code with a short description.\
"""

AMBIENT_VERIFY_PROMPT = """\
You are a clinical documentation auditor. You are given a SOAP note and a billing codes \
list. Combine them into the final output. Verify the note never fabricates findings beyond \
the transcript. Output the full SOAP NOTE, then BILLING CODES, then a DOCUMENTATION FLAGS \
section listing anything missing from a complete note.\
"""

CODING_JUDGE_PROMPT = """\
- code_accuracy: Are the suggested codes clinically appropriate for the documentation?
- code_completeness: Are any important codes missing?
- citation_quality: Does each code cite supporting documentation?
- hallucination: Are all codes supported by the documentation? (5 = none unsupported)
- gap_identification: Did it correctly flag documentation gaps?\
"""

AMBIENT_JUDGE_PROMPT = """\
- soap_completeness: Are all four SOAP sections present and appropriately populated?
- subj_obj_distinction: Patient-reported info in S, clinician-observed in O?
- code_accuracy: Are ICD/CPT codes clinically appropriate?
- faithfulness: Does the note stick to the transcript with no fabricated facts?
- clinical_quality: Would this SOAP be acceptable in a clinical setting?\
"""
```

- [ ] **Step 2: Verify it imports**

Run: `uv run python -c "from medic_agent.config import settings as s; print(s.USE_CASE_CODING, s.ROUTER_MODEL_ID, s.JUDGE_MODEL_ID)"`
Expected: `Medical Coding claude-haiku-4-5-20251001 claude-sonnet-4-6`

- [ ] **Step 3: Commit**

```bash
git add src/medic_agent/config/settings.py
git commit -m "feat(v2): add router/agent/judge prompts and canonical use-case names"
```

---

## Task 3: Centralized prompt persistence (`config/prompts.py`)

**Files:**
- Create: `src/medic_agent/config/prompts.py`
- Test: `tests/agents/test_prompts.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/test_prompts.py
import json

import medic_agent.config.prompts as prompts_module
from medic_agent.config.prompts import get_prompt, load_all, save_all, PROMPT_DEFAULTS


def test_get_prompt_falls_back_to_defaults(tmp_path, mocker):
    mocker.patch.object(prompts_module, "PROMPTS_FILE", tmp_path / "prompts.json")
    assert get_prompt("router") == PROMPT_DEFAULTS["router"]
    assert get_prompt("coding", "extract") == PROMPT_DEFAULTS["coding"]["extract"]


def test_save_then_get_prompt_round_trips(tmp_path, mocker):
    pf = tmp_path / "prompts.json"
    mocker.patch.object(prompts_module, "PROMPTS_FILE", pf)
    mocker.patch.object(prompts_module, "_push_to_langfuse", lambda *a, **k: None)

    edited = json.loads(json.dumps(PROMPT_DEFAULTS))
    edited["coding"]["verify"] = "CUSTOM VERIFY"
    version = save_all(edited)

    assert version == 1
    assert get_prompt("coding", "verify") == "CUSTOM VERIFY"
    assert get_prompt("router") == PROMPT_DEFAULTS["router"]  # untouched key still default


def test_missing_nested_key_uses_default(tmp_path, mocker):
    pf = tmp_path / "prompts.json"
    pf.write_text(json.dumps({"version": 1, "coding": {"extract": "X"}}))
    mocker.patch.object(prompts_module, "PROMPTS_FILE", pf)
    assert get_prompt("coding", "extract") == "X"
    assert get_prompt("coding", "verify") == PROMPT_DEFAULTS["coding"]["verify"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/agents/test_prompts.py -v`
Expected: FAIL with `ModuleNotFoundError: medic_agent.config.prompts`

- [ ] **Step 3: Implement `config/prompts.py`**

```python
import json
from datetime import datetime, timezone

from medic_agent.config.settings import (
    AMBIENT_CODE_PROMPT,
    AMBIENT_JUDGE_PROMPT,
    AMBIENT_SYSTEM_PROMPT,
    AMBIENT_VERIFY_PROMPT,
    CODING_EXTRACT_PROMPT,
    CODING_JUDGE_PROMPT,
    CODING_SYSTEM_PROMPT,
    CODING_VERIFY_PROMPT,
    PROMPTS_FILE,
    ROUTER_SYSTEM_PROMPT,
)

PROMPT_DEFAULTS: dict = {
    "router": ROUTER_SYSTEM_PROMPT,
    "coding": {
        "extract": CODING_EXTRACT_PROMPT,
        "code": CODING_SYSTEM_PROMPT,
        "verify": CODING_VERIFY_PROMPT,
    },
    "ambient": {
        "soap": AMBIENT_SYSTEM_PROMPT,
        "code": AMBIENT_CODE_PROMPT,
        "verify": AMBIENT_VERIFY_PROMPT,
    },
    "judge": {
        "coding": CODING_JUDGE_PROMPT,
        "ambient": AMBIENT_JUDGE_PROMPT,
    },
}


def _read_file() -> dict:
    if PROMPTS_FILE.exists():
        try:
            return json.loads(PROMPTS_FILE.read_text())
        except Exception:
            pass
    return {}


def get_prompt(*keys: str) -> str:
    node = _read_file()
    default = PROMPT_DEFAULTS
    for key in keys:
        node = node.get(key, {}) if isinstance(node, dict) else {}
        default = default[key]
    if isinstance(node, str) and node:
        return node
    return default


def load_all() -> dict:
    stored = _read_file()
    merged = json.loads(json.dumps(PROMPT_DEFAULTS))
    for group, val in PROMPT_DEFAULTS.items():
        if isinstance(val, dict):
            merged[group] = {**val, **stored.get(group, {})}
        else:
            merged[group] = stored.get(group, val)
    merged["version"] = stored.get("version", 0)
    return merged


def save_all(prompts: dict) -> int:
    existing = _read_file()
    version = existing.get("version", 0) + 1
    payload = {
        "version": version,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "router": prompts["router"],
        "coding": prompts["coding"],
        "ambient": prompts["ambient"],
        "judge": prompts["judge"],
    }
    PROMPTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROMPTS_FILE.write_text(json.dumps(payload, indent=2))
    _push_to_langfuse(payload, version)
    return version


def _push_to_langfuse(payload: dict, version: int) -> None:
    try:
        from medic_agent.observability.tracer import (
            _LANGFUSE_ENABLED,
            _langfuse_client,
        )

        if not _LANGFUSE_ENABLED or _langfuse_client is None:
            return
        flat = {
            "router": payload["router"],
            "coding-code": payload["coding"]["code"],
            "ambient-soap": payload["ambient"]["soap"],
        }
        for name, text in flat.items():
            _langfuse_client.create_prompt(
                name=f"medic-{name}-prompt", prompt=text, labels=["production"]
            )
    except Exception:
        pass
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/agents/test_prompts.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/medic_agent/config/prompts.py tests/agents/test_prompts.py
git commit -m "feat(v2): centralized nested prompt load/save in config/prompts.py"
```

---

## Task 4: Non-logging `complete()` primitive

**Files:**
- Modify: `src/medic_agent/llm/client.py`
- Test: `tests/llm/test_complete.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/llm/test_complete.py
from unittest.mock import MagicMock

import medic_agent.llm.client as client_module
from medic_agent.llm.client import complete


def _mock_response(text: str):
    resp = MagicMock()
    resp.choices[0].message.content = text
    resp.usage.prompt_tokens = 10
    resp.usage.completion_tokens = 5
    resp.usage.total_tokens = 15
    return resp


def test_complete_returns_text_and_usage(mocker):
    mocker.patch.object(
        client_module, "completion", return_value=_mock_response("E11.65")
    )
    text, usage = complete("claude-haiku-4-5-20251001", "sys", "user query")
    assert text == "E11.65"
    assert usage["total_tokens"] == 15


def test_complete_does_not_log_session(mocker, tmp_path):
    mocker.patch.object(
        client_module, "completion", return_value=_mock_response("ok")
    )
    spy = mocker.patch.object(client_module, "log_session")
    complete("claude-haiku-4-5-20251001", "sys", "q")
    spy.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/llm/test_complete.py -v`
Expected: FAIL with `ImportError: cannot import name 'complete'`

- [ ] **Step 3: Add `complete()` to `client.py`**

Insert after `_build_messages` (before `ask`):

```python
def complete(
    model_id: str,
    system_prompt: str,
    user_query: str,
    context: list[dict] | None = None,
) -> tuple[str, dict]:
    messages = _build_messages(system_prompt, user_query, context)
    response = completion(model=model_id, messages=messages)
    text = response.choices[0].message.content
    usage: dict = {}
    try:
        u = response.usage
        usage = {
            "prompt_tokens": u.prompt_tokens,
            "completion_tokens": u.completion_tokens,
            "total_tokens": u.total_tokens,
        }
    except Exception:
        pass
    return text, usage
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/llm/test_complete.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/medic_agent/llm/client.py tests/llm/test_complete.py
git commit -m "feat(v2): add non-logging complete() primitive for agent nodes"
```

---

## Task 5: Agent state + `llm_step` helper

**Files:**
- Create: `src/medic_agent/agents/state.py`
- Test: `tests/agents/test_state.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/test_state.py
import medic_agent.agents.state as state_module
from medic_agent.agents.state import llm_step


def test_llm_step_returns_text_and_step_record(mocker):
    mocker.patch.object(
        state_module, "complete", return_value=("RESULT", {"total_tokens": 7})
    )
    text, step = llm_step(
        "claude-haiku-4-5-20251001", "extract", "sys prompt", "user input"
    )
    assert text == "RESULT"
    assert step["name"] == "extract"
    assert step["type"] == "generation"
    assert step["model_id"] == "claude-haiku-4-5-20251001"
    assert step["token_usage"] == {"total_tokens": 7}
    assert "latency_ms" in step
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/agents/test_state.py -v`
Expected: FAIL with `ModuleNotFoundError: medic_agent.agents.state`

- [ ] **Step 3: Implement `agents/state.py`**

```python
import operator
import time
from typing import Annotated, TypedDict

from medic_agent.llm.client import complete


class AgentState(TypedDict, total=False):
    query: str
    model_id: str
    override: str
    judge_on: bool
    use_case: str
    route: dict
    chunks: list[dict]
    scratch: dict
    response: str
    judge_scores: dict
    agent_steps: Annotated[list[dict], operator.add]


def llm_step(
    model_id: str,
    name: str,
    system_prompt: str,
    user_query: str,
    context: list[dict] | None = None,
    as_type: str = "generation",
) -> tuple[str, dict]:
    start = time.monotonic()
    text, usage = complete(model_id, system_prompt, user_query, context)
    latency = round((time.monotonic() - start) * 1000, 1)
    step = {
        "name": name,
        "type": as_type,
        "model_id": model_id,
        "token_usage": usage,
        "latency_ms": latency,
        "input_summary": (user_query or "")[:200],
        "output_summary": (text or "")[:200],
    }
    return text, step
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/agents/test_state.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add src/medic_agent/agents/state.py tests/agents/test_state.py
git commit -m "feat(v2): AgentState (with agent_steps reducer) + llm_step helper"
```

---

## Task 6: Hybrid router

**Files:**
- Create: `src/medic_agent/agents/router.py`
- Test: `tests/agents/test_router.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/test_router.py
import medic_agent.agents.router as router_module
from medic_agent.agents.router import RouteDecision, heuristic_route, route
from medic_agent.config.settings import USE_CASE_AMBIENT, USE_CASE_CODING

TRANSCRIPT = "Doctor: What brings you in?\nPatient: My knee hurts.\nDoctor: How long?"


def test_heuristic_detects_transcript_as_ambient():
    d = heuristic_route(TRANSCRIPT)
    assert d is not None and d.use_case == USE_CASE_AMBIENT and d.method == "heuristic"


def test_heuristic_detects_coding_keyword():
    d = heuristic_route("Assign the ICD-10 and CPT codes for this encounter.")
    assert d is not None and d.use_case == USE_CASE_CODING


def test_heuristic_returns_none_when_ambiguous():
    assert heuristic_route("Summarize this clinical note for me.") is None


def test_override_short_circuits_to_manual():
    d = route("anything", override=USE_CASE_CODING)
    assert d.use_case == USE_CASE_CODING and d.method == "manual" and d.confidence == 1.0


def test_ambiguous_query_falls_back_to_llm(mocker):
    mocker.patch.object(
        router_module,
        "llm_route",
        return_value=RouteDecision(USE_CASE_CODING, "llm", 0.7, "looks like docs"),
    )
    d = route("Summarize this clinical note for me.", override="Auto")
    assert d.method == "llm" and d.use_case == USE_CASE_CODING
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/agents/test_router.py -v`
Expected: FAIL with `ModuleNotFoundError: medic_agent.agents.router`

- [ ] **Step 3: Implement `agents/router.py`**

```python
import json
import re
from dataclasses import dataclass

from medic_agent.config.settings import (
    ROUTER_MODEL_ID,
    ROUTER_SYSTEM_PROMPT,
    USE_CASE_AMBIENT,
    USE_CASE_CODING,
)
from medic_agent.llm.client import complete

_SPEAKER_RE = re.compile(
    r"(?im)^\s*(doctor|patient|dr|pt|physician|provider|nurse)\s*[:>\-]"
)
_CODING_KW_RE = re.compile(
    r"(?i)\b(icd[-\s]?10|cpt|hcpcs|e/?m\s+code|billing\s+code|what\s+codes|code\s+this)\b"
)


@dataclass
class RouteDecision:
    use_case: str
    method: str  # "manual" | "heuristic" | "llm"
    confidence: float
    reasoning: str


def heuristic_route(query: str) -> RouteDecision | None:
    speaker_turns = len(_SPEAKER_RE.findall(query))
    if speaker_turns >= 2:
        return RouteDecision(
            USE_CASE_AMBIENT,
            "heuristic",
            0.9,
            f"Detected {speaker_turns} dialogue speaker turns",
        )
    if _CODING_KW_RE.search(query):
        return RouteDecision(
            USE_CASE_CODING, "heuristic", 0.85, "Matched coding keyword (ICD/CPT/billing)"
        )
    return None


def llm_route(query: str) -> RouteDecision:
    text, _ = complete(ROUTER_MODEL_ID, ROUTER_SYSTEM_PROMPT, query)
    use_case, confidence, reasoning = USE_CASE_CODING, 0.5, ""
    start, end = text.find("{"), text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start:end])
            raw = str(data.get("use_case", "")).lower()
            use_case = USE_CASE_AMBIENT if "ambient" in raw else USE_CASE_CODING
            confidence = float(data.get("confidence", 0.5))
            reasoning = str(data.get("reasoning", ""))
        except Exception:
            pass
    return RouteDecision(use_case, "llm", confidence, reasoning)


def route(query: str, override: str = "Auto") -> RouteDecision:
    if override in (USE_CASE_CODING, USE_CASE_AMBIENT):
        return RouteDecision(override, "manual", 1.0, "User override")
    decision = heuristic_route(query)
    if decision is not None:
        return decision
    return llm_route(query)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/agents/test_router.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/medic_agent/agents/router.py tests/agents/test_router.py
git commit -m "feat(v2): hybrid router (heuristic + Haiku LLM fallback)"
```

---

## Task 7: Online judge (shared with eval)

**Files:**
- Create: `src/medic_agent/agents/judge.py`
- Test: `tests/agents/test_judge.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/test_judge.py
import medic_agent.agents.judge as judge_module
from medic_agent.agents.judge import build_judge_prompt, judge_output, run_judge


def test_build_judge_prompt_includes_schema_keys():
    criteria = "- code_accuracy: x\n- hallucination: y"
    prompt = build_judge_prompt("a coding case", "RESPONSE", criteria)
    assert '"code_accuracy"' in prompt and '"hallucination"' in prompt
    assert '"overall"' in prompt


def test_run_judge_parses_embedded_json(mocker):
    raw = 'Here you go: {"code_accuracy": 4, "overall": 4.2} done'
    mocker.patch.object(judge_module, "complete", return_value=(raw, {}))
    scores, returned_raw, prompt = run_judge("desc", "resp", "- code_accuracy: x")
    assert scores == {"code_accuracy": 4, "overall": 4.2}
    assert returned_raw == raw


def test_judge_output_uses_rubric_for_use_case(mocker):
    mocker.patch.object(
        judge_module, "complete", return_value=('{"overall": 5.0}', {})
    )
    scores = judge_output("Medical Coding", "some response")
    assert scores == {"overall": 5.0}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/agents/test_judge.py -v`
Expected: FAIL with `ModuleNotFoundError: medic_agent.agents.judge`

- [ ] **Step 3: Implement `agents/judge.py`**

```python
import json
import warnings

from medic_agent.config.prompts import get_prompt
from medic_agent.config.settings import (
    JUDGE_MODEL_ID,
    USE_CASE_AMBIENT,
)
from medic_agent.llm.client import complete

JUDGE_SYSTEM_PROMPT = (
    "You are a medical documentation evaluation expert. "
    "Return only valid JSON with no markdown."
)


def build_judge_prompt(description: str, response: str, criteria_text: str) -> str:
    criteria_keys = [
        line.split(":")[0].lstrip("- ").strip()
        for line in criteria_text.splitlines()
        if ":" in line
    ]
    schema = (
        "{"
        + ", ".join(f'"{k}": <1-5>' for k in criteria_keys)
        + ', "overall": <1.0-5.0>}'
    )
    return (
        f"Case: {description}\n\n"
        f"Evaluation criteria:\n{criteria_text}\n\n"
        f"Response to evaluate (first 1500 chars):\n{response[:1500]}\n\n"
        f"Score each criterion 1-5 and provide a weighted overall score 1.0-5.0.\n"
        f"Return ONLY this JSON (no markdown, no explanation):\n{schema}"
    )


def run_judge(
    description: str,
    response: str,
    criteria_text: str,
    model_id: str = JUDGE_MODEL_ID,
) -> tuple[dict | None, str, str]:
    prompt = build_judge_prompt(description, response, criteria_text)
    try:
        raw, _ = complete(model_id, JUDGE_SYSTEM_PROMPT, prompt)
        start, end = raw.find("{"), raw.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(raw[start:end]), raw, prompt
    except Exception as e:
        warnings.warn(f"LLM-as-judge failed: {e}")
        return {"error": str(e)}, "", prompt
    return None, "", prompt


def judge_output(use_case: str, response: str, model_id: str = JUDGE_MODEL_ID) -> dict:
    rubric_key = "ambient" if use_case == USE_CASE_AMBIENT else "coding"
    criteria_text = get_prompt("judge", rubric_key)
    scores, _, _ = run_judge(
        f"{use_case} response", response, criteria_text, model_id
    )
    return scores or {}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/agents/test_judge.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/medic_agent/agents/judge.py tests/agents/test_judge.py
git commit -m "feat(v2): online LLM-as-judge with shared rubric/prompt builder"
```

---

## Task 8: Coding agent subgraph

**Files:**
- Create: `src/medic_agent/agents/coding_agent.py`
- Test: `tests/agents/test_coding_agent.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/test_coding_agent.py
import medic_agent.agents.coding_agent as coding_module
from medic_agent.agents.coding_agent import build_coding_agent


def test_coding_agent_runs_all_steps_and_produces_response(mocker):
    # Each llm_step call returns (text, step_record); sequence: extract, code, verify
    mocker.patch.object(
        coding_module,
        "llm_step",
        side_effect=[
            ("Diagnoses:\n- DM2", {"name": "extract", "type": "generation"}),
            ("DRAFT CODES E11.9", {"name": "code", "type": "generation"}),
            ("FINAL E11.9 99213", {"name": "verify", "type": "generation"}),
        ],
    )
    mocker.patch.object(
        coding_module,
        "retrieve",
        return_value=[{"text": "ctx", "source_filename": "f", "chunk_index": 0}],
    )

    graph = build_coding_agent()
    out = graph.invoke(
        {"query": "code this", "model_id": "claude-haiku-4-5-20251001", "scratch": {}}
    )
    assert out["response"] == "FINAL E11.9 99213"
    assert len(out["chunks"]) == 1
    step_names = [s["name"] for s in out["agent_steps"]]
    assert step_names == ["extract", "retrieval", "code", "verify"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/agents/test_coding_agent.py -v`
Expected: FAIL with `ModuleNotFoundError: medic_agent.agents.coding_agent`

- [ ] **Step 3: Implement `agents/coding_agent.py`**

```python
import time

from langgraph.graph import END, START, StateGraph

from medic_agent.agents.state import AgentState, llm_step
from medic_agent.config.prompts import get_prompt
from medic_agent.rag.retriever import retrieve


def extract(state: AgentState) -> dict:
    text, step = llm_step(
        state["model_id"], "extract", get_prompt("coding", "extract"), state["query"]
    )
    scratch = {**state.get("scratch", {}), "entities": text}
    return {"scratch": scratch, "agent_steps": [step]}


def retrieve_node(state: AgentState) -> dict:
    entities = state.get("scratch", {}).get("entities", "")
    start = time.monotonic()
    chunks = retrieve(f"{state['query']}\n{entities}", k=5)
    step = {
        "name": "retrieval",
        "type": "retriever",
        "model_id": "",
        "token_usage": {},
        "latency_ms": round((time.monotonic() - start) * 1000, 1),
        "input_summary": state["query"][:200],
        "output_summary": f"{len(chunks)} chunks",
    }
    return {"chunks": chunks, "agent_steps": [step]}


def code(state: AgentState) -> dict:
    text, step = llm_step(
        state["model_id"],
        "code",
        get_prompt("coding", "code"),
        state["query"],
        context=state.get("chunks"),
    )
    scratch = {**state.get("scratch", {}), "draft": text}
    return {"scratch": scratch, "agent_steps": [step]}


def verify(state: AgentState) -> dict:
    draft = state.get("scratch", {}).get("draft", "")
    user = f"DRAFT coding output:\n{draft}\n\nReview against the documentation and finalize."
    text, step = llm_step(
        state["model_id"],
        "verify",
        get_prompt("coding", "verify"),
        user,
        context=state.get("chunks"),
    )
    return {"response": text, "agent_steps": [step]}


def build_coding_agent():
    g = StateGraph(AgentState)
    g.add_node("extract", extract)
    g.add_node("retrieve", retrieve_node)
    g.add_node("code", code)
    g.add_node("verify", verify)
    g.add_edge(START, "extract")
    g.add_edge("extract", "retrieve")
    g.add_edge("retrieve", "code")
    g.add_edge("code", "verify")
    g.add_edge("verify", END)
    return g.compile()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/agents/test_coding_agent.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add src/medic_agent/agents/coding_agent.py tests/agents/test_coding_agent.py
git commit -m "feat(v2): coding agent subgraph (extract->retrieve->code->verify)"
```

---

## Task 9: Ambient agent subgraph

**Files:**
- Create: `src/medic_agent/agents/ambient_agent.py`
- Test: `tests/agents/test_ambient_agent.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/test_ambient_agent.py
import medic_agent.agents.ambient_agent as ambient_module
from medic_agent.agents.ambient_agent import build_ambient_agent


def test_ambient_agent_runs_all_steps_and_produces_response(mocker):
    # llm_step sequence: soap, code, verify
    mocker.patch.object(
        ambient_module,
        "llm_step",
        side_effect=[
            ("S O A P NOTE", {"name": "soap", "type": "generation"}),
            ("BILLING CODES E11.9", {"name": "code", "type": "generation"}),
            ("FINAL NOTE + CODES + FLAGS", {"name": "verify", "type": "generation"}),
        ],
    )
    mocker.patch.object(ambient_module, "retrieve", return_value=[])

    graph = build_ambient_agent()
    out = graph.invoke(
        {
            "query": "Doctor: ...\nPatient: ...",
            "model_id": "claude-haiku-4-5-20251001",
            "scratch": {},
        }
    )
    assert out["response"] == "FINAL NOTE + CODES + FLAGS"
    step_names = [s["name"] for s in out["agent_steps"]]
    assert step_names == ["retrieval", "soap", "code", "verify"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/agents/test_ambient_agent.py -v`
Expected: FAIL with `ModuleNotFoundError: medic_agent.agents.ambient_agent`

- [ ] **Step 3: Implement `agents/ambient_agent.py`**

```python
import time

from langgraph.graph import END, START, StateGraph

from medic_agent.agents.state import AgentState, llm_step
from medic_agent.config.prompts import get_prompt
from medic_agent.rag.retriever import retrieve


def retrieve_node(state: AgentState) -> dict:
    start = time.monotonic()
    chunks = retrieve(state["query"], k=5)
    step = {
        "name": "retrieval",
        "type": "retriever",
        "model_id": "",
        "token_usage": {},
        "latency_ms": round((time.monotonic() - start) * 1000, 1),
        "input_summary": state["query"][:200],
        "output_summary": f"{len(chunks)} chunks",
    }
    return {"chunks": chunks, "agent_steps": [step]}


def soap(state: AgentState) -> dict:
    text, step = llm_step(
        state["model_id"],
        "soap",
        get_prompt("ambient", "soap"),
        state["query"],
        context=state.get("chunks"),
    )
    scratch = {**state.get("scratch", {}), "soap": text}
    return {"scratch": scratch, "agent_steps": [step]}


def code(state: AgentState) -> dict:
    soap_note = state.get("scratch", {}).get("soap", "")
    text, step = llm_step(
        state["model_id"], "code", get_prompt("ambient", "code"), soap_note
    )
    scratch = {**state.get("scratch", {}), "codes": text}
    return {"scratch": scratch, "agent_steps": [step]}


def verify(state: AgentState) -> dict:
    sc = state.get("scratch", {})
    user = (
        f"SOAP NOTE:\n{sc.get('soap', '')}\n\n"
        f"BILLING CODES:\n{sc.get('codes', '')}\n\n"
        "Combine into the final output and add documentation flags."
    )
    text, step = llm_step(
        state["model_id"], "verify", get_prompt("ambient", "verify"), user
    )
    return {"response": text, "agent_steps": [step]}


def build_ambient_agent():
    g = StateGraph(AgentState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("soap", soap)
    g.add_node("code", code)
    g.add_node("verify", verify)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "soap")
    g.add_edge("soap", "code")
    g.add_edge("code", "verify")
    g.add_edge("verify", END)
    return g.compile()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/agents/test_ambient_agent.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add src/medic_agent/agents/ambient_agent.py tests/agents/test_ambient_agent.py
git commit -m "feat(v2): ambient agent subgraph (retrieve->soap->code->verify)"
```

---

## Task 10: Orchestrator graph + Session logging

**Files:**
- Create: `src/medic_agent/agents/orchestrator.py`
- Test: `tests/agents/test_orchestrator.py`
- Depends on: Task 11 adds the new `Session` fields. **Do Task 11 before Task 10's Step 3** (or stub the fields). Order note: implement Task 11 first if running strictly sequentially.

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/test_orchestrator.py
import medic_agent.agents.orchestrator as orch_module
from medic_agent.agents.router import RouteDecision
from medic_agent.config.settings import USE_CASE_AMBIENT, USE_CASE_CODING


def test_run_routes_to_coding_and_logs_one_session(mocker):
    mocker.patch.object(
        orch_module,
        "route",
        return_value=RouteDecision(USE_CASE_CODING, "heuristic", 0.85, "kw"),
    )
    mocker.patch.object(
        orch_module, "_run_coding", return_value={"response": "CODES", "chunks": []}
    )
    mocker.patch.object(orch_module, "judge_output", return_value={"overall": 4.0})
    log_spy = mocker.patch.object(orch_module, "log_session")

    out = orch_module.run("code this", "claude-haiku-4-5-20251001")

    assert out["use_case"] == USE_CASE_CODING
    assert out["response"] == "CODES"
    assert out["judge_scores"] == {"overall": 4.0}
    assert out["route"]["method"] == "heuristic"
    log_spy.assert_called_once()


def test_run_respects_manual_override(mocker):
    mocker.patch.object(
        orch_module,
        "route",
        return_value=RouteDecision(USE_CASE_AMBIENT, "manual", 1.0, "override"),
    )
    ambient_spy = mocker.patch.object(
        orch_module, "_run_ambient", return_value={"response": "SOAP", "chunks": []}
    )
    mocker.patch.object(orch_module, "judge_output", return_value={})
    mocker.patch.object(orch_module, "log_session")

    out = orch_module.run("x", "claude-haiku-4-5-20251001", override=USE_CASE_AMBIENT)
    assert out["use_case"] == USE_CASE_AMBIENT
    ambient_spy.assert_called_once()


def test_run_skips_judge_when_disabled(mocker):
    mocker.patch.object(
        orch_module,
        "route",
        return_value=RouteDecision(USE_CASE_CODING, "heuristic", 0.85, "kw"),
    )
    mocker.patch.object(
        orch_module, "_run_coding", return_value={"response": "CODES", "chunks": []}
    )
    judge_spy = mocker.patch.object(orch_module, "judge_output")
    mocker.patch.object(orch_module, "log_session")

    out = orch_module.run("code this", "claude-haiku-4-5-20251001", judge_on=False)
    judge_spy.assert_not_called()
    assert out["judge_scores"] == {}
```

**Design note for the implementer:** the test mocks `route`, `_run_coding`, `_run_ambient`, `judge_output`, and `log_session` at the orchestrator module level. Keep the agent dispatch behind module-level helpers `_run_coding(state)` / `_run_ambient(state)` (which invoke the compiled subgraphs) so they are mockable without running LangGraph in unit tests. A separate non-mocked smoke test (Step 4b) exercises the real compiled graph.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/agents/test_orchestrator.py -v`
Expected: FAIL with `ModuleNotFoundError: medic_agent.agents.orchestrator`

- [ ] **Step 3: Implement `agents/orchestrator.py`**

```python
from dataclasses import asdict

from langgraph.graph import END, START, StateGraph

from medic_agent.agents.ambient_agent import build_ambient_agent
from medic_agent.agents.coding_agent import build_coding_agent
from medic_agent.agents.judge import judge_output
from medic_agent.agents.router import route
from medic_agent.agents.state import AgentState
from medic_agent.config.settings import USE_CASE_AMBIENT
from medic_agent.observability.tracer import Session, log_session

_CODING_AGENT = build_coding_agent()
_AMBIENT_AGENT = build_ambient_agent()


def _run_coding(state: AgentState) -> dict:
    return _CODING_AGENT.invoke(state)


def _run_ambient(state: AgentState) -> dict:
    return _AMBIENT_AGENT.invoke(state)


def _router_node(state: AgentState) -> dict:
    decision = route(state["query"], state.get("override", "Auto"))
    step = {
        "name": "router",
        "type": "span",
        "model_id": "",
        "token_usage": {},
        "latency_ms": 0,
        "input_summary": state["query"][:200],
        "output_summary": f"{decision.use_case} ({decision.method}, {decision.confidence})",
    }
    return {"use_case": decision.use_case, "route": asdict(decision), "agent_steps": [step]}


def _agent_node(state: AgentState) -> dict:
    if state["use_case"] == USE_CASE_AMBIENT:
        return _run_ambient(state)
    return _run_coding(state)


def _judge_node(state: AgentState) -> dict:
    if not state.get("judge_on", True) or not state.get("response"):
        return {}
    scores = judge_output(state["use_case"], state["response"])
    step = {
        "name": "judge",
        "type": "generation",
        "model_id": "",
        "token_usage": {},
        "latency_ms": 0,
        "input_summary": state["response"][:200],
        "output_summary": str(scores)[:200],
    }
    return {"judge_scores": scores, "agent_steps": [step]}


def _build_graph():
    g = StateGraph(AgentState)
    g.add_node("router", _router_node)
    g.add_node("agent", _agent_node)
    g.add_node("judge", _judge_node)
    g.add_edge(START, "router")
    g.add_edge("router", "agent")
    g.add_edge("agent", "judge")
    g.add_edge("judge", END)
    return g.compile()


_GRAPH = _build_graph()


def run(
    query: str,
    model_id: str,
    override: str = "Auto",
    judge_on: bool = True,
) -> dict:
    init: AgentState = {
        "query": query,
        "model_id": model_id,
        "override": override,
        "judge_on": judge_on,
        "scratch": {},
        "agent_steps": [],
    }
    final = _GRAPH.invoke(init)

    steps = final.get("agent_steps", [])
    token_usage = {
        "total_tokens": sum(s.get("token_usage", {}).get("total_tokens", 0) for s in steps)
    }
    session = Session(
        use_case=final.get("use_case", ""),
        model_id=model_id,
        query=query,
        response=final.get("response", ""),
        chunks_retrieved=final.get("chunks", []),
        latency_ms=sum(s.get("latency_ms", 0) for s in steps),
        token_usage=token_usage,
        route_decision=final.get("route", {}),
        agent_steps=steps,
        judge_scores=final.get("judge_scores", {}),
    )
    log_session(session)

    return {
        "use_case": final.get("use_case", ""),
        "route": final.get("route", {}),
        "response": final.get("response", ""),
        "chunks": final.get("chunks", []),
        "judge_scores": final.get("judge_scores", {}),
        "agent_steps": steps,
    }
```

- [ ] **Step 4: Run unit tests to verify they pass**

Run: `uv run pytest tests/agents/test_orchestrator.py -v`
Expected: 3 passed

- [ ] **Step 4b: Add a real-graph smoke test (mocks only external LLM/retrieval)**

Append to `tests/agents/test_orchestrator.py`:

```python
def test_real_graph_dispatches_to_ambient(mocker):
    import medic_agent.agents.ambient_agent as ambient_module
    import medic_agent.agents.coding_agent as coding_module

    mocker.patch.object(ambient_module, "retrieve", return_value=[])
    mocker.patch.object(coding_module, "retrieve", return_value=[])
    mocker.patch.object(
        ambient_module, "llm_step", return_value=("SOAP OUT", {"name": "soap", "type": "generation"})
    )
    mocker.patch.object(orch_module, "judge_output", return_value={})
    mocker.patch.object(orch_module, "log_session")

    out = orch_module.run(
        "Doctor: hi\nPatient: hi\nDoctor: ok", "claude-haiku-4-5-20251001"
    )
    assert out["use_case"] == USE_CASE_AMBIENT
    assert out["response"] == "SOAP OUT"
```

Run: `uv run pytest tests/agents/test_orchestrator.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/medic_agent/agents/orchestrator.py tests/agents/test_orchestrator.py
git commit -m "feat(v2): orchestrator graph (router->agent->judge) with single Session log"
```

---

## Task 11: Enhanced multi-agent tracing

> Implement this BEFORE Task 10 Step 3 if running strictly in order (orchestrator constructs `Session` with the new fields).

**Files:**
- Modify: `src/medic_agent/observability/tracer.py`
- Test: `tests/observability/test_tracer.py`

- [ ] **Step 1: Add new fields to the `Session` dataclass**

In `tracer.py`, add to `Session` (after `error`):

```python
    route_decision: dict = field(default_factory=dict)
    agent_steps: list[dict] = field(default_factory=list)
    judge_scores: dict = field(default_factory=dict)
```

- [ ] **Step 2: Write the failing test for nested emission**

Append to `tests/observability/test_tracer.py`:

```python
def _make_v2_session(**overrides):
    s = _make_session(**overrides)
    s.route_decision = {"use_case": "Medical Coding", "method": "heuristic", "confidence": 0.85}
    s.agent_steps = [
        {"name": "router", "type": "span", "model_id": "", "output_summary": "Medical Coding"},
        {"name": "extract", "type": "generation", "model_id": "haiku", "output_summary": "dx"},
        {"name": "retrieval", "type": "retriever", "model_id": "", "output_summary": "5 chunks"},
        {"name": "code", "type": "generation", "model_id": "haiku", "output_summary": "E11.9"},
        {"name": "verify", "type": "generation", "model_id": "haiku", "output_summary": "final"},
        {"name": "judge", "type": "generation", "model_id": "sonnet", "output_summary": "{}"},
    ]
    s.judge_scores = {"overall": 4.2}
    return s


def test_v2_session_emits_nested_agent_spans(mocker):
    mock_client = MagicMock()
    mocker.patch.object(tracer_module, "_langfuse_client", mock_client)

    tracer_module._send_to_langfuse(_make_v2_session())

    # outer(1) + router(1) + agent(1) + 4 agent steps + judge(1) = 8
    assert mock_client.start_as_current_observation.call_count == 8
    mock_client.flush.assert_called_once()


def test_v1_session_still_emits_three_spans(mocker):
    mock_client = MagicMock()
    mocker.patch.object(tracer_module, "_langfuse_client", mock_client)

    tracer_module._send_to_langfuse(_make_session())  # no agent_steps
    assert mock_client.start_as_current_observation.call_count == 3
```

- [ ] **Step 3: Run test to verify the new one fails**

Run: `uv run pytest tests/observability/test_tracer.py::test_v2_session_emits_nested_agent_spans -v`
Expected: FAIL (count is 3, not 8)

- [ ] **Step 4: Rewrite `_send_to_langfuse` to branch on `agent_steps`**

Replace the body of `_send_to_langfuse` (keep the `try/except` wrapper and `TraceContext` import):

```python
def _send_to_langfuse(session: Session) -> None:
    try:
        from langfuse.types import TraceContext

        ctx = TraceContext(trace_id=session.session_id.replace("-", ""))
        metadata = {
            "use_case": session.use_case,
            "system_prompt_version": session.system_prompt_version,
            "chunks_retrieved_count": len(session.chunks_retrieved),
            "latency_ms": session.latency_ms,
            "route_decision": session.route_decision,
            "judge_scores": session.judge_scores,
        }

        with _langfuse_client.start_as_current_observation(
            trace_context=ctx,
            name="medic-agent-query",
            as_type="span",
            input=session.query,
            output=session.response,
            metadata=metadata,
        ):
            if session.agent_steps:
                _emit_agent_spans(session)
            else:
                _emit_legacy_spans(session)

        _langfuse_client.flush()
    except Exception as e:
        warnings.warn(f"LangFuse trace failed: {e}. Session saved locally.")


def _emit_agent_spans(session: Session) -> None:
    steps = session.agent_steps
    agent_steps = [s for s in steps if s["name"] not in ("router", "judge")]
    has_router = any(s["name"] == "router" for s in steps)
    has_judge = any(s["name"] == "judge" for s in steps)

    if has_router:
        with _langfuse_client.start_as_current_observation(
            name="router",
            as_type="span",
            input=session.query,
            output=session.route_decision,
        ):
            pass

    with _langfuse_client.start_as_current_observation(
        name=f"agent:{session.use_case}",
        as_type="span",
        input=session.query,
        output=session.response,
    ):
        for s in agent_steps:
            as_type = "retriever" if s.get("type") == "retriever" else "generation"
            with _langfuse_client.start_as_current_observation(
                name=s["name"],
                as_type=as_type,
                model=s.get("model_id") or None,
                output=s.get("output_summary"),
            ):
                pass

    if has_judge:
        with _langfuse_client.start_as_current_observation(
            name="judge",
            as_type="generation",
            output=session.judge_scores,
        ):
            pass
        overall = session.judge_scores.get("overall")
        if overall is not None:
            try:
                _langfuse_client.score_current_trace(
                    name="judge_overall", value=float(overall)
                )
            except Exception:
                pass


def _emit_legacy_spans(session: Session) -> None:
    with _langfuse_client.start_as_current_observation(
        name="retrieval",
        as_type="retriever",
        input={"query": session.query},
        output={"chunks": session.chunks_retrieved},
    ):
        pass

    with _langfuse_client.start_as_current_observation(
        name="llm-completion",
        as_type="generation",
        model=session.model_id,
        input=session.query,
        output=session.response,
        usage_details={
            "input": session.token_usage.get("prompt_tokens", 0),
            "output": session.token_usage.get("completion_tokens", 0),
            "total": session.token_usage.get("total_tokens", 0),
        },
    ):
        pass
```

- [ ] **Step 5: Run the full tracer test file**

Run: `uv run pytest tests/observability/test_tracer.py -v`
Expected: all passed (existing + 2 new)

- [ ] **Step 6: Commit**

```bash
git add src/medic_agent/observability/tracer.py tests/observability/test_tracer.py
git commit -m "feat(v2): nested multi-agent LangFuse spans + judge score; legacy path kept"
```

---

## Task 12: Eval runner — route through orchestrator + router accuracy

**Files:**
- Modify: `src/medic_agent/evaluation/runner.py`
- Test: `tests/agents/test_runner_router.py` (unit, mocked) + existing `tests/eval/test_eval.py` (integration)

- [ ] **Step 1: Write the failing unit test for router accuracy mapping**

```python
# tests/agents/test_runner_router.py
import medic_agent.evaluation.runner as runner_module
from medic_agent.agents.router import RouteDecision
from medic_agent.config.settings import USE_CASE_CODING
from medic_agent.evaluation.runner import EvalRunner


def test_router_check_maps_legacy_use_case(mocker):
    mocker.patch.object(
        runner_module,
        "route",
        return_value=RouteDecision(USE_CASE_CODING, "heuristic", 0.85, "kw"),
    )
    case = {"id": "c1", "use_case": "medical_coding", "input": "code this"}
    assert EvalRunner().run_router_check(case) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/agents/test_runner_router.py -v`
Expected: FAIL (`run_router_check` / `route` not present)

- [ ] **Step 3: Modify `runner.py`**

Replace the imports block and `run_layer3`/`run_all`, and add `run_router_check`. Specifically:

Change the top imports to:

```python
from medic_agent.agents.judge import run_judge
from medic_agent.agents.orchestrator import run as orchestrate
from medic_agent.agents.router import route
from medic_agent.config.settings import (
    DEFAULT_MODEL_ID,
    USE_CASE_AMBIENT,
    USE_CASE_CODING,
)
from medic_agent.rag.retriever import retrieve

EVAL_USE_CASE_MAP = {
    "medical_coding": USE_CASE_CODING,
    "ambient_note": USE_CASE_AMBIENT,
}
```

Remove the now-unused `from medic_agent.llm.client import ask` and the `_load_system_prompt` method and `_build_judge_prompt` function (logic now lives in `agents/judge.py`). Keep `JUDGE_MODEL` removed (judge model comes from settings via `agents.judge`).

Add `router_pass` to `EvalResult`:

```python
    router_pass: bool | None = None
```

Replace `run_layer3`:

```python
    def run_layer3(self, case: dict, response: str) -> tuple[dict | None, str, str]:
        criteria = case.get("judge_criteria", {})
        criteria_text = "\n".join(f"- {k}: {v}" for k, v in criteria.items())
        return run_judge(case["description"], response, criteria_text)
```

Add `run_router_check`:

```python
    def run_router_check(self, case: dict) -> bool:
        expected = EVAL_USE_CASE_MAP.get(case["use_case"])
        decision = route(case["input"], override="Auto")
        return decision.use_case == expected
```

Replace `run_all` so the response comes from the orchestrator (forcing the known agent so we evaluate the agent, not routing; judging disabled because Layer 3 does its own judging):

```python
    def run_all(self, cases: list[dict], layers: list[int]) -> list[EvalResult]:
        results = []
        for case in cases:
            display_uc = EVAL_USE_CASE_MAP.get(case["use_case"], USE_CASE_CODING)
            out = orchestrate(
                case["input"],
                DEFAULT_MODEL_ID,
                override=display_uc,
                judge_on=False,
            )
            response = out["response"]
            chunks = out["chunks"]
            result = EvalResult(
                case_id=case["id"],
                case_input=case["input"],
                case_description=case.get("description", ""),
                response=response,
                chunks=chunks,
            )
            if 1 in layers:
                result.layer1_pass = self.run_layer1(case, response)
                result.router_pass = self.run_router_check(case)
            if 2 in layers:
                result.ragas_scores = self.run_layer2(case, response, chunks)
            if 3 in layers:
                result.judge_scores, result.judge_raw, result.judge_prompt = (
                    self.run_layer3(case, response)
                )
            results.append(result)
        return results
```

- [ ] **Step 4: Run the unit test to verify it passes**

Run: `uv run pytest tests/agents/test_runner_router.py -v`
Expected: 1 passed

- [ ] **Step 5: Add a router-accuracy assertion to the integration eval suite**

Append to `tests/eval/test_eval.py`:

```python
def test_router_picks_correct_agent_for_golden_cases():
    """The Auto router must classify every golden case to its known use case."""
    cases = _load_cases()
    runner = EvalRunner()
    misses = [c["id"] for c in cases if not runner.run_router_check(c)]
    assert not misses, f"Router misclassified: {misses}"
```

- [ ] **Step 6: Run the default (non-eval) unit suite to confirm nothing broke**

Run: `uv run pytest -v`
Expected: all passed (eval tests are excluded by `-m 'not eval'`)

- [ ] **Step 7: Commit**

```bash
git add src/medic_agent/evaluation/runner.py tests/agents/test_runner_router.py tests/eval/test_eval.py
git commit -m "feat(v2): eval runs through orchestrator; shared judge; router-accuracy check"
```

---

## Task 13: UI — routing mode, routing panel, all-prompts Tab 2, Tab 3 columns

**Files:**
- Modify: `src/medic_agent/ui/app.py`

> No unit tests (Streamlit UI). Verified by running the app (Step 6). Keep edits surgical; the dark theme `_STYLES` block and helper functions are unchanged.

- [ ] **Step 1: Update imports**

In `app.py`, replace the `from medic_agent.llm.client import ask` and the use-case-related settings imports with:

```python
from medic_agent.agents.orchestrator import run as orchestrate
from medic_agent.config.prompts import PROMPT_DEFAULTS, load_all, save_all
from medic_agent.config.settings import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL_NAME,
    LANGFUSE_BASE_URL,
    LANGFUSE_PUBLIC_KEY,
    SESSIONS_DIR,
    USE_CASE_AMBIENT,
    USE_CASE_CODING,
)
```

Remove now-unused imports: `ask`, `Session`, `retrieve`, `AMBIENT_SYSTEM_PROMPT`, `CODING_SYSTEM_PROMPT`, `DEFAULT_USE_CASE`, `PROMPTS_FILE`, `USE_CASES`, and the `_load_prompts`/`_save_prompts`/`_push_prompts_to_langfuse` helpers (replaced by `config/prompts.py`).

- [ ] **Step 2: Replace `_render_sidebar` to return routing mode + judge toggle**

```python
def _render_sidebar() -> tuple[str, str, bool]:
    st.sidebar.markdown(
        """
<div style="padding:1.4rem 0 1.8rem">
  <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:800;
              color:#d8e8f8;letter-spacing:-.04em;line-height:1">
    medic<span style="color:#00c8b4">·</span>agent
  </div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:.6rem;color:#334f70;
              letter-spacing:.18em;text-transform:uppercase;margin-top:6px">
    Clinical AI &nbsp;·&nbsp; v2
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    routing_mode = st.sidebar.selectbox(
        "Routing Mode",
        options=["Auto", USE_CASE_CODING, USE_CASE_AMBIENT],
    )
    judge_on = st.sidebar.toggle("Score every response (judge)", value=True)

    st.sidebar.divider()

    docs = get_document_info()
    total_chunks = sum(d["chunk_count"] for d in docs)
    st.sidebar.caption(
        f"{'📄 ' if docs else ''}"
        f"{len(docs)} doc{'s' if len(docs) != 1 else ''} · {total_chunks} chunks"
    )
    st.sidebar.caption("→ Manage in **Knowledge Base** tab")

    model_name = st.sidebar.selectbox(
        "Model",
        options=list(AVAILABLE_MODELS.keys()),
        index=list(AVAILABLE_MODELS.keys()).index(DEFAULT_MODEL_NAME),
    )
    return routing_mode, AVAILABLE_MODELS[model_name], judge_on
```

- [ ] **Step 3: Replace `_render_agent_tab` to call the orchestrator and show the routing panel**

```python
def _render_agent_tab(routing_mode: str, model_id: str, judge_on: bool) -> None:
    st.markdown(
        """<div style="display:flex;align-items:center;gap:.5rem;margin-bottom:1.1rem">
  <div style="font-family:'Syne',sans-serif;font-size:1.35rem;font-weight:700;
              color:#d8e8f8;letter-spacing:-.02em">Clinical Agent</div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:.62rem;color:#334f70;
              background:#112038;border:1px solid #1c2e48;border-radius:3px;
              padding:.15em .45em;letter-spacing:.08em">orchestrated</div>
</div>""",
        unsafe_allow_html=True,
    )

    user_input = st.text_area(
        "Paste encounter documentation or a visit transcript",
        placeholder=(
            "Paste clinical documentation to code, OR a physician-patient "
            "conversation transcript to turn into a SOAP note. The orchestrator "
            "picks the right agent."
        ),
        height=200,
    )

    if st.button("Submit", type="primary", disabled=not user_input.strip()):
        with st.spinner("Routing & running agent…"):
            try:
                result = orchestrate(
                    user_input, model_id, override=routing_mode, judge_on=judge_on
                )
                st.session_state["last_result"] = result
            except RuntimeError as e:
                st.error(str(e))
                return

    result = st.session_state.get("last_result")
    if result:
        _render_routing_panel(result["route"])
        st.markdown(result["response"])

        chunks = result.get("chunks", [])
        if chunks:
            with st.expander(f"Sources used — {len(chunks)} chunk(s)"):
                for c in chunks:
                    st.caption(f"**{c['source_filename']}** — chunk {c['chunk_index']}")
                    st.text(c["text"][:300] + ("…" if len(c["text"]) > 300 else ""))

        scores = result.get("judge_scores") or {}
        if scores and "error" not in scores:
            with st.expander("LLM-as-judge scores"):
                for k, v in scores.items():
                    st.caption(f"`{k}`: {v}")


def _render_routing_panel(route: dict) -> None:
    if not route:
        return
    use_case = route.get("use_case", "?")
    method = route.get("method", "?")
    confidence = route.get("confidence", 0)
    reasoning = route.get("reasoning", "")
    st.markdown(
        f"""<div style="background:#0c1726;border:1px solid #1c2e48;border-radius:8px;
padding:.85rem 1.1rem;margin-bottom:1rem">
  <div style="font-family:'JetBrains Mono',monospace;font-size:.6rem;color:#334f70;
              letter-spacing:.12em;text-transform:uppercase;margin-bottom:.35rem">
    Routed to</div>
  <div style="font-family:'Syne',sans-serif;font-size:1.05rem;font-weight:700;
              color:#00c8b4">{use_case}</div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:.7rem;color:#6888aa;
              margin-top:.3rem">
    {method} · confidence {confidence} · {reasoning}</div>
</div>""",
        unsafe_allow_html=True,
    )
```

- [ ] **Step 4: Replace `_render_kb_prompts_section` to edit all prompts grouped by agent**

```python
def _render_kb_prompts_section() -> None:
    st.divider()
    _section_header("System Prompts")

    prompts = load_all()
    version = prompts.get("version", 0)
    if version:
        st.markdown(
            f"""<div style="font-family:'JetBrains Mono',monospace;font-size:.68rem;
color:#6888aa;margin-bottom:.85rem">active: <span style="color:#00c8b4">v{version}</span></div>""",
            unsafe_allow_html=True,
        )

    edited = {"router": "", "coding": {}, "ambient": {}, "judge": {}}

    _section_label("Router")
    edited["router"] = st.text_area(
        "Router classifier prompt", value=prompts["router"], height=160, key="p_router"
    )

    _section_label("Medical Coding Agent")
    for step in ("extract", "code", "verify"):
        edited["coding"][step] = st.text_area(
            f"Coding · {step}", value=prompts["coding"][step], height=180, key=f"p_coding_{step}"
        )

    _section_label("Ambient Note Taking Agent")
    for step in ("soap", "code", "verify"):
        edited["ambient"][step] = st.text_area(
            f"Ambient · {step}", value=prompts["ambient"][step], height=180, key=f"p_ambient_{step}"
        )

    _section_label("Judge Rubrics")
    for rubric in ("coding", "ambient"):
        edited["judge"][rubric] = st.text_area(
            f"Judge · {rubric}", value=prompts["judge"][rubric], height=160, key=f"p_judge_{rubric}"
        )

    col1, col2 = st.columns(2)
    if col1.button("Save Prompts", type="primary"):
        new_version = save_all(edited)
        st.success(f"Prompts saved as **v{new_version}**.")
        st.rerun()
    if col2.button("Reset to Defaults"):
        st.session_state["p_router"] = PROMPT_DEFAULTS["router"]
        for step in ("extract", "code", "verify"):
            st.session_state[f"p_coding_{step}"] = PROMPT_DEFAULTS["coding"][step]
        for step in ("soap", "code", "verify"):
            st.session_state[f"p_ambient_{step}"] = PROMPT_DEFAULTS["ambient"][step]
        for rubric in ("coding", "ambient"):
            st.session_state[f"p_judge_{rubric}"] = PROMPT_DEFAULTS["judge"][rubric]
        st.info("Defaults restored. Click **Save Prompts** to persist.")
```

- [ ] **Step 5: Add route + judge to Tab 3 rows and update `main()`**

In `_render_observability_tab`, inside the `for s in reversed(filtered):` expander body, add after the existing `st.caption(...)`:

```python
            route = s.get("route_decision", {})
            if route:
                st.caption(
                    f"Routed: {route.get('use_case','?')} "
                    f"({route.get('method','?')}, conf {route.get('confidence','?')})"
                )
            judge = s.get("judge_scores", {})
            if judge and "overall" in judge:
                st.caption(f"Judge overall: {judge['overall']}")
```

Update `main()` to thread the new sidebar return values:

```python
def main() -> None:
    st.set_page_config(
        page_title="medic-agent",
        page_icon="⬡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_STYLES, unsafe_allow_html=True)

    routing_mode, model_id, judge_on = _render_sidebar()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Agent", "Knowledge Base", "Observability", "Evaluation"]
    )
    with tab1:
        _render_agent_tab(routing_mode, model_id, judge_on)
    with tab2:
        _render_kb_upload_section()
        _render_kb_prompts_section()
    with tab3:
        _render_observability_tab()
    with tab4:
        _render_eval_tab()
```

- [ ] **Step 6: Verify the app runs end-to-end**

Run: `uv run streamlit run src/medic_agent/ui/app.py --server.headless true --server.port 8502`

Manually check (browser at `localhost:8502`):
- Sidebar shows **Routing Mode** (Auto/Coding/Ambient) + judge toggle, no use-case radio
- Paste a coding doc → Submit → routing panel shows **Medical Coding** + method + reasoning; response renders
- Paste a `Doctor:/Patient:` transcript with Auto → routes to **Ambient Note Taking**
- Force override to the other agent → routing panel shows `manual`
- Tab 2 shows all 9 prompt boxes grouped (Router / Coding ×3 / Ambient ×3 / Judge ×2); Save bumps version; Reset restores defaults
- Tab 3 shows routed use case + judge overall on session rows

- [ ] **Step 7: Commit**

```bash
git add src/medic_agent/ui/app.py
git commit -m "feat(v2): orchestrated Agent tab, routing panel, all-prompts Tab 2, Tab 3 route/judge"
```

---

## Task 14: Full-suite verification & acceptance

- [ ] **Step 1: Run the entire unit suite**

Run: `uv run pytest -v`
Expected: all passed (eval excluded by marker)

- [ ] **Step 2: Run the integration eval suite (real LLM calls)**

Run: `uv run pytest tests/eval/ -v -m eval`
Expected: Layer 1 passes for all golden cases; router classifies every case correctly. (Costs a few cents.)

- [ ] **Step 3: Check off V2 acceptance criteria in `CLAUDE.md`**

Edit the "V2 Acceptance Criteria" checklist — mark satisfied items `[x]`.

- [ ] **Step 4: Refresh the eval baseline**

Run the Eval tab (or the runner) with Layer 3, then click **Set as Baseline** so `tests/eval/baseline.json` reflects the new orchestrated pipeline.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md tests/eval/baseline.json
git commit -m "chore(v2): check off acceptance criteria; refresh eval baseline"
```

---

## Self-review notes (for the implementer)

- **Sequencing:** Task 11 (Session fields) must land before Task 10 Step 3 compiles, since the orchestrator constructs `Session(route_decision=…, agent_steps=…, judge_scores=…)`. If doing strict TDD order, implement Task 11 Step 1 (add fields) early.
- **`agent_steps` reducer:** the `Annotated[list, operator.add]` reducer is what lets each node return `{"agent_steps": [one_step]}` and have them concatenate. Without it, steps overwrite each other.
- **Router vs. agent model:** the agent uses the user-selected `model_id`; the router always uses Haiku and the judge always uses Sonnet (`ROUTER_MODEL_ID`, `JUDGE_MODEL_ID`).
- **Use-case key mismatch:** golden cases use `medical_coding`/`ambient_note`; everything in `agents/` uses the display names. `EVAL_USE_CASE_MAP` is the only bridge — do not leak legacy keys into `agents/`.
- **Legacy trace path:** `_emit_legacy_spans` keeps V1 sessions (no `agent_steps`) emitting the original 3 spans, so the existing tracer tests stay green.
