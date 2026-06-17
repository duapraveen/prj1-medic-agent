import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import medic_agent.observability.tracer as tracer_module
from medic_agent.observability.tracer import Session, log_session


def _make_session(**overrides) -> Session:
    defaults = dict(
        use_case="Medical Coding",
        model_id="claude-haiku-4-5-20251001",
        query="What are the ICD-10 codes?",
        response="E11.65 — Type 2 diabetes with hyperglycemia.",
        system_prompt_version="v1",
        chunks_retrieved=[{"text": "HbA1c 9.2%", "source_filename": "note.pdf", "chunk_index": 0}],
        latency_ms=1234.5,
        token_usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        error=None,
    )
    defaults.update(overrides)
    return Session(**defaults)


# --- Session dataclass ---

def test_session_has_auto_session_id():
    s = _make_session()
    assert s.session_id and len(s.session_id) == 36  # UUID format


def test_session_has_auto_timestamp():
    s = _make_session()
    assert s.timestamp and "T" in s.timestamp  # ISO 8601


def test_two_sessions_get_different_ids():
    s1 = _make_session()
    s2 = _make_session()
    assert s1.session_id != s2.session_id


# --- Local file writing ---

def test_log_session_writes_jsonl(tmp_path, mocker):
    mocker.patch.object(tracer_module, "SESSIONS_DIR", tmp_path)
    mocker.patch.object(tracer_module, "_LANGFUSE_ENABLED", False)

    session = _make_session()
    log_session(session)

    log_file = tmp_path / "sessions.jsonl"
    assert log_file.exists()
    lines = log_file.read_text().strip().splitlines()
    assert len(lines) == 1


def test_log_session_jsonl_contains_correct_fields(tmp_path, mocker):
    mocker.patch.object(tracer_module, "SESSIONS_DIR", tmp_path)
    mocker.patch.object(tracer_module, "_LANGFUSE_ENABLED", False)

    session = _make_session()
    log_session(session)

    record = json.loads((tmp_path / "sessions.jsonl").read_text())
    assert record["use_case"] == "Medical Coding"
    assert record["model_id"] == "claude-haiku-4-5-20251001"
    assert record["query"] == "What are the ICD-10 codes?"
    assert record["latency_ms"] == 1234.5
    assert record["token_usage"]["total_tokens"] == 150
    assert record["system_prompt_version"] == "v1"
    assert len(record["chunks_retrieved"]) == 1


def test_log_session_appends_multiple_sessions(tmp_path, mocker):
    mocker.patch.object(tracer_module, "SESSIONS_DIR", tmp_path)
    mocker.patch.object(tracer_module, "_LANGFUSE_ENABLED", False)

    log_session(_make_session(query="first query"))
    log_session(_make_session(query="second query"))

    lines = (tmp_path / "sessions.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2


# --- LangFuse integration ---

def test_log_session_calls_langfuse_when_enabled(tmp_path, mocker):
    mocker.patch.object(tracer_module, "SESSIONS_DIR", tmp_path)
    mocker.patch.object(tracer_module, "_LANGFUSE_ENABLED", True)

    mock_send = mocker.patch.object(tracer_module, "_send_to_langfuse")
    log_session(_make_session())
    mock_send.assert_called_once()


def test_log_session_skips_langfuse_when_disabled(tmp_path, mocker):
    mocker.patch.object(tracer_module, "SESSIONS_DIR", tmp_path)
    mocker.patch.object(tracer_module, "_LANGFUSE_ENABLED", False)

    mock_send = mocker.patch.object(tracer_module, "_send_to_langfuse")
    log_session(_make_session())
    mock_send.assert_not_called()


def test_send_to_langfuse_creates_trace_span_generation(mocker):
    mock_client = MagicMock()
    mocker.patch.object(tracer_module, "_langfuse_client", mock_client)

    tracer_module._send_to_langfuse(_make_session())

    # Called 3 times: outer pipeline span, retrieval, llm-completion
    assert mock_client.start_as_current_observation.call_count == 3
    mock_client.flush.assert_called_once()


def test_send_to_langfuse_passes_session_id_as_trace_id(mocker):
    mock_client = MagicMock()
    mocker.patch.object(tracer_module, "_langfuse_client", mock_client)

    session = _make_session()
    tracer_module._send_to_langfuse(session)

    # First call carries the trace_context with the session ID (dashes stripped for OTel format)
    first_call_kwargs = mock_client.start_as_current_observation.call_args_list[0].kwargs
    assert first_call_kwargs["trace_context"]["trace_id"] == session.session_id.replace("-", "")


def test_send_to_langfuse_warns_on_error(mocker):
    mock_client = MagicMock()
    mock_client.start_as_current_observation.side_effect = RuntimeError("network error")
    mocker.patch.object(tracer_module, "_langfuse_client", mock_client)

    with pytest.warns(UserWarning, match="LangFuse trace failed"):
        tracer_module._send_to_langfuse(_make_session())


# --- V2 multi-agent nested spans ---

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
