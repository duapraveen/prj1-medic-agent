from unittest.mock import MagicMock

import pytest
from litellm.exceptions import AuthenticationError, BadRequestError, APIConnectionError, RateLimitError

from medic_agent.llm.client import ask


def _mock_response(content: str) -> MagicMock:
    response = MagicMock()
    response.choices[0].message.content = content
    return response


# --- Happy path ---

def test_ask_returns_response_content(mocker):
    mocker.patch("medic_agent.llm.client.completion", return_value=_mock_response("Aspirin reduces fever."))
    result = ask("claude-haiku-4-5-20251001", "You are helpful.", "What is aspirin?")
    assert result == "Aspirin reduces fever."


def test_ask_passes_correct_model(mocker):
    mock_completion = mocker.patch("medic_agent.llm.client.completion", return_value=_mock_response("ok"))
    ask("claude-sonnet-4-6", "You are helpful.", "What is aspirin?")
    call_kwargs = mock_completion.call_args
    assert call_kwargs.kwargs["model"] == "claude-sonnet-4-6"


def test_ask_includes_system_and_user_messages(mocker):
    mock_completion = mocker.patch("medic_agent.llm.client.completion", return_value=_mock_response("ok"))
    ask("claude-haiku-4-5-20251001", "Be concise.", "What is aspirin?")
    messages = mock_completion.call_args.kwargs["messages"]
    roles = [m["role"] for m in messages]
    assert "system" in roles
    assert "user" in roles


def test_ask_system_message_has_cache_control(mocker):
    mock_completion = mocker.patch("medic_agent.llm.client.completion", return_value=_mock_response("ok"))
    ask("claude-haiku-4-5-20251001", "Be concise.", "What is aspirin?")
    messages = mock_completion.call_args.kwargs["messages"]
    system_msg = next(m for m in messages if m["role"] == "system")
    assert system_msg["content"][0]["cache_control"] == {"type": "ephemeral"}


# --- Input validation ---

def test_ask_raises_on_empty_query(mocker):
    mocker.patch("medic_agent.llm.client.completion")
    with pytest.raises(ValueError, match="Query cannot be empty"):
        ask("claude-haiku-4-5-20251001", "You are helpful.", "")


def test_ask_raises_on_whitespace_query(mocker):
    mocker.patch("medic_agent.llm.client.completion")
    with pytest.raises(ValueError, match="Query cannot be empty"):
        ask("claude-haiku-4-5-20251001", "You are helpful.", "   ")


# --- Error handling ---

def test_ask_maps_authentication_error(mocker):
    mocker.patch("medic_agent.llm.client.completion", side_effect=AuthenticationError("bad key", llm_provider="anthropic", model="claude-haiku-4-5-20251001"))
    with pytest.raises(RuntimeError, match="Invalid API key"):
        ask("claude-haiku-4-5-20251001", "You are helpful.", "What is aspirin?")


def test_ask_maps_rate_limit_error(mocker):
    mocker.patch("medic_agent.llm.client.completion", side_effect=RateLimitError("limit", llm_provider="anthropic", model="claude-haiku-4-5-20251001"))
    with pytest.raises(RuntimeError, match="Rate limit"):
        ask("claude-haiku-4-5-20251001", "You are helpful.", "What is aspirin?")


def test_ask_maps_connection_error(mocker):
    mocker.patch("medic_agent.llm.client.completion", side_effect=APIConnectionError("no connection", llm_provider="anthropic", model="claude-haiku-4-5-20251001"))
    with pytest.raises(RuntimeError, match="connect"):
        ask("claude-haiku-4-5-20251001", "You are helpful.", "What is aspirin?")


def test_ask_maps_bad_request_error(mocker):
    mocker.patch("medic_agent.llm.client.completion", side_effect=BadRequestError("bad", llm_provider="anthropic", model="claude-haiku-4-5-20251001"))
    with pytest.raises(RuntimeError, match="API rejected"):
        ask("claude-haiku-4-5-20251001", "You are helpful.", "What is aspirin?")
