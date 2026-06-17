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


def test_complete_does_not_log_session(mocker):
    mocker.patch.object(
        client_module, "completion", return_value=_mock_response("ok")
    )
    spy = mocker.patch.object(client_module, "log_session")
    complete("claude-haiku-4-5-20251001", "sys", "q")
    spy.assert_not_called()
