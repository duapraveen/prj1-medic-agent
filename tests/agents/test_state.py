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
