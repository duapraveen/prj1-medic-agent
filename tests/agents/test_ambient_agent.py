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
