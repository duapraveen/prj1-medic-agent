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
