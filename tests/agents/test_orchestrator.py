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
