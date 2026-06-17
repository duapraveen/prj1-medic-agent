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
