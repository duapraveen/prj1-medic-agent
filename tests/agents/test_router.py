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
