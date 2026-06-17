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


def _sub_input(state: AgentState) -> dict:
    # Strip parent agent_steps so the subgraph does not return them back and
    # cause the parent's reducer to concatenate (and duplicate) them.
    return {k: v for k, v in state.items() if k != "agent_steps"}


def _run_coding(state: AgentState) -> dict:
    return _CODING_AGENT.invoke(_sub_input(state))


def _run_ambient(state: AgentState) -> dict:
    return _AMBIENT_AGENT.invoke(_sub_input(state))


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
        "total_tokens": sum(
            s.get("token_usage", {}).get("total_tokens", 0) for s in steps
        )
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
