import time

from langgraph.graph import END, START, StateGraph

from medic_agent.agents.state import AgentState, llm_step
from medic_agent.config.prompts import get_prompt
from medic_agent.rag.retriever import retrieve


def retrieve_node(state: AgentState) -> dict:
    start = time.monotonic()
    chunks = retrieve(state["query"], k=5)
    step = {
        "name": "retrieval",
        "type": "retriever",
        "model_id": "",
        "token_usage": {},
        "latency_ms": round((time.monotonic() - start) * 1000, 1),
        "input_summary": state["query"][:200],
        "output_summary": f"{len(chunks)} chunks",
    }
    return {"chunks": chunks, "agent_steps": [step]}


def soap(state: AgentState) -> dict:
    text, step = llm_step(
        state["model_id"],
        "soap",
        get_prompt("ambient", "soap"),
        state["query"],
        context=state.get("chunks"),
    )
    scratch = {**state.get("scratch", {}), "soap": text}
    return {"scratch": scratch, "agent_steps": [step]}


def code(state: AgentState) -> dict:
    soap_note = state.get("scratch", {}).get("soap", "")
    text, step = llm_step(
        state["model_id"], "code", get_prompt("ambient", "code"), soap_note
    )
    scratch = {**state.get("scratch", {}), "codes": text}
    return {"scratch": scratch, "agent_steps": [step]}


def verify(state: AgentState) -> dict:
    sc = state.get("scratch", {})
    user = (
        f"SOAP NOTE:\n{sc.get('soap', '')}\n\n"
        f"BILLING CODES:\n{sc.get('codes', '')}\n\n"
        "Combine into the final output and add documentation flags."
    )
    text, step = llm_step(
        state["model_id"], "verify", get_prompt("ambient", "verify"), user
    )
    return {"response": text, "agent_steps": [step]}


def build_ambient_agent():
    g = StateGraph(AgentState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("soap", soap)
    g.add_node("code", code)
    g.add_node("verify", verify)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "soap")
    g.add_edge("soap", "code")
    g.add_edge("code", "verify")
    g.add_edge("verify", END)
    return g.compile()
