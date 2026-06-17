import re
import time

from langgraph.graph import END, START, StateGraph

from medic_agent.agents.state import AgentState, llm_step
from medic_agent.config.prompts import get_prompt
from medic_agent.rag.retriever import graph_retrieve, retrieve


def extract(state: AgentState) -> dict:
    text, step = llm_step(
        state["model_id"], "extract", get_prompt("coding", "extract"), state["query"]
    )
    scratch = {**state.get("scratch", {}), "entities": text}
    return {"scratch": scratch, "agent_steps": [step]}


def _parse_entity_texts(entities_text: str) -> list[str]:
    return [
        m.group(1).strip()
        for m in re.finditer(r"^[-]\s+(.+)$", entities_text, re.MULTILINE)
    ]


def retrieve_node(state: AgentState) -> dict:
    entities_text = state.get("scratch", {}).get("entities", "")
    start = time.monotonic()
    vector_chunks = retrieve(f"{state['query']}\n{entities_text}", k=5)
    entity_texts = _parse_entity_texts(entities_text)
    graph_chunks = graph_retrieve(entity_texts)
    chunks = vector_chunks + graph_chunks
    step = {
        "name": "retrieval",
        "type": "retriever",
        "model_id": "",
        "token_usage": {},
        "latency_ms": round((time.monotonic() - start) * 1000, 1),
        "input_summary": state["query"][:200],
        "output_summary": f"{len(vector_chunks)} vector + {len(graph_chunks)} graph chunks",
    }
    return {"chunks": chunks, "agent_steps": [step]}


def code(state: AgentState) -> dict:
    text, step = llm_step(
        state["model_id"],
        "code",
        get_prompt("coding", "code"),
        state["query"],
        context=state.get("chunks"),
    )
    scratch = {**state.get("scratch", {}), "draft": text}
    return {"scratch": scratch, "agent_steps": [step]}


def verify(state: AgentState) -> dict:
    draft = state.get("scratch", {}).get("draft", "")
    user = f"DRAFT coding output:\n{draft}\n\nReview against the documentation and finalize."
    text, step = llm_step(
        state["model_id"],
        "verify",
        get_prompt("coding", "verify"),
        user,
        context=state.get("chunks"),
    )
    return {"response": text, "agent_steps": [step]}


def build_coding_agent():
    g = StateGraph(AgentState)
    g.add_node("extract", extract)
    g.add_node("retrieve", retrieve_node)
    g.add_node("code", code)
    g.add_node("verify", verify)
    g.add_edge(START, "extract")
    g.add_edge("extract", "retrieve")
    g.add_edge("retrieve", "code")
    g.add_edge("code", "verify")
    g.add_edge("verify", END)
    return g.compile()
