import operator
import time
from typing import Annotated, TypedDict

from medic_agent.llm.client import complete


class AgentState(TypedDict, total=False):
    query: str
    model_id: str
    override: str
    judge_on: bool
    use_case: str
    route: dict
    chunks: list[dict]
    scratch: dict
    response: str
    judge_scores: dict
    agent_steps: Annotated[list[dict], operator.add]


def llm_step(
    model_id: str,
    name: str,
    system_prompt: str,
    user_query: str,
    context: list[dict] | None = None,
    as_type: str = "generation",
) -> tuple[str, dict]:
    start = time.monotonic()
    text, usage = complete(model_id, system_prompt, user_query, context)
    latency = round((time.monotonic() - start) * 1000, 1)
    step = {
        "name": name,
        "type": as_type,
        "model_id": model_id,
        "token_usage": usage,
        "latency_ms": latency,
        "input_summary": (user_query or "")[:200],
        "output_summary": (text or "")[:200],
    }
    return text, step
