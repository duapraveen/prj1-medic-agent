import json
import warnings

from medic_agent.config.prompts import get_prompt
from medic_agent.config.settings import JUDGE_MODEL_ID, USE_CASE_AMBIENT
from medic_agent.llm.client import complete

JUDGE_SYSTEM_PROMPT = (
    "You are a medical documentation evaluation expert. "
    "Return only valid JSON with no markdown."
)


def build_judge_prompt(description: str, response: str, criteria_text: str) -> str:
    criteria_keys = [
        line.split(":")[0].lstrip("- ").strip()
        for line in criteria_text.splitlines()
        if ":" in line
    ]
    schema = (
        "{"
        + ", ".join(f'"{k}": <1-5>' for k in criteria_keys)
        + ', "overall": <1.0-5.0>}'
    )
    return (
        f"Case: {description}\n\n"
        f"Evaluation criteria:\n{criteria_text}\n\n"
        f"Response to evaluate (first 1500 chars):\n{response[:1500]}\n\n"
        f"Score each criterion 1-5 and provide a weighted overall score 1.0-5.0.\n"
        f"Return ONLY this JSON (no markdown, no explanation):\n{schema}"
    )


def run_judge(
    description: str,
    response: str,
    criteria_text: str,
    model_id: str = JUDGE_MODEL_ID,
) -> tuple[dict | None, str, str]:
    prompt = build_judge_prompt(description, response, criteria_text)
    try:
        raw, _ = complete(model_id, JUDGE_SYSTEM_PROMPT, prompt)
        start, end = raw.find("{"), raw.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(raw[start:end]), raw, prompt
    except Exception as e:
        warnings.warn(f"LLM-as-judge failed: {e}")
        return {"error": str(e)}, "", prompt
    return None, "", prompt


def judge_output(use_case: str, response: str, model_id: str = JUDGE_MODEL_ID) -> dict:
    rubric_key = "ambient" if use_case == USE_CASE_AMBIENT else "coding"
    criteria_text = get_prompt("judge", rubric_key)
    scores, _, _ = run_judge(
        f"{use_case} response", response, criteria_text, model_id
    )
    return scores or {}
