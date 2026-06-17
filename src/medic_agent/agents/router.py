import json
import re
from dataclasses import dataclass

from medic_agent.config.settings import (
    ROUTER_MODEL_ID,
    ROUTER_SYSTEM_PROMPT,
    USE_CASE_AMBIENT,
    USE_CASE_CODING,
)
from medic_agent.llm.client import complete

_SPEAKER_RE = re.compile(
    r"(?im)^\s*(doctor|patient|dr|pt|physician|provider|nurse)\s*[:>\-]"
)
_CODING_KW_RE = re.compile(
    r"(?i)\b(icd[-\s]?10|cpt|hcpcs|e/?m\s+code|billing\s+code|what\s+codes|code\s+this)\b"
)


@dataclass
class RouteDecision:
    use_case: str
    method: str  # "manual" | "heuristic" | "llm"
    confidence: float
    reasoning: str


def heuristic_route(query: str) -> RouteDecision | None:
    speaker_turns = len(_SPEAKER_RE.findall(query))
    if speaker_turns >= 2:
        return RouteDecision(
            USE_CASE_AMBIENT,
            "heuristic",
            0.9,
            f"Detected {speaker_turns} dialogue speaker turns",
        )
    if _CODING_KW_RE.search(query):
        return RouteDecision(
            USE_CASE_CODING, "heuristic", 0.85, "Matched coding keyword (ICD/CPT/billing)"
        )
    return None


def llm_route(query: str) -> RouteDecision:
    text, _ = complete(ROUTER_MODEL_ID, ROUTER_SYSTEM_PROMPT, query)
    use_case, confidence, reasoning = USE_CASE_CODING, 0.5, ""
    start, end = text.find("{"), text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start:end])
            raw = str(data.get("use_case", "")).lower()
            use_case = USE_CASE_AMBIENT if "ambient" in raw else USE_CASE_CODING
            confidence = float(data.get("confidence", 0.5))
            reasoning = str(data.get("reasoning", ""))
        except Exception:
            pass
    return RouteDecision(use_case, "llm", confidence, reasoning)


def route(query: str, override: str = "Auto") -> RouteDecision:
    if override in (USE_CASE_CODING, USE_CASE_AMBIENT):
        return RouteDecision(override, "manual", 1.0, "User override")
    decision = heuristic_route(query)
    if decision is not None:
        return decision
    return llm_route(query)
