import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone

from medic_agent.agents.judge import run_judge
from medic_agent.agents.orchestrator import run as orchestrate
from medic_agent.agents.router import route
from medic_agent.config.settings import (
    DEFAULT_MODEL_ID,
    USE_CASE_AMBIENT,
    USE_CASE_CODING,
)

EVAL_USE_CASE_MAP = {
    "medical_coding": USE_CASE_CODING,
    "ambient_note": USE_CASE_AMBIENT,
}

# SOAP section detection — matches "SUBJECTIVE", "S — ...", "S:", "S -"
_SOAP_RE: dict[str, list[str]] = {
    "S": [r"SUBJECTIVE", r"(?m)^S\s*[—\-:]"],
    "O": [r"OBJECTIVE", r"(?m)^O\s*[—\-:]"],
    "A": [r"ASSESSMENT", r"(?m)^A\s*[—\-:]"],
    "P": [r"(?<!\w)PLAN\b", r"(?m)^P\s*[—\-:]"],
}


@dataclass
class EvalResult:
    case_id: str
    layer1_pass: bool = False
    router_pass: bool | None = None
    ragas_scores: dict | None = None
    judge_scores: dict | None = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    # provenance — populated by run_all() for use in the UI inspector
    case_input: str = ""
    case_description: str = ""
    response: str = ""
    chunks: list[dict] = field(default_factory=list)
    judge_prompt: str = ""
    judge_raw: str = ""


class EvalRunner:

    def _has_soap_section(self, section: str, response: str) -> bool:
        return any(
            re.search(p, response, re.IGNORECASE) for p in _SOAP_RE.get(section, [])
        )

    def run_layer1(self, case: dict, response: str) -> bool:
        """Structural/format checks — verifies the response has the right shape,
        not that specific codes are clinically correct (that is Layer 3's job).
        """
        use_case = case["use_case"]

        if use_case == "medical_coding":
            # Must contain at least one ICD-10-CM formatted code
            has_icd10 = bool(re.search(r'\b[A-Z]\d{2}\.?\w*\b', response))
            # Must contain at least one 5-digit CPT/HCPCS code
            has_cpt = bool(re.search(r'\b\d{5}\b', response))
            return has_icd10 and has_cpt

        if use_case == "ambient_note":
            # Must have all four SOAP section headers
            for section in case["expected"].get("soap_sections", []):
                if not self._has_soap_section(section, response):
                    return False
            # Must contain at least one ICD-10-CM formatted code
            return bool(re.search(r'\b[A-Z]\d{2}\.?\w*\b', response))

        return True

    def run_layer2(self, case: dict, response: str, chunks: list[dict]) -> dict | None:
        try:
            import warnings as _w
            import instructor
            import litellm as _litellm
            from ragas import evaluate
            from ragas.metrics import Faithfulness as _Faithfulness
            from ragas.llms.litellm_llm import LiteLLMStructuredLLM
            from ragas.dataset_schema import EvaluationDataset, SingleTurnSample

            _w.filterwarnings("ignore", category=DeprecationWarning, module="ragas")

            client = instructor.from_litellm(_litellm.completion)
            llm = LiteLLMStructuredLLM(
                client=client, model=DEFAULT_MODEL_ID, provider="anthropic"
            )
            metric = _Faithfulness()
            metric.llm = llm

            ctx = [c["text"] for c in chunks] if chunks else [case["input"][:500]]
            sample = SingleTurnSample(
                user_input=case["input"][:500],
                retrieved_contexts=ctx,
                response=response,
            )
            dataset = EvaluationDataset(samples=[sample])
            result = evaluate(dataset, metrics=[metric], show_progress=False)
            scores = result["faithfulness"]
            score = float(scores[0]) if isinstance(scores, list) else float(scores)
            return {"faithfulness": score}
        except Exception as e:
            warnings.warn(f"RAGAS layer 2 failed for {case['id']}: {e}")
            return {"error": str(e)}

    def run_layer3(self, case: dict, response: str) -> tuple[dict | None, str, str]:
        """Returns (scores_dict, raw_llm_response, prompt_sent)."""
        criteria = case.get("judge_criteria", {})
        criteria_text = "\n".join(f"- {k}: {v}" for k, v in criteria.items())
        return run_judge(case["description"], response, criteria_text)

    def run_router_check(self, case: dict) -> bool:
        expected = EVAL_USE_CASE_MAP.get(case["use_case"])
        decision = route(case["input"], override="Auto")
        return decision.use_case == expected

    def run_all(self, cases: list[dict], layers: list[int]) -> list[EvalResult]:
        results = []
        for case in cases:
            display_uc = EVAL_USE_CASE_MAP.get(case["use_case"], USE_CASE_CODING)
            out = orchestrate(
                case["input"],
                DEFAULT_MODEL_ID,
                override=display_uc,
                judge_on=False,
            )
            response = out["response"]
            chunks = out["chunks"]
            result = EvalResult(
                case_id=case["id"],
                case_input=case["input"],
                case_description=case.get("description", ""),
                response=response,
                chunks=chunks,
            )
            if 1 in layers:
                result.layer1_pass = self.run_layer1(case, response)
                result.router_pass = self.run_router_check(case)
            if 2 in layers:
                result.ragas_scores = self.run_layer2(case, response, chunks)
            if 3 in layers:
                result.judge_scores, result.judge_raw, result.judge_prompt = (
                    self.run_layer3(case, response)
                )
            results.append(result)
        return results
