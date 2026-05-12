import json
import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone

from medic_agent.config.settings import (
    AMBIENT_SYSTEM_PROMPT,
    CODING_SYSTEM_PROMPT,
    DEFAULT_MODEL_ID,
    PROMPTS_FILE,
)
from medic_agent.llm.client import ask
from medic_agent.rag.retriever import retrieve

JUDGE_MODEL = "claude-sonnet-4-6"

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
    ragas_scores: dict | None = None
    judge_scores: dict | None = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class EvalRunner:

    def _load_system_prompt(self, use_case: str) -> str:
        if PROMPTS_FILE.exists():
            try:
                data = json.loads(PROMPTS_FILE.read_text())
                key = "coding" if use_case == "medical_coding" else "ambient"
                return data.get(key, "")
            except Exception:
                pass
        return CODING_SYSTEM_PROMPT if use_case == "medical_coding" else AMBIENT_SYSTEM_PROMPT

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

    def run_layer3(self, case: dict, response: str) -> dict | None:
        criteria = case.get("judge_criteria", {})
        criteria_text = "\n".join(f"- {k}: {v}" for k, v in criteria.items())
        prompt = _build_judge_prompt(case["description"], response, criteria_text)
        try:
            raw = ask(
                model_id=JUDGE_MODEL,
                system_prompt=(
                    "You are a medical coding evaluation expert. "
                    "Return only valid JSON with no markdown."
                ),
                user_query=prompt,
            )
            start, end = raw.find("{"), raw.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(raw[start:end])
        except Exception as e:
            warnings.warn(f"LLM-as-judge failed for {case['id']}: {e}")
            return {"error": str(e)}
        return None

    def run_all(self, cases: list[dict], layers: list[int]) -> list[EvalResult]:
        results = []
        for case in cases:
            chunks = retrieve(case["input"], k=5)
            response = ask(
                model_id=DEFAULT_MODEL_ID,
                system_prompt=self._load_system_prompt(case["use_case"]),
                user_query=case["input"],
                context=chunks or None,
            )
            result = EvalResult(case_id=case["id"])
            if 1 in layers:
                result.layer1_pass = self.run_layer1(case, response)
            if 2 in layers:
                result.ragas_scores = self.run_layer2(case, response, chunks)
            if 3 in layers:
                result.judge_scores = self.run_layer3(case, response)
            results.append(result)
        return results


def _build_judge_prompt(description: str, response: str, criteria_text: str) -> str:
    criteria_keys = [
        line.split(":")[0].lstrip("- ").strip()
        for line in criteria_text.splitlines()
        if ":" in line
    ]
    schema = "{" + ", ".join(f'"{k}": <1-5>' for k in criteria_keys) + ', "overall": <1.0-5.0>}'
    return (
        f"Case: {description}\n\n"
        f"Evaluation criteria:\n{criteria_text}\n\n"
        f"Response to evaluate (first 1500 chars):\n{response[:1500]}\n\n"
        f"Score each criterion 1-5 and provide a weighted overall score 1.0-5.0.\n"
        f"Return ONLY this JSON (no markdown, no explanation):\n{schema}"
    )
