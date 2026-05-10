"""
Integration eval tests — make real LLM calls.
Run explicitly:  uv run pytest tests/eval/ -v -m eval
NOT included in the default pytest run (filtered out by pyproject.toml addopts).
"""
import json
from pathlib import Path

import pytest

from medic_agent.evaluation.runner import EvalResult, EvalRunner

pytestmark = pytest.mark.eval

GOLDEN_CASES_PATH = Path("tests/eval/golden_cases.json")
BASELINE_PATH = Path("tests/eval/baseline.json")


def _load_cases() -> list[dict]:
    return json.loads(GOLDEN_CASES_PATH.read_text())["cases"]


def test_layer1_passes_all_golden_cases():
    """All 5 golden cases must pass Layer 1 deterministic checks."""
    cases = _load_cases()
    runner = EvalRunner()
    results = runner.run_all(cases, layers=[1])

    failures = [r.case_id for r in results if not r.layer1_pass]
    assert not failures, f"Layer 1 failed for: {failures}"


def test_run_all_returns_one_result_per_case():
    cases = _load_cases()
    runner = EvalRunner()
    results = runner.run_all(cases, layers=[1])
    assert len(results) == len(cases)
    assert all(isinstance(r, EvalResult) for r in results)


def test_eval_results_have_required_fields():
    cases = _load_cases()[:1]  # one case is enough
    runner = EvalRunner()
    results = runner.run_all(cases, layers=[1])
    r = results[0]
    assert r.case_id == cases[0]["id"]
    assert isinstance(r.layer1_pass, bool)
    assert r.timestamp


def test_no_judge_score_regression_vs_baseline():
    """No case's judge overall score may drop >0.5 vs baseline (Layer 3)."""
    if not BASELINE_PATH.exists():
        pytest.skip("No baseline.json — run eval and click 'Set as Baseline' first.")
    baseline = json.loads(BASELINE_PATH.read_text())
    cases = _load_cases()
    runner = EvalRunner()
    results = runner.run_all(cases, layers=[3])
    regressions = []
    for r in results:
        if r.case_id not in baseline:
            continue
        current = (r.judge_scores or {}).get("overall", 0)
        base_val = baseline[r.case_id].get("judge_overall", 0)
        if current < base_val - 0.5:
            regressions.append(f"{r.case_id}: {current:.1f} vs baseline {base_val:.1f}")
    assert not regressions, f"Judge score regressions detected: {regressions}"
