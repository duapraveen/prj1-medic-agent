import medic_agent.agents.judge as judge_module
from medic_agent.agents.judge import build_judge_prompt, judge_output, run_judge


def test_build_judge_prompt_includes_schema_keys():
    criteria = "- code_accuracy: x\n- hallucination: y"
    prompt = build_judge_prompt("a coding case", "RESPONSE", criteria)
    assert '"code_accuracy"' in prompt and '"hallucination"' in prompt
    assert '"overall"' in prompt


def test_run_judge_parses_embedded_json(mocker):
    raw = 'Here you go: {"code_accuracy": 4, "overall": 4.2} done'
    mocker.patch.object(judge_module, "complete", return_value=(raw, {}))
    scores, returned_raw, prompt = run_judge("desc", "resp", "- code_accuracy: x")
    assert scores == {"code_accuracy": 4, "overall": 4.2}
    assert returned_raw == raw


def test_judge_output_uses_rubric_for_use_case(mocker):
    mocker.patch.object(
        judge_module, "complete", return_value=('{"overall": 5.0}', {})
    )
    scores = judge_output("Medical Coding", "some response")
    assert scores == {"overall": 5.0}
