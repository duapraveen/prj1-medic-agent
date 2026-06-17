import json

import medic_agent.config.prompts as prompts_module
from medic_agent.config.prompts import get_prompt, load_all, save_all, PROMPT_DEFAULTS


def test_get_prompt_falls_back_to_defaults(tmp_path, mocker):
    mocker.patch.object(prompts_module, "PROMPTS_FILE", tmp_path / "prompts.json")
    assert get_prompt("router") == PROMPT_DEFAULTS["router"]
    assert get_prompt("coding", "extract") == PROMPT_DEFAULTS["coding"]["extract"]


def test_save_then_get_prompt_round_trips(tmp_path, mocker):
    pf = tmp_path / "prompts.json"
    mocker.patch.object(prompts_module, "PROMPTS_FILE", pf)
    mocker.patch.object(prompts_module, "_push_to_langfuse", lambda *a, **k: None)

    edited = json.loads(json.dumps(PROMPT_DEFAULTS))
    edited["coding"]["verify"] = "CUSTOM VERIFY"
    version = save_all(edited)

    assert version == 1
    assert get_prompt("coding", "verify") == "CUSTOM VERIFY"
    assert get_prompt("router") == PROMPT_DEFAULTS["router"]  # untouched key still default


def test_missing_nested_key_uses_default(tmp_path, mocker):
    pf = tmp_path / "prompts.json"
    pf.write_text(json.dumps({"version": 1, "coding": {"extract": "X"}}))
    mocker.patch.object(prompts_module, "PROMPTS_FILE", pf)
    assert get_prompt("coding", "extract") == "X"
    assert get_prompt("coding", "verify") == PROMPT_DEFAULTS["coding"]["verify"]
