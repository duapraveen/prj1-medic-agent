import json
from datetime import datetime, timezone

from medic_agent.config.settings import (
    AMBIENT_CODE_PROMPT,
    AMBIENT_JUDGE_PROMPT,
    AMBIENT_SYSTEM_PROMPT,
    AMBIENT_VERIFY_PROMPT,
    CODING_EXTRACT_PROMPT,
    CODING_JUDGE_PROMPT,
    CODING_SYSTEM_PROMPT,
    CODING_VERIFY_PROMPT,
    PROMPTS_FILE,
    ROUTER_SYSTEM_PROMPT,
)

PROMPT_DEFAULTS: dict = {
    "router": ROUTER_SYSTEM_PROMPT,
    "coding": {
        "extract": CODING_EXTRACT_PROMPT,
        "code": CODING_SYSTEM_PROMPT,
        "verify": CODING_VERIFY_PROMPT,
    },
    "ambient": {
        "soap": AMBIENT_SYSTEM_PROMPT,
        "code": AMBIENT_CODE_PROMPT,
        "verify": AMBIENT_VERIFY_PROMPT,
    },
    "judge": {
        "coding": CODING_JUDGE_PROMPT,
        "ambient": AMBIENT_JUDGE_PROMPT,
    },
}


def _read_file() -> dict:
    if PROMPTS_FILE.exists():
        try:
            return json.loads(PROMPTS_FILE.read_text())
        except Exception:
            pass
    return {}


def get_prompt(*keys: str) -> str:
    node = _read_file()
    default = PROMPT_DEFAULTS
    for key in keys:
        node = node.get(key, {}) if isinstance(node, dict) else {}
        default = default[key]
    if isinstance(node, str) and node:
        return node
    return default


def load_all() -> dict:
    stored = _read_file()
    merged = json.loads(json.dumps(PROMPT_DEFAULTS))
    for group, val in PROMPT_DEFAULTS.items():
        if isinstance(val, dict):
            merged[group] = {**val, **stored.get(group, {})}
        else:
            merged[group] = stored.get(group, val)
    merged["version"] = stored.get("version", 0)
    return merged


def save_all(prompts: dict) -> int:
    existing = _read_file()
    version = existing.get("version", 0) + 1
    payload = {
        "version": version,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "router": prompts["router"],
        "coding": prompts["coding"],
        "ambient": prompts["ambient"],
        "judge": prompts["judge"],
    }
    PROMPTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROMPTS_FILE.write_text(json.dumps(payload, indent=2))
    _push_to_langfuse(payload, version)
    return version


def _push_to_langfuse(payload: dict, version: int) -> None:
    try:
        from medic_agent.observability.tracer import (
            _LANGFUSE_ENABLED,
            _langfuse_client,
        )

        if not _LANGFUSE_ENABLED or _langfuse_client is None:
            return
        flat = {
            "router": payload["router"],
            "coding-code": payload["coding"]["code"],
            "ambient-soap": payload["ambient"]["soap"],
        }
        for name, text in flat.items():
            _langfuse_client.create_prompt(
                name=f"medic-{name}-prompt", prompt=text, labels=["production"]
            )
    except Exception:
        pass
