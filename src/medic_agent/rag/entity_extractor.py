import json
import re

from medic_agent.config.settings import ENTITY_EXTRACTOR_MODEL_ID
from medic_agent.llm.client import complete

_SYSTEM_PROMPT = """\
Extract medical entities from clinical text.
Return ONLY valid JSON (no markdown):
{"entities": [{"text": "...", "entity_type": "Diagnosis|Medication|Procedure|Finding|Anatomy|Provider"}]}
Extract only explicitly documented entities. Do not infer. Return {"entities": []} if none found.\
"""

_MAX_INPUT_CHARS = 2000


def extract_entities(chunk_text: str) -> list[dict]:
    text, _ = complete(ENTITY_EXTRACTOR_MODEL_ID, _SYSTEM_PROMPT, chunk_text[:_MAX_INPUT_CHARS])
    clean = re.sub(r"```(?:json)?\n?|```", "", text).strip()
    try:
        data = json.loads(clean)
        entities = data.get("entities", [])
        return [
            e for e in entities
            if isinstance(e.get("text"), str) and isinstance(e.get("entity_type"), str)
        ]
    except (json.JSONDecodeError, AttributeError):
        return []
