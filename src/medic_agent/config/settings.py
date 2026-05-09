import os

from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

if not ANTHROPIC_API_KEY:
    raise EnvironmentError(
        "ANTHROPIC_API_KEY is not set. "
        "Copy .env.example to .env and add your key."
    )

# --- Model Registry ---

AVAILABLE_MODELS: dict[str, str] = {
    "Claude Haiku (Fast)": "claude-haiku-4-5-20251001",
    "Claude Sonnet (Balanced)": "claude-sonnet-4-6",
    "Claude Opus (Powerful)": "claude-opus-4-6",
}

DEFAULT_MODEL_NAME = "Claude Haiku (Fast)"
DEFAULT_MODEL_ID = AVAILABLE_MODELS[DEFAULT_MODEL_NAME]

# --- System Prompt ---

DEFAULT_SYSTEM_PROMPT = (
    "You are a knowledgeable healthcare assistant. "
    "You help clinicians, administrators, and patients with healthcare-related questions. "
    "You provide accurate, evidence-based information. "
    "You always recommend consulting a licensed healthcare professional for medical decisions. "
    "You do not provide diagnoses."
)
