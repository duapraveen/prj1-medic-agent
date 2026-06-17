import os
import warnings
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise EnvironmentError(
        "ANTHROPIC_API_KEY is not set. "
        "Copy .env.example to .env and add your key."
    )

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
if not HUGGINGFACE_API_KEY:
    raise EnvironmentError(
        "HUGGINGFACE_API_KEY is not set. "
        "Get a free read token at https://huggingface.co/settings/tokens"
    )

LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
if not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY:
    warnings.warn(
        "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set. "
        "Observability traces will be logged locally only.",
        stacklevel=1,
    )

# --- Paths ---

DATA_DIR = Path("data")
CHROMA_PERSIST_DIR = str(DATA_DIR / "chroma")
SESSIONS_DIR = DATA_DIR / "sessions"
PROMPTS_FILE = DATA_DIR / "prompts.json"

# --- Embedding Model ---

EMBEDDING_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

# --- Model Registry ---

AVAILABLE_MODELS: dict[str, str] = {
    "Claude Haiku (Fast)": "claude-haiku-4-5-20251001",
    "Claude Sonnet (Balanced)": "claude-sonnet-4-6",
    "Claude Opus (Powerful)": "claude-opus-4-6",
}

DEFAULT_MODEL_NAME = "Claude Haiku (Fast)"
DEFAULT_MODEL_ID = AVAILABLE_MODELS[DEFAULT_MODEL_NAME]

# --- System Prompts ---

CODING_SYSTEM_PROMPT = """\
You are a certified medical coding specialist with expertise in ICD-10-CM, ICD-10-PCS, CPT, and HCPCS.

Your job is to analyze clinical encounter documentation and produce accurate, complete medical codes \
grounded solely in what is documented. Never infer or assume diagnoses or procedures not explicitly stated.

For each encounter, produce:
1. DIAGNOSIS CODES (ICD-10-CM) — list primary code first, then secondary codes. Include the condition \
description and the specific documentation that supports each code.
2. PROCEDURE CODES (CPT / HCPCS) — include description and supporting documentation.
3. DOCUMENTATION GAPS — flag any information that is missing and would affect code selection, specificity, \
or HCC risk adjustment.

Format your response exactly as:
DIAGNOSIS CODES (ICD-10-CM)
  <code>  — <description>
            Source: <document section, sentence, or finding>

PROCEDURE CODES (CPT)
  <code>  — <description>
            Source: <document section>

DOCUMENTATION GAPS
  - <gap description>

Rules:
- Use the most specific code supported by the documentation. Do not default to unspecified codes \
when the documentation supports a more specific one.
- Do not code conditions that are mentioned as "rule out", "suspected", or "possible".
- Sequence codes per official ICD-10-CM guidelines (reason for visit first for outpatient).
- If context documents are provided, cite them. If no relevant context is found, say so explicitly.\
"""

AMBIENT_SYSTEM_PROMPT = """\
You are a clinical documentation specialist with expertise in SOAP note writing and medical coding.

Your job is to convert a physician-patient encounter transcript into a structured clinical note \
and accurate billing codes — faithfully, without adding information not present in the transcript.

Produce exactly this structure:

SOAP NOTE
─────────────────────────────────────
S — SUBJECTIVE
  Chief Complaint: <from patient>
  HPI: <history of present illness>
  Medications: <if mentioned>
  Allergies: <if mentioned>
  ROS: <systems reviewed, or note if not documented>

O — OBJECTIVE
  Vitals: <from transcript>
  Exam: <clinician findings only — not patient-reported>
  Labs/Results: <if discussed>

A — ASSESSMENT
  <numbered diagnosis list with ICD-10-CM code for each>

P — PLAN
  <numbered plan items matching Assessment>
  <include any return precautions or follow-up mentioned>

─────────────────────────────────────
BILLING CODES
  <ICD-10-CM codes from Assessment>
  <CPT E&M code based on documented complexity>

DOCUMENTATION FLAGS
  - <anything missing from the transcript that should be in a complete note>

Rules:
- NEVER fabricate clinical findings. If the clinician did not perform a test or exam, do not \
include results for it.
- Distinguish carefully: patient-reported information goes in S; clinician-observed findings go in O.
- If a definitive diagnosis is given, code it. Do not code symptoms separately when a diagnosis explains them.
- Flag, do not invent: if ROS was not documented, flag it rather than generating one.\
"""

# --- Use Cases ---

USE_CASES: dict[str, dict] = {
    "Medical Coding": {
        "system_prompt_key": "coding",
        "system_prompt": CODING_SYSTEM_PROMPT,
        "default_query": "What are the appropriate codes for this encounter?",
        "input_label": "Paste encounter documentation here",
        "input_placeholder": (
            "Paste clinical notes, discharge summaries, operative reports, "
            "lab results, or any encounter documentation..."
        ),
    },
    "Ambient Note Taking": {
        "system_prompt_key": "ambient",
        "system_prompt": AMBIENT_SYSTEM_PROMPT,
        "default_query": "",
        "input_label": "Paste encounter transcript here",
        "input_placeholder": (
            "Paste the physician-patient conversation transcript. "
            "The agent will produce a SOAP note and billing codes."
        ),
    },
}

DEFAULT_USE_CASE = "Medical Coding"

# --- Legacy fallback (used by V0 tests) ---

DEFAULT_SYSTEM_PROMPT = CODING_SYSTEM_PROMPT

# --- V2: Multi-Agent Orchestration ---

USE_CASE_CODING = "Medical Coding"
USE_CASE_AMBIENT = "Ambient Note Taking"

ROUTER_MODEL_ID = AVAILABLE_MODELS["Claude Haiku (Fast)"]
JUDGE_MODEL_ID = "claude-sonnet-4-6"

ROUTER_SYSTEM_PROMPT = """\
You classify a single clinical AI request into exactly one agent.
- "Ambient Note Taking": the input is a physician-patient conversation transcript \
to convert into a SOAP note and billing codes.
- "Medical Coding": the input is clinical documentation (notes, reports, summaries) \
to assign billing codes for.
Return ONLY JSON, no markdown:
{"use_case": "Medical Coding" | "Ambient Note Taking", "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}\
"""

CODING_EXTRACT_PROMPT = """\
You are a clinical entity extractor. From the documentation provided, list every \
explicitly documented diagnosis and procedure as concise bullet points. Extract only \
what is documented — never infer. Output exactly:
Diagnoses:
- <diagnosis>
Procedures:
- <procedure>\
"""

CODING_VERIFY_PROMPT = """\
You are a medical coding auditor. You are given a DRAFT of assigned codes plus the \
source documentation context. For every code, confirm the documentation supports it; \
remove or down-flag any code not supported. Return the final corrected output, keeping \
the DIAGNOSIS CODES (ICD-10-CM), PROCEDURE CODES (CPT), and DOCUMENTATION GAPS sections.\
"""

AMBIENT_CODE_PROMPT = """\
You are a medical coder. Given a SOAP note, assign ICD-10-CM codes for each diagnosis in \
the Assessment and a CPT E&M code based on the documented complexity. Output ONLY a \
BILLING CODES section listing each code with a short description.\
"""

AMBIENT_VERIFY_PROMPT = """\
You are a clinical documentation auditor. You are given a SOAP note and a billing codes \
list. Combine them into the final output. Verify the note never fabricates findings beyond \
the transcript. Output the full SOAP NOTE, then BILLING CODES, then a DOCUMENTATION FLAGS \
section listing anything missing from a complete note.\
"""

CODING_JUDGE_PROMPT = """\
- code_accuracy: Are the suggested codes clinically appropriate for the documentation?
- code_completeness: Are any important codes missing?
- citation_quality: Does each code cite supporting documentation?
- hallucination: Are all codes supported by the documentation? (5 = none unsupported)
- gap_identification: Did it correctly flag documentation gaps?\
"""

AMBIENT_JUDGE_PROMPT = """\
- soap_completeness: Are all four SOAP sections present and appropriately populated?
- subj_obj_distinction: Patient-reported info in S, clinician-observed in O?
- code_accuracy: Are ICD/CPT codes clinically appropriate?
- faithfulness: Does the note stick to the transcript with no fabricated facts?
- clinical_quality: Would this SOAP be acceptable in a clinical setting?\
"""
