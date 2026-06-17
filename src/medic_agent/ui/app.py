import csv
import io
import json
from pathlib import Path

import streamlit as st

from medic_agent.agents.orchestrator import run as orchestrate
from medic_agent.config.prompts import PROMPT_DEFAULTS, load_all, save_all
from medic_agent.config.settings import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL_NAME,
    LANGFUSE_BASE_URL,
    LANGFUSE_PUBLIC_KEY,
    SESSIONS_DIR,
    USE_CASE_AMBIENT,
    USE_CASE_CODING,
)
from medic_agent.rag.embedder import embed_texts
from medic_agent.rag.ingestor import load_pdf, load_text
from medic_agent.rag.store import (
    add_document,
    delete_document,
    document_exists,
    get_document_info,
)

GOLDEN_CASES_PATH = Path("tests/eval/golden_cases.json")
BASELINE_PATH = Path("tests/eval/baseline.json")

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

_STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg:      #060d1a;
    --surf:    #0c1726;
    --elev:    #112038;
    --bdr:     #1c2e48;
    --acc:     #00c8b4;
    --acc-dim: rgba(0,200,180,.14);
    --acc-glo: rgba(0,200,180,.07);
    --t1: #d8e8f8;
    --t2: #6888aa;
    --t3: #334f70;
    --green: #10b981;
    --red:   #ef4444;
    --amber: #f59e0b;
    --ff-d: 'Syne', sans-serif;
    --ff-b: 'DM Sans', sans-serif;
    --ff-m: 'JetBrains Mono', monospace;
}

/* ── App ── */
.stApp {
    background: var(--bg) !important;
    color: var(--t1) !important;
    font-family: var(--ff-b) !important;
}
.main .block-container {
    padding-top: 1.75rem !important;
    max-width: 1100px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surf) !important;
    border-right: 1px solid var(--bdr) !important;
}
[data-testid="stSidebar"] * {
    font-family: var(--ff-b) !important;
}
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] small {
    color: var(--t2) !important;
    font-family: var(--ff-m) !important;
    font-size: .7rem !important;
}
[data-testid="stSidebar"] hr {
    border-color: var(--bdr) !important;
    margin: .75rem 0 !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio label {
    color: var(--t2) !important;
    font-size: .7rem !important;
    letter-spacing: .07em !important;
    text-transform: uppercase !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    color: var(--t1) !important;
    font-size: .88rem !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--bdr) !important;
    gap: 0 !important;
    padding: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--t2) !important;
    font-family: var(--ff-b) !important;
    font-size: .72rem !important;
    font-weight: 500 !important;
    letter-spacing: .07em !important;
    text-transform: uppercase !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
    padding: .65rem 1.1rem !important;
    transition: color .15s, background .15s !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--t1) !important;
    background: var(--acc-glo) !important;
}
.stTabs [aria-selected="true"] {
    color: var(--acc) !important;
    border-bottom-color: var(--acc) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 1.5rem !important;
}

/* ── Headings ── */
h1, h2, h3 {
    font-family: var(--ff-d) !important;
    color: var(--t1) !important;
    letter-spacing: -.02em !important;
}
h2 { font-size: 1.1rem !important; font-weight: 700 !important; }
h3 { font-size: .95rem !important; font-weight: 600 !important; }

/* ── Buttons ── */
.stButton > button {
    font-family: var(--ff-b) !important;
    font-size: .72rem !important;
    font-weight: 600 !important;
    letter-spacing: .07em !important;
    text-transform: uppercase !important;
    border-radius: 5px !important;
    transition: all .18s ease !important;
}
.stButton > button[kind="primary"] {
    background: var(--acc) !important;
    color: #040c18 !important;
    border: none !important;
}
.stButton > button[kind="primary"]:hover {
    background: #00dcc6 !important;
    box-shadow: 0 0 22px rgba(0,200,180,.38) !important;
}
.stButton > button[kind="secondary"] {
    background: transparent !important;
    border: 1px solid var(--bdr) !important;
    color: var(--t2) !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color: var(--acc) !important;
    color: var(--acc) !important;
}
.stButton > button:disabled { opacity: .38 !important; }

/* ── Inputs ── */
.stTextArea textarea,
.stTextInput input {
    background: var(--elev) !important;
    border: 1px solid var(--bdr) !important;
    border-radius: 6px !important;
    color: var(--t1) !important;
    font-family: var(--ff-b) !important;
    font-size: .9rem !important;
    caret-color: var(--acc) !important;
    transition: border-color .18s, box-shadow .18s !important;
}
.stTextArea textarea:focus,
.stTextInput input:focus {
    border-color: var(--acc) !important;
    box-shadow: 0 0 0 2px var(--acc-dim) !important;
    outline: none !important;
}
.stTextArea label,
.stTextInput label {
    color: var(--t2) !important;
    font-family: var(--ff-m) !important;
    font-size: .68rem !important;
    letter-spacing: .07em !important;
    text-transform: uppercase !important;
}

/* ── Select ── */
.stSelectbox [data-baseweb="select"] > div {
    background: var(--elev) !important;
    border-color: var(--bdr) !important;
    color: var(--t1) !important;
    font-family: var(--ff-b) !important;
}
.stSelectbox label {
    color: var(--t2) !important;
    font-family: var(--ff-m) !important;
    font-size: .68rem !important;
    letter-spacing: .07em !important;
    text-transform: uppercase !important;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: var(--elev) !important;
    border: 1px solid var(--bdr) !important;
    border-radius: 8px !important;
    padding: .9rem 1.1rem !important;
}
[data-testid="metric-container"] label {
    color: var(--t2) !important;
    font-family: var(--ff-m) !important;
    font-size: .66rem !important;
    letter-spacing: .1em !important;
    text-transform: uppercase !important;
}
[data-testid="metric-container"] [data-testid="metric-value"] {
    color: var(--t1) !important;
    font-family: var(--ff-d) !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
}

/* ── Expanders ── */
.stExpander {
    background: var(--surf) !important;
    border: 1px solid var(--bdr) !important;
    border-radius: 6px !important;
    margin-bottom: .4rem !important;
}
.stExpander summary {
    color: var(--t1) !important;
    font-family: var(--ff-b) !important;
    font-size: .88rem !important;
}
.stExpander summary:hover { background: var(--acc-glo) !important; }
[data-testid="stExpanderDetails"] {
    border-top: 1px solid var(--bdr) !important;
    padding-top: .85rem !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--elev) !important;
    border: 1px dashed var(--bdr) !important;
    border-radius: 8px !important;
    transition: border-color .18s !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--acc) !important; }
[data-testid="stFileUploader"] * { color: var(--t2) !important; }
[data-testid="stFileUploader"] button { color: var(--acc) !important; }

/* ── Dividers ── */
hr { border-color: var(--bdr) !important; margin: 1.25rem 0 !important; }

/* ── Captions ── */
.stCaption, small {
    color: var(--t2) !important;
    font-family: var(--ff-m) !important;
    font-size: .7rem !important;
}

/* ── Markdown text ── */
.stMarkdown p { color: var(--t1) !important; line-height: 1.7 !important; }
.stMarkdown h1,.stMarkdown h2,.stMarkdown h3 {
    font-family: var(--ff-d) !important;
    color: var(--t1) !important;
}

/* ── Code ── */
code {
    font-family: var(--ff-m) !important;
    background: var(--elev) !important;
    color: var(--acc) !important;
    padding: .1em .35em !important;
    border-radius: 3px !important;
    font-size: .82em !important;
}
pre { background: var(--elev) !important; border: 1px solid var(--bdr) !important; border-radius: 6px !important; padding: .85rem !important; }
pre code { background: transparent !important; padding: 0 !important; color: var(--t1) !important; }
.stCodeBlock { border: 1px solid var(--bdr) !important; border-radius: 6px !important; }

/* ── Checkboxes ── */
.stCheckbox label {
    color: var(--t1) !important;
    font-size: .88rem !important;
    font-family: var(--ff-b) !important;
}

/* ── Alerts ── */
[data-baseweb="notification"] {
    background: var(--elev) !important;
    border-radius: 6px !important;
    border: none !important;
}
.stAlert > div { border-radius: 6px !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--acc) !important; }

/* ── Download button ── */
.stDownloadButton > button {
    font-family: var(--ff-b) !important;
    font-size: .72rem !important;
    font-weight: 600 !important;
    letter-spacing: .07em !important;
    text-transform: uppercase !important;
}

/* ── Dialog ── */
[data-testid="stDialog"] > div > div {
    background: var(--surf) !important;
    border: 1px solid var(--bdr) !important;
    border-radius: 12px !important;
}

/* ── Scrollbars ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--bdr); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--t3); }
</style>
"""


def _section_label(text: str) -> None:
    st.markdown(
        f"""<div style="font-family:'JetBrains Mono',monospace;font-size:.66rem;"""
        f"""color:#334f70;letter-spacing:.12em;text-transform:uppercase;"""
        f"""margin-bottom:.6rem">{text}</div>""",
        unsafe_allow_html=True,
    )


def _section_header(title: str, mono_tag: str = "") -> None:
    tag_html = (
        f"""<span style="font-family:'JetBrains Mono',monospace;font-size:.65rem;"""
        f"""color:#334f70;letter-spacing:.1em;text-transform:uppercase;"""
        f"""margin-left:.6rem;vertical-align:middle">{mono_tag}</span>"""
        if mono_tag
        else ""
    )
    st.markdown(
        f"""<div style="font-family:'Syne',sans-serif;font-size:1.05rem;"""
        f"""font-weight:700;color:#d8e8f8;letter-spacing:-.02em;"""
        f"""margin-bottom:1rem">{title}{tag_html}</div>""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _render_sidebar() -> tuple[str, str, bool]:
    st.sidebar.markdown(
        """
<div style="padding:1.4rem 0 1.8rem">
  <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:800;
              color:#d8e8f8;letter-spacing:-.04em;line-height:1">
    medic<span style="color:#00c8b4">·</span>agent
  </div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:.6rem;color:#334f70;
              letter-spacing:.18em;text-transform:uppercase;margin-top:6px">
    Clinical AI &nbsp;·&nbsp; v2
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    routing_mode = st.sidebar.selectbox(
        "Routing Mode",
        options=["Auto", USE_CASE_CODING, USE_CASE_AMBIENT],
    )
    judge_on = st.sidebar.toggle("Score every response (judge)", value=True)

    st.sidebar.divider()

    docs = get_document_info()
    total_chunks = sum(d["chunk_count"] for d in docs)
    st.sidebar.caption(
        f"{'📄 ' if docs else ''}"
        f"{len(docs)} doc{'s' if len(docs) != 1 else ''} · {total_chunks} chunks"
    )
    st.sidebar.caption("→ Manage in **Knowledge Base** tab")

    model_name = st.sidebar.selectbox(
        "Model",
        options=list(AVAILABLE_MODELS.keys()),
        index=list(AVAILABLE_MODELS.keys()).index(DEFAULT_MODEL_NAME),
    )
    return routing_mode, AVAILABLE_MODELS[model_name], judge_on


# ---------------------------------------------------------------------------
# Tab 1 — Agent
# ---------------------------------------------------------------------------


def _render_routing_panel(route: dict) -> None:
    if not route:
        return
    use_case = route.get("use_case", "?")
    method = route.get("method", "?")
    confidence = route.get("confidence", 0)
    reasoning = route.get("reasoning", "")
    st.markdown(
        f"""<div style="background:#0c1726;border:1px solid #1c2e48;border-radius:8px;
padding:.85rem 1.1rem;margin-bottom:1rem">
  <div style="font-family:'JetBrains Mono',monospace;font-size:.6rem;color:#334f70;
              letter-spacing:.12em;text-transform:uppercase;margin-bottom:.35rem">
    Routed to</div>
  <div style="font-family:'Syne',sans-serif;font-size:1.05rem;font-weight:700;
              color:#00c8b4">{use_case}</div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:.7rem;color:#6888aa;
              margin-top:.3rem">
    {method} · confidence {confidence} · {reasoning}</div>
</div>""",
        unsafe_allow_html=True,
    )


def _render_agent_tab(routing_mode: str, model_id: str, judge_on: bool) -> None:
    st.markdown(
        """<div style="display:flex;align-items:center;gap:.5rem;margin-bottom:1.1rem">
  <div style="font-family:'Syne',sans-serif;font-size:1.35rem;font-weight:700;
              color:#d8e8f8;letter-spacing:-.02em">Clinical Agent</div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:.62rem;color:#334f70;
              background:#112038;border:1px solid #1c2e48;border-radius:3px;
              padding:.15em .45em;letter-spacing:.08em">orchestrated</div>
</div>""",
        unsafe_allow_html=True,
    )

    user_input = st.text_area(
        "Paste encounter documentation or a visit transcript",
        placeholder=(
            "Paste clinical documentation to code, OR a physician-patient "
            "conversation transcript to turn into a SOAP note. The orchestrator "
            "picks the right agent."
        ),
        height=200,
    )

    if st.button("Submit", type="primary", disabled=not user_input.strip()):
        with st.spinner("Routing & running agent…"):
            try:
                result = orchestrate(
                    user_input, model_id, override=routing_mode, judge_on=judge_on
                )
                st.session_state["last_result"] = result
            except RuntimeError as e:
                st.error(str(e))
                return

    result = st.session_state.get("last_result")
    if result:
        _render_routing_panel(result["route"])
        st.markdown(result["response"])

        chunks = result.get("chunks", [])
        if chunks:
            with st.expander(f"Sources used — {len(chunks)} chunk(s)"):
                for c in chunks:
                    st.caption(f"**{c['source_filename']}** — chunk {c['chunk_index']}")
                    st.text(c["text"][:300] + ("…" if len(c["text"]) > 300 else ""))

        scores = result.get("judge_scores") or {}
        if scores and "error" not in scores:
            with st.expander("LLM-as-judge scores"):
                for k, v in scores.items():
                    st.caption(f"`{k}`: {v}")


# ---------------------------------------------------------------------------
# Tab 2 — Knowledge Base & System Prompts
# ---------------------------------------------------------------------------


def _render_kb_upload_section() -> None:
    _section_header("Knowledge Base")

    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 0

    uploaded = st.file_uploader(
        "Upload document", type=["pdf", "txt"], key=f"uploader_{st.session_state['uploader_key']}"
    )

    if uploaded:
        doc_id = uploaded.name
        if document_exists(doc_id):
            st.warning(f"**{doc_id}** is already in the knowledge base.")
        else:
            with st.spinner(f"Ingesting {doc_id}…"):
                raw = uploaded.read()
                chunks = (
                    load_pdf(raw, doc_id)
                    if uploaded.type == "application/pdf"
                    else load_text(raw.decode("utf-8", errors="replace"), doc_id)
                )
                if not chunks:
                    st.error("Could not extract any text from the file.")
                    return
                embeddings = embed_texts([c["text"] for c in chunks])
                add_document(doc_id, chunks, embeddings)
            st.success(f"**{doc_id}** ingested — {len(chunks)} chunks added.")
            st.session_state["uploader_key"] += 1
            st.rerun()

    docs = get_document_info()
    if docs:
        total_chunks = sum(d["chunk_count"] for d in docs)
        st.markdown(
            f"""<div style="font-family:'JetBrains Mono',monospace;font-size:.68rem;
color:#6888aa;margin:.85rem 0 .5rem;letter-spacing:.05em">
  {len(docs)} document{'s' if len(docs) != 1 else ''} &nbsp;·&nbsp;
  <span style="color:#00c8b4">{total_chunks}</span> total chunks
</div>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            """<div style="display:grid;grid-template-columns:1fr 56px 120px 44px;
gap:0;border:1px solid #1c2e48;border-radius:6px;overflow:hidden">
  <div style="font-family:'JetBrains Mono',monospace;font-size:.62rem;color:#334f70;
              letter-spacing:.1em;text-transform:uppercase;padding:.45rem .75rem;
              background:#0c1726;border-bottom:1px solid #1c2e48">File</div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:.62rem;color:#334f70;
              letter-spacing:.1em;text-transform:uppercase;padding:.45rem .5rem;
              background:#0c1726;border-bottom:1px solid #1c2e48;text-align:right">
    Chunks</div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:.62rem;color:#334f70;
              letter-spacing:.1em;text-transform:uppercase;padding:.45rem .75rem;
              background:#0c1726;border-bottom:1px solid #1c2e48">Uploaded</div>
  <div style="background:#0c1726;border-bottom:1px solid #1c2e48"></div>
</div>""",
            unsafe_allow_html=True,
        )
        for i, doc in enumerate(docs):
            is_last = i == len(docs) - 1
            border = "" if is_last else "border-bottom:1px solid #1c2e48"
            col_file, col_chunks, col_date, col_del = st.columns([4, 1, 2, 1])
            col_file.markdown(
                f"""<div style="font-family:'DM Sans',sans-serif;font-size:.85rem;
color:#d8e8f8;padding:.3rem 0;{border}">{doc['filename']}</div>""",
                unsafe_allow_html=True,
            )
            col_chunks.markdown(
                f"""<div style="font-family:'JetBrains Mono',monospace;font-size:.8rem;
color:#00c8b4;padding:.3rem 0;text-align:right;{border}">{doc['chunk_count']}</div>""",
                unsafe_allow_html=True,
            )
            col_date.markdown(
                f"""<div style="font-family:'JetBrains Mono',monospace;font-size:.72rem;
color:#334f70;padding:.3rem 0;{border}">{doc['upload_date'][:10]}</div>""",
                unsafe_allow_html=True,
            )
            if col_del.button("✕", key=f"del_{doc['filename']}"):
                delete_document(doc["filename"])
                st.rerun()
    else:
        st.info("No documents uploaded yet. Upload a PDF or TXT file above.")


def _render_kb_prompts_section() -> None:
    st.divider()
    _section_header("System Prompts")

    prompts = load_all()
    version = prompts.get("version", 0)
    if version:
        st.markdown(
            f"""<div style="font-family:'JetBrains Mono',monospace;font-size:.68rem;
color:#6888aa;margin-bottom:.85rem">active: <span style="color:#00c8b4">v{version}</span></div>""",
            unsafe_allow_html=True,
        )

    edited = {"router": "", "coding": {}, "ambient": {}, "judge": {}}

    _section_label("Router")
    edited["router"] = st.text_area(
        "Router classifier prompt", value=prompts["router"], height=160, key="p_router"
    )

    _section_label("Medical Coding Agent")
    for step in ("extract", "code", "verify"):
        edited["coding"][step] = st.text_area(
            f"Coding · {step}", value=prompts["coding"][step], height=180, key=f"p_coding_{step}"
        )

    _section_label("Ambient Note Taking Agent")
    for step in ("soap", "code", "verify"):
        edited["ambient"][step] = st.text_area(
            f"Ambient · {step}", value=prompts["ambient"][step], height=180, key=f"p_ambient_{step}"
        )

    _section_label("Judge Rubrics")
    for rubric in ("coding", "ambient"):
        edited["judge"][rubric] = st.text_area(
            f"Judge · {rubric}", value=prompts["judge"][rubric], height=160, key=f"p_judge_{rubric}"
        )

    col1, col2 = st.columns(2)
    if col1.button("Save Prompts", type="primary"):
        new_version = save_all(edited)
        st.success(f"Prompts saved as **v{new_version}**.")
        st.rerun()
    if col2.button("Reset to Defaults"):
        st.session_state["p_router"] = PROMPT_DEFAULTS["router"]
        for step in ("extract", "code", "verify"):
            st.session_state[f"p_coding_{step}"] = PROMPT_DEFAULTS["coding"][step]
        for step in ("soap", "code", "verify"):
            st.session_state[f"p_ambient_{step}"] = PROMPT_DEFAULTS["ambient"][step]
        for rubric in ("coding", "ambient"):
            st.session_state[f"p_judge_{rubric}"] = PROMPT_DEFAULTS["judge"][rubric]
        st.info("Defaults restored. Click **Save Prompts** to persist.")


# ---------------------------------------------------------------------------
# Tab 3 — Observability
# ---------------------------------------------------------------------------


def _load_sessions() -> list[dict]:
    log_file = SESSIONS_DIR / "sessions.jsonl"
    if not log_file.exists():
        return []
    sessions = []
    for line in log_file.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                sessions.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return sessions


def _render_observability_tab() -> None:
    sessions = _load_sessions()
    if not sessions:
        st.info("No sessions logged yet. Run a query in the Agent tab.")
        return

    total = len(sessions)
    errors = sum(1 for s in sessions if s.get("error"))
    avg_latency = sum(s.get("latency_ms", 0) for s in sessions) / total
    total_tokens = sum(
        s.get("token_usage", {}).get("total_tokens", 0) for s in sessions
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sessions", total)
    c2.metric("Avg latency", f"{avg_latency:.0f} ms")
    c3.metric("Total tokens", f"{total_tokens:,}")
    c4.metric("Errors", errors)

    st.divider()

    use_cases = ["All"] + sorted({s.get("use_case", "") for s in sessions})
    filter_uc = st.selectbox("Filter by use case", use_cases)
    filtered = (
        sessions
        if filter_uc == "All"
        else [s for s in sessions if s.get("use_case") == filter_uc]
    )

    for s in reversed(filtered):
        ts = s.get("timestamp", "")[:16].replace("T", " ")
        label = (
            f"{ts}  ·  {s.get('use_case','')}  ·  {s.get('model_id','')}"
        )
        with st.expander(label):
            st.markdown(f"**Query:** {s.get('query','')}")
            st.markdown(f"**Response:** {s.get('response','')[:500]}…")
            st.caption(
                f"Latency: {s.get('latency_ms',0):.0f} ms  ·  "
                f"Tokens: {s.get('token_usage',{}).get('total_tokens','?')}  ·  "
                f"Prompt: {s.get('system_prompt_version','?')}"
            )
            route = s.get("route_decision", {})
            if route:
                st.caption(
                    f"Routed: {route.get('use_case','?')} "
                    f"({route.get('method','?')}, conf {route.get('confidence','?')})"
                )
            judge = s.get("judge_scores", {})
            if judge and "overall" in judge:
                st.caption(f"Judge overall: {judge['overall']}")

    st.divider()
    col1, col2 = st.columns(2)
    buf = io.StringIO()
    flat = [
        {
            "timestamp": s.get("timestamp", ""),
            "use_case": s.get("use_case", ""),
            "model_id": s.get("model_id", ""),
            "query": s.get("query", ""),
            "latency_ms": s.get("latency_ms", ""),
            "total_tokens": s.get("token_usage", {}).get("total_tokens", ""),
            "error": s.get("error", ""),
        }
        for s in filtered
    ]
    writer = csv.DictWriter(buf, fieldnames=list(flat[0].keys()))
    writer.writeheader()
    writer.writerows(flat)
    col1.download_button(
        "Export CSV", buf.getvalue(), "sessions.csv", "text/csv"
    )
    if LANGFUSE_PUBLIC_KEY:
        col2.link_button("Open in LangFuse", LANGFUSE_BASE_URL)


# ---------------------------------------------------------------------------
# Tab 4 — Evaluation
# ---------------------------------------------------------------------------


def _load_golden_cases() -> list[dict]:
    if not GOLDEN_CASES_PATH.exists():
        return []
    return json.loads(GOLDEN_CASES_PATH.read_text()).get("cases", [])


@st.dialog("Evidence & Reasoning", width="large")
def _evidence_dialog(result) -> None:
    st.caption(f"`{result.case_id}` — {result.case_description}")
    tab_io, tab_chunks, tab_ragas, tab_judge = st.tabs(
        ["Input & Output", "Retrieved Chunks", "RAGAS Evidence", "Judge Reasoning"]
    )

    with tab_io:
        st.markdown("**Clinical Input**")
        st.text_area(
            "input",
            value=result.case_input,
            height=220,
            disabled=True,
            label_visibility="collapsed",
            key=f"dlg_input_{result.case_id}",
        )
        st.markdown("**Model Response**")
        st.text_area(
            "response",
            value=result.response,
            height=280,
            disabled=True,
            label_visibility="collapsed",
            key=f"dlg_resp_{result.case_id}",
        )

    with tab_chunks:
        if result.chunks:
            st.caption(
                f"{len(result.chunks)} chunk(s) retrieved from ChromaDB "
                f"and injected as context."
            )
            for c in result.chunks:
                with st.expander(
                    f"**{c['source_filename']}** — chunk {c['chunk_index']}"
                ):
                    st.text(c["text"])
        else:
            st.info(
                "No chunks retrieved — knowledge base was empty at eval time."
            )

    with tab_ragas:
        if result.ragas_scores and "faithfulness" in result.ragas_scores:
            score = result.ragas_scores["faithfulness"]
            st.metric(
                "Faithfulness",
                f"{score:.3f}",
                help="0 = fully hallucinated · 1 = fully grounded",
            )
            st.markdown(
                "**How RAGAS computes this score:**\n"
                "1. Decomposes the model response into atomic factual claims\n"
                "2. For each claim, runs NLI to check whether it can be inferred "
                "from the retrieved context\n"
                "3. Faithfulness = claims supported ÷ total claims\n\n"
                "A low score here does **not** mean the response is wrong — it "
                "means the model drew on clinical knowledge beyond what was in the "
                "retrieved chunks. For a coding agent, this is expected: codes like "
                "E11.65 require medical reasoning, not just text extraction."
            )
            st.divider()
            st.markdown("**Context windows sent to RAGAS:**")
            ctx = [c["text"] for c in result.chunks] if result.chunks else []
            for i, text in enumerate(ctx):
                with st.expander(f"Context {i + 1} ({len(text)} chars)"):
                    st.text(text)
            st.markdown("**Response evaluated:**")
            st.text_area(
                "ragas_resp",
                value=result.response[:1500],
                height=160,
                disabled=True,
                label_visibility="collapsed",
                key=f"dlg_rresp_{result.case_id}",
            )
        else:
            st.info("Layer 2 (RAGAS) was not run for this case.")

    with tab_judge:
        if result.judge_prompt:
            st.markdown(
                "**Prompt sent to judge model** (`claude-sonnet-4-6`):"
            )
            st.code(result.judge_prompt, language=None)
        if result.judge_raw:
            st.markdown("**Raw judge response (JSON):**")
            st.code(result.judge_raw, language="json")
        if result.judge_scores:
            st.markdown("**Parsed scores:**")
            criteria = {
                k: v for k, v in result.judge_scores.items() if k != "overall"
            }
            for criterion, score in criteria.items():
                try:
                    s = float(score)
                    bar = "█" * round(s) + "░" * (5 - round(s))
                    st.caption(
                        f"`{criterion}`: {s:.1f} {bar} — {_judge_label(s)}"
                    )
                except (TypeError, ValueError):
                    st.caption(f"`{criterion}`: {score}")
            if "overall" in result.judge_scores:
                overall = result.judge_scores["overall"]
                st.markdown(
                    f"**Overall: {overall:.1f}/5 — {_judge_label(overall)}**"
                )
        if not result.judge_prompt and not result.judge_raw:
            st.info(
                "Layer 3 (LLM-as-Judge) was not run for this case."
            )


def _judge_label(score: float) -> str:
    if score >= 4.3:
        return "Excellent"
    if score >= 3.6:
        return "Good"
    if score >= 3.0:
        return "Acceptable"
    if score >= 2.0:
        return "Fair"
    return "Poor"


def _render_eval_results(results: list) -> None:
    if not results:
        return
    st.divider()
    _section_header("Results")
    st.caption(
        "Judge scores: 1 = Poor · 2 = Fair · 3 = Acceptable · 4 = Good · 5 = Excellent"
    )
    baseline = (
        json.loads(BASELINE_PATH.read_text()) if BASELINE_PATH.exists() else {}
    )

    for r in results:
        l1 = "✓ pass" if r.layer1_pass else "✗ fail"

        judge_summary = "—"
        if r.judge_scores and "overall" in r.judge_scores:
            overall = r.judge_scores["overall"]
            judge_summary = f"{overall:.1f}/5 — {_judge_label(overall)}"

        ragas_summary = "—"
        if r.ragas_scores:
            if "error" in r.ragas_scores:
                ragas_summary = f"error: {r.ragas_scores['error'][:40]}"
            elif "faithfulness" in r.ragas_scores:
                ragas_summary = (
                    f"{r.ragas_scores['faithfulness']:.2f}"
                )

        delta_str = ""
        if baseline and r.case_id in baseline:
            diff = (r.judge_scores or {}).get("overall", 0) - baseline[
                r.case_id
            ].get("judge_overall", 0)
            delta_str = (
                f"  ·  Δ {'+' if diff > 0 else ''}{diff:.1f} vs baseline"
            )

        label = (
            f"**{r.case_id}**  ·  L1: {l1}  ·  "
            f"Faithfulness: {ragas_summary}  ·  Judge: {judge_summary}{delta_str}"
        )
        with st.expander(label):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**RAGAS Faithfulness**")
                if r.ragas_scores and "faithfulness" in r.ragas_scores:
                    v = r.ragas_scores["faithfulness"]
                    st.markdown(
                        f"`{v:.2f}` — measures whether every claim in the response "
                        f"is supported by the retrieved context "
                        f"(0 = hallucinated, 1 = fully grounded)"
                    )
                else:
                    st.caption(
                        ragas_summary if ragas_summary != "—" else "Not run"
                    )
            with col2:
                st.markdown("**LLM Judge Scores (1–5)**")
                if r.judge_scores:
                    criteria = {
                        k: v
                        for k, v in r.judge_scores.items()
                        if k != "overall"
                    }
                    for criterion, score in criteria.items():
                        try:
                            s = float(score)
                            bar = "█" * round(s) + "░" * (5 - round(s))
                            st.caption(
                                f"`{criterion}`: {s:.1f} {bar} — "
                                f"{_judge_label(s)}"
                            )
                        except (TypeError, ValueError):
                            st.caption(f"`{criterion}`: {score}")
                    if "overall" in r.judge_scores:
                        overall = r.judge_scores["overall"]
                        st.markdown(
                            f"**Overall: {overall:.1f}/5 — "
                            f"{_judge_label(overall)}**"
                        )
                else:
                    st.caption("Not run")
            has_detail = (
                r.response or r.chunks or r.judge_prompt or r.judge_raw
            )
            if has_detail:
                st.divider()
                if st.button(
                    "Inspect — input, chunks, reasoning",
                    key=f"inspect_{r.case_id}",
                ):
                    _evidence_dialog(r)


def _render_eval_tab() -> None:
    cases = _load_golden_cases()
    if not cases:
        st.warning(f"Golden cases not found at `{GOLDEN_CASES_PATH}`.")
        return

    _section_header("Golden Dataset")
    selected = {
        case["id"]: st.checkbox(
            f"**{case['id']}** — {case['description']}",
            value=True,
            key=f"case_{case['id']}",
        )
        for case in cases
    }

    st.divider()
    _section_header("Layers to Run")
    run_l1 = st.checkbox("Layer 1: Deterministic (~2 s, free)", value=True)
    run_l2 = st.checkbox(
        "Layer 2: RAGAS (~3 min, small LLM cost)", value=False
    )
    run_l3 = st.checkbox(
        "Layer 3: LLM-as-Judge (~5 min, ~$0.05)", value=False
    )
    if run_l2 or run_l3:
        st.warning("Layer 2/3 make real LLM calls and incur API cost.")

    chosen_ids = [cid for cid, checked in selected.items() if checked]
    layers = [l for l, run in [(1, run_l1), (2, run_l2), (3, run_l3)] if run]

    if st.button(
        "Run Evaluation",
        type="primary",
        disabled=not (chosen_ids and layers),
    ):
        try:
            from medic_agent.evaluation.runner import EvalRunner

            chosen_cases = [c for c in cases if c["id"] in chosen_ids]
            with st.spinner("Running evaluation…"):
                results = EvalRunner().run_all(chosen_cases, layers)
            st.session_state["eval_results"] = results
            st.session_state["eval_layers"] = layers
        except ImportError:
            st.info(
                "EvalRunner not yet implemented (coming in Step 1.10)."
            )

    if "eval_results" in st.session_state:
        _render_eval_results(st.session_state["eval_results"])

    st.divider()
    if BASELINE_PATH.exists():
        st.caption(f"Baseline: `{BASELINE_PATH}`")

    if st.button("Set as Baseline"):
        results = st.session_state.get("eval_results")
        if not results:
            st.warning("Run evaluation first, then set as baseline.")
        else:
            baseline = {
                r.case_id: {
                    "layer1_pass": r.layer1_pass,
                    "judge_overall": (r.judge_scores or {}).get("overall", 0),
                    "ragas_faithfulness": (r.ragas_scores or {}).get(
                        "faithfulness", 0
                    ),
                    "layers_run": st.session_state.get("eval_layers", []),
                    "timestamp": r.timestamp,
                }
                for r in results
            }
            BASELINE_PATH.write_text(json.dumps(baseline, indent=2))
            st.success(
                f"Baseline saved to `{BASELINE_PATH}` ({len(results)} cases)."
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="medic-agent",
        page_icon="⬡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_STYLES, unsafe_allow_html=True)

    routing_mode, model_id, judge_on = _render_sidebar()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Agent", "Knowledge Base", "Observability", "Evaluation"]
    )
    with tab1:
        _render_agent_tab(routing_mode, model_id, judge_on)
    with tab2:
        _render_kb_upload_section()
        _render_kb_prompts_section()
    with tab3:
        _render_observability_tab()
    with tab4:
        _render_eval_tab()


if __name__ == "__main__":
    main()
