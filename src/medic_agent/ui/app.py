import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from medic_agent.config.settings import (
    AMBIENT_SYSTEM_PROMPT,
    AVAILABLE_MODELS,
    CODING_SYSTEM_PROMPT,
    DEFAULT_MODEL_NAME,
    DEFAULT_USE_CASE,
    LANGFUSE_BASE_URL,
    LANGFUSE_PUBLIC_KEY,
    PROMPTS_FILE,
    SESSIONS_DIR,
    USE_CASES,
)
from medic_agent.llm.client import ask
from medic_agent.observability.tracer import Session
from medic_agent.rag.embedder import embed_texts
from medic_agent.rag.ingestor import load_pdf, load_text
from medic_agent.rag.retriever import retrieve
from medic_agent.rag.store import (
    add_document,
    delete_document,
    document_exists,
    get_document_info,
)

GOLDEN_CASES_PATH = Path("tests/eval/golden_cases.json")
BASELINE_PATH = Path("tests/eval/baseline.json")


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def _load_prompts() -> dict:
    if PROMPTS_FILE.exists():
        return json.loads(PROMPTS_FILE.read_text())
    return {"version": 0, "coding": CODING_SYSTEM_PROMPT, "ambient": AMBIENT_SYSTEM_PROMPT}


def _save_prompts(coding: str, ambient: str) -> int:
    existing = _load_prompts()
    version = existing.get("version", 0) + 1
    PROMPTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROMPTS_FILE.write_text(
        json.dumps(
            {
                "version": version,
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "coding": coding,
                "ambient": ambient,
            },
            indent=2,
        )
    )
    _push_prompts_to_langfuse(coding, ambient, version)
    return version


def _push_prompts_to_langfuse(coding: str, ambient: str, version: int) -> None:
    try:
        from medic_agent.observability.tracer import _LANGFUSE_ENABLED, _langfuse_client

        if not _LANGFUSE_ENABLED or _langfuse_client is None:
            return
        _langfuse_client.create_prompt(
            name="medic-coding-prompt", prompt=coding, labels=["production"]
        )
        _langfuse_client.create_prompt(
            name="medic-ambient-prompt", prompt=ambient, labels=["production"]
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar() -> tuple[str, str]:
    """Returns (use_case_name, model_id)."""
    st.sidebar.title("medic-agent")

    use_case = st.sidebar.radio(
        "Use Case",
        options=list(USE_CASES.keys()),
        index=list(USE_CASES.keys()).index(DEFAULT_USE_CASE),
    )

    st.sidebar.divider()
    docs = get_document_info()
    total_chunks = sum(d["chunk_count"] for d in docs)
    st.sidebar.caption(
        f"📄 {len(docs)} document{'s' if len(docs) != 1 else ''} ({total_chunks} chunks)"
    )
    st.sidebar.caption("→ Manage in **Knowledge Base** tab")

    model_name = st.sidebar.selectbox(
        "Model",
        options=list(AVAILABLE_MODELS.keys()),
        index=list(AVAILABLE_MODELS.keys()).index(DEFAULT_MODEL_NAME),
    )
    return use_case, AVAILABLE_MODELS[model_name]


# ---------------------------------------------------------------------------
# Tab 1 — Agent
# ---------------------------------------------------------------------------

def _render_agent_tab(use_case: str, model_id: str) -> None:
    uc = USE_CASES[use_case]
    prompts = _load_prompts()
    system_prompt = prompts.get(uc["system_prompt_key"], uc["system_prompt"])
    prompt_version = f"v{prompts['version']}" if prompts.get("version") else "default"

    st.caption(f"System prompt: **{prompt_version}**")

    user_input = st.text_area(
        uc["input_label"],
        value=uc["default_query"],
        placeholder=uc["input_placeholder"],
        height=160,
    )

    if st.button("Submit", type="primary", disabled=not user_input.strip()):
        session = Session(
            use_case=use_case,
            model_id=model_id,
            query=user_input,
            response="",
            system_prompt_version=prompt_version,
        )
        with st.spinner("Retrieving context…"):
            chunks = retrieve(user_input, k=5)
            session.chunks_retrieved = chunks

        with st.spinner("Generating response…"):
            try:
                response = ask(
                    model_id=model_id,
                    system_prompt=system_prompt,
                    user_query=user_input,
                    context=chunks,
                    session=session,
                )
                st.session_state["last_response"] = response
                st.session_state["last_chunks"] = chunks
            except RuntimeError as e:
                st.error(str(e))
                return

    if "last_response" in st.session_state:
        st.divider()
        st.markdown(st.session_state["last_response"])
        chunks = st.session_state.get("last_chunks", [])
        if chunks:
            with st.expander(f"📎 Sources used ({len(chunks)} chunks)"):
                for c in chunks:
                    st.caption(f"**{c['source_filename']}** — chunk {c['chunk_index']}")
                    st.text(c["text"][:300] + ("…" if len(c["text"]) > 300 else ""))


# ---------------------------------------------------------------------------
# Tab 2 — Knowledge Base & System Prompts
# ---------------------------------------------------------------------------

def _render_kb_upload_section() -> None:
    st.subheader("Knowledge Base")
    uploaded = st.file_uploader("Upload document", type=["pdf", "txt"])

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
            st.success(f"✅ **{doc_id}** ingested — {len(chunks)} chunks added.")
            st.rerun()

    docs = get_document_info()
    if docs:
        st.caption(
            f"Total: {len(docs)} documents, {sum(d['chunk_count'] for d in docs)} chunks"
        )
        for doc in docs:
            col1, col2, col3, col4 = st.columns([4, 1, 2, 1])
            col1.write(doc["filename"])
            col2.write(str(doc["chunk_count"]))
            col3.caption(doc["upload_date"])
            if col4.button("🗑", key=f"del_{doc['filename']}"):
                delete_document(doc["filename"])
                st.rerun()
    else:
        st.info("No documents uploaded yet. Upload a PDF or TXT file above.")


def _render_kb_prompts_section() -> None:
    st.divider()
    st.subheader("System Prompts")
    prompts = _load_prompts()
    version = prompts.get("version", 0)
    saved_at = prompts.get("saved_at", "")
    if version:
        st.caption(
            f"Active: **v{version}**" + (f" — saved {saved_at[:10]}" if saved_at else "")
        )

    coding_text = st.text_area(
        "Medical Coding Prompt",
        value=prompts.get("coding", CODING_SYSTEM_PROMPT),
        height=280,
        key="coding_prompt_area",
    )
    ambient_text = st.text_area(
        "Ambient Note Taking Prompt",
        value=prompts.get("ambient", AMBIENT_SYSTEM_PROMPT),
        height=280,
        key="ambient_prompt_area",
    )

    col1, col2 = st.columns(2)
    if col1.button("💾 Save Prompts", type="primary"):
        new_version = _save_prompts(coding_text, ambient_text)
        st.success(f"Prompts saved as **v{new_version}**.")
        st.rerun()
    if col2.button("↩ Reset to Defaults"):
        st.session_state["coding_prompt_area"] = CODING_SYSTEM_PROMPT
        st.session_state["ambient_prompt_area"] = AMBIENT_SYSTEM_PROMPT
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
    total_tokens = sum(s.get("token_usage", {}).get("total_tokens", 0) for s in sessions)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sessions", total)
    c2.metric("Avg latency", f"{avg_latency:.0f} ms")
    c3.metric("Total tokens", f"{total_tokens:,}")
    c4.metric("Errors", errors)
    st.divider()

    use_cases = ["All"] + sorted({s.get("use_case", "") for s in sessions})
    filter_uc = st.selectbox("Filter by use case", use_cases)
    filtered = (
        sessions if filter_uc == "All" else [s for s in sessions if s.get("use_case") == filter_uc]
    )

    for s in reversed(filtered):
        ts = s.get("timestamp", "")[:16].replace("T", " ")
        label = f"{ts}  |  {s.get('use_case','')}  |  {s.get('model_id','')}"
        with st.expander(label):
            st.markdown(f"**Query:** {s.get('query','')}")
            st.markdown(f"**Response:** {s.get('response','')[:500]}…")
            st.caption(
                f"Latency: {s.get('latency_ms',0):.0f} ms  |  "
                f"Tokens: {s.get('token_usage',{}).get('total_tokens','?')}  |  "
                f"Prompt: {s.get('system_prompt_version','?')}"
            )

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
    col1.download_button("📤 Export CSV", buf.getvalue(), "sessions.csv", "text/csv")
    if LANGFUSE_PUBLIC_KEY:
        col2.link_button("🔗 Open in LangFuse", LANGFUSE_BASE_URL)


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
            "input", value=result.case_input, height=220,
            disabled=True, label_visibility="collapsed", key=f"dlg_input_{result.case_id}"
        )
        st.markdown("**Model Response**")
        st.text_area(
            "response", value=result.response, height=280,
            disabled=True, label_visibility="collapsed", key=f"dlg_resp_{result.case_id}"
        )

    with tab_chunks:
        if result.chunks:
            st.caption(f"{len(result.chunks)} chunk(s) retrieved from ChromaDB and injected as context.")
            for c in result.chunks:
                with st.expander(f"**{c['source_filename']}** — chunk {c['chunk_index']}"):
                    st.text(c["text"])
        else:
            st.info("No chunks retrieved — knowledge base was empty at eval time.")

    with tab_ragas:
        if result.ragas_scores and "faithfulness" in result.ragas_scores:
            score = result.ragas_scores["faithfulness"]
            st.metric("Faithfulness", f"{score:.3f}", help="0 = fully hallucinated · 1 = fully grounded")
            st.markdown(
                "**How RAGAS computes this score:**\n"
                "1. Decomposes the model response into atomic factual claims\n"
                "2. For each claim, runs NLI to check whether it can be inferred from the retrieved context\n"
                "3. Faithfulness = claims supported ÷ total claims\n\n"
                "A low score here does **not** mean the response is wrong — it means the model drew on "
                "clinical knowledge beyond what was in the retrieved chunks. For a coding agent, this is "
                "expected: codes like E11.65 require medical reasoning, not just text extraction."
            )
            st.divider()
            st.markdown("**Context windows sent to RAGAS:**")
            ctx = [c["text"] for c in result.chunks] if result.chunks else []
            for i, text in enumerate(ctx):
                with st.expander(f"Context {i + 1} ({len(text)} chars)"):
                    st.text(text)
            st.markdown("**Response evaluated:**")
            st.text_area(
                "ragas_resp", value=result.response[:1500], height=160,
                disabled=True, label_visibility="collapsed", key=f"dlg_rresp_{result.case_id}"
            )
        else:
            st.info("Layer 2 (RAGAS) was not run for this case.")

    with tab_judge:
        if result.judge_prompt:
            st.markdown("**Prompt sent to judge model** (`claude-sonnet-4-6`):")
            st.code(result.judge_prompt, language=None)
        if result.judge_raw:
            st.markdown("**Raw judge response (JSON):**")
            st.code(result.judge_raw, language="json")
        if result.judge_scores:
            st.markdown("**Parsed scores:**")
            criteria = {k: v for k, v in result.judge_scores.items() if k != "overall"}
            for criterion, score in criteria.items():
                try:
                    s = float(score)
                    bar = "█" * round(s) + "░" * (5 - round(s))
                    st.caption(f"`{criterion}`: {s:.1f} {bar} — {_judge_label(s)}")
                except (TypeError, ValueError):
                    st.caption(f"`{criterion}`: {score}")
            if "overall" in result.judge_scores:
                overall = result.judge_scores["overall"]
                st.markdown(f"**Overall: {overall:.1f}/5 — {_judge_label(overall)}**")
        if not result.judge_prompt and not result.judge_raw:
            st.info("Layer 3 (LLM-as-Judge) was not run for this case.")


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
    st.subheader("Results")
    st.caption("Judge scores use a 1–5 scale: 1 = Poor · 2 = Fair · 3 = Acceptable · 4 = Good · 5 = Excellent")
    baseline = json.loads(BASELINE_PATH.read_text()) if BASELINE_PATH.exists() else {}

    for r in results:
        l1 = "✅ pass" if r.layer1_pass else "❌ fail"

        judge_summary = "—"
        if r.judge_scores and "overall" in r.judge_scores:
            overall = r.judge_scores["overall"]
            judge_summary = f"{overall:.1f}/5 — {_judge_label(overall)}"

        ragas_summary = "—"
        if r.ragas_scores:
            if "error" in r.ragas_scores:
                ragas_summary = f"error: {r.ragas_scores['error'][:40]}"
            elif "faithfulness" in r.ragas_scores:
                ragas_summary = f"{r.ragas_scores['faithfulness']:.2f} (0–1 grounding score)"

        delta_str = ""
        if baseline and r.case_id in baseline:
            diff = (r.judge_scores or {}).get("overall", 0) - baseline[r.case_id].get("judge_overall", 0)
            delta_str = f"  |  Δ {'+' if diff > 0 else ''}{diff:.1f} vs baseline"

        label = f"**{r.case_id}**  |  L1: {l1}  |  Faithfulness: {ragas_summary}  |  Judge: {judge_summary}{delta_str}"
        with st.expander(label):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**RAGAS Faithfulness**")
                if r.ragas_scores and "faithfulness" in r.ragas_scores:
                    v = r.ragas_scores["faithfulness"]
                    st.markdown(
                        f"`{v:.2f}` — measures whether every claim in the response "
                        f"is supported by the retrieved context (0 = hallucinated, 1 = fully grounded)"
                    )
                else:
                    st.caption(ragas_summary if ragas_summary != "—" else "Not run")
            with col2:
                st.markdown("**LLM Judge Scores (1–5)**")
                if r.judge_scores:
                    criteria = {k: v for k, v in r.judge_scores.items() if k != "overall"}
                    for criterion, score in criteria.items():
                        try:
                            s = float(score)
                            bar = "█" * round(s) + "░" * (5 - round(s))
                            st.caption(f"`{criterion}`: {s:.1f} {bar} — {_judge_label(s)}")
                        except (TypeError, ValueError):
                            st.caption(f"`{criterion}`: {score}")
                    if "overall" in r.judge_scores:
                        overall = r.judge_scores["overall"]
                        st.markdown(f"**Overall: {overall:.1f}/5 — {_judge_label(overall)}**")
                else:
                    st.caption("Not run")
            has_detail = r.response or r.chunks or r.judge_prompt or r.judge_raw
            if has_detail:
                st.divider()
                if st.button("🔍 Inspect — input, chunks, reasoning", key=f"inspect_{r.case_id}"):
                    _evidence_dialog(r)


def _render_eval_tab() -> None:
    cases = _load_golden_cases()
    if not cases:
        st.warning(f"Golden cases not found at `{GOLDEN_CASES_PATH}`.")
        return

    st.subheader("Golden Dataset")
    selected = {
        case["id"]: st.checkbox(
            f"**{case['id']}** — {case['description']}", value=True, key=f"case_{case['id']}"
        )
        for case in cases
    }

    st.divider()
    st.subheader("Layers to Run")
    run_l1 = st.checkbox("Layer 1: Deterministic (~2 s, free)", value=True)
    run_l2 = st.checkbox("Layer 2: RAGAS (~3 min, small LLM cost)", value=False)
    run_l3 = st.checkbox("Layer 3: LLM-as-Judge (~5 min, ~$0.05)", value=False)
    if run_l2 or run_l3:
        st.warning("⚠️ Layer 2/3 make real LLM calls and incur API cost.")

    chosen_ids = [cid for cid, checked in selected.items() if checked]
    layers = [l for l, run in [(1, run_l1), (2, run_l2), (3, run_l3)] if run]

    if st.button("▶ Run Evaluation", type="primary", disabled=not (chosen_ids and layers)):
        try:
            from medic_agent.evaluation.runner import EvalRunner

            chosen_cases = [c for c in cases if c["id"] in chosen_ids]
            with st.spinner("Running evaluation…"):
                results = EvalRunner().run_all(chosen_cases, layers)
            st.session_state["eval_results"] = results
            st.session_state["eval_layers"] = layers
        except ImportError:
            st.info("EvalRunner not yet implemented (coming in Step 1.10).")

    if "eval_results" in st.session_state:
        _render_eval_results(st.session_state["eval_results"])

    st.divider()
    if BASELINE_PATH.exists():
        st.caption(f"Baseline: `{BASELINE_PATH}`")

    if st.button("📌 Set as Baseline"):
        results = st.session_state.get("eval_results")
        if not results:
            st.warning("Run evaluation first, then set as baseline.")
        else:
            baseline = {
                r.case_id: {
                    "layer1_pass": r.layer1_pass,
                    "judge_overall": (r.judge_scores or {}).get("overall", 0),
                    "ragas_faithfulness": (r.ragas_scores or {}).get("faithfulness", 0),
                    "layers_run": st.session_state.get("eval_layers", []),
                    "timestamp": r.timestamp,
                }
                for r in results
            }
            BASELINE_PATH.write_text(json.dumps(baseline, indent=2))
            st.success(f"✅ Baseline saved to `{BASELINE_PATH}` ({len(results)} cases).")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="medic-agent", page_icon="🏥", layout="wide")
    use_case, model_id = _render_sidebar()
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🤖 Agent", "📚 Knowledge Base & Prompts", "📊 Observability", "🧪 Evaluation"]
    )
    with tab1:
        _render_agent_tab(use_case, model_id)
    with tab2:
        _render_kb_upload_section()
        _render_kb_prompts_section()
    with tab3:
        _render_observability_tab()
    with tab4:
        _render_eval_tab()


if __name__ == "__main__":
    main()
