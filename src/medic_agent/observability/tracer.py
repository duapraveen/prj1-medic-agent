import json
import uuid
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from medic_agent.config.settings import (
    LANGFUSE_BASE_URL,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
    SESSIONS_DIR,
)

_LANGFUSE_ENABLED = bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)
_langfuse_client = None

if _LANGFUSE_ENABLED:
    try:
        from langfuse import Langfuse

        _langfuse_client = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_BASE_URL,
        )
    except Exception as e:
        warnings.warn(f"LangFuse client init failed: {e}. Local logging only.")
        _LANGFUSE_ENABLED = False


@dataclass
class Session:
    use_case: str
    model_id: str
    query: str
    response: str
    system_prompt_version: str = "default"
    chunks_retrieved: list[dict] = field(default_factory=list)
    latency_ms: float = 0.0
    token_usage: dict = field(default_factory=dict)
    error: str | None = None
    route_decision: dict = field(default_factory=dict)
    agent_steps: list[dict] = field(default_factory=list)
    judge_scores: dict = field(default_factory=dict)
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


def log_session(session: Session) -> None:
    _write_local(session)
    if _LANGFUSE_ENABLED and _langfuse_client:
        _send_to_langfuse(session)


def _write_local(session: Session) -> None:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = SESSIONS_DIR / "sessions.jsonl"
    with log_file.open("a") as f:
        f.write(json.dumps(asdict(session)) + "\n")


def _send_to_langfuse(session: Session) -> None:
    try:
        from langfuse.types import TraceContext

        # LangFuse v4 uses OpenTelemetry trace IDs: 32 lowercase hex chars (no dashes)
        ctx = TraceContext(trace_id=session.session_id.replace("-", ""))
        metadata = {
            "use_case": session.use_case,
            "system_prompt_version": session.system_prompt_version,
            "chunks_retrieved_count": len(session.chunks_retrieved),
            "latency_ms": session.latency_ms,
            "route_decision": session.route_decision,
            "judge_scores": session.judge_scores,
        }

        with _langfuse_client.start_as_current_observation(
            trace_context=ctx,
            name="medic-agent-query",
            as_type="span",
            input=session.query,
            output=session.response,
            metadata=metadata,
        ):
            if session.agent_steps:
                _emit_agent_spans(session)
            else:
                _emit_legacy_spans(session)

        _langfuse_client.flush()
    except Exception as e:
        warnings.warn(f"LangFuse trace failed: {e}. Session saved locally.")


def _emit_agent_spans(session: Session) -> None:
    steps = session.agent_steps
    agent_steps = [s for s in steps if s["name"] not in ("router", "judge")]
    has_router = any(s["name"] == "router" for s in steps)
    has_judge = any(s["name"] == "judge" for s in steps)

    if has_router:
        with _langfuse_client.start_as_current_observation(
            name="router",
            as_type="span",
            input=session.query,
            output=session.route_decision,
        ):
            pass

    with _langfuse_client.start_as_current_observation(
        name=f"agent:{session.use_case}",
        as_type="span",
        input=session.query,
        output=session.response,
    ):
        for s in agent_steps:
            as_type = "retriever" if s.get("type") == "retriever" else "generation"
            with _langfuse_client.start_as_current_observation(
                name=s["name"],
                as_type=as_type,
                model=s.get("model_id") or None,
                output=s.get("output_summary"),
            ):
                pass

    if has_judge:
        with _langfuse_client.start_as_current_observation(
            name="judge",
            as_type="generation",
            output=session.judge_scores,
        ):
            pass
        overall = session.judge_scores.get("overall")
        if overall is not None:
            try:
                _langfuse_client.score_current_trace(
                    name="judge_overall", value=float(overall)
                )
            except Exception:
                pass


def _emit_legacy_spans(session: Session) -> None:
    with _langfuse_client.start_as_current_observation(
        name="retrieval",
        as_type="retriever",
        input={"query": session.query},
        output={"chunks": session.chunks_retrieved},
    ):
        pass

    with _langfuse_client.start_as_current_observation(
        name="llm-completion",
        as_type="generation",
        model=session.model_id,
        input=session.query,
        output=session.response,
        usage_details={
            "input": session.token_usage.get("prompt_tokens", 0),
            "output": session.token_usage.get("completion_tokens", 0),
            "total": session.token_usage.get("total_tokens", 0),
        },
    ):
        pass
