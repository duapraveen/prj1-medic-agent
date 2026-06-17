import time

from litellm import completion
from litellm.exceptions import AuthenticationError, BadRequestError, APIConnectionError, RateLimitError

from medic_agent.config.settings import DEFAULT_MODEL_ID, DEFAULT_SYSTEM_PROMPT
from medic_agent.observability.tracer import Session, log_session


def _format_context(chunks: list[dict]) -> str:
    parts = []
    for chunk in chunks:
        header = f"[Source: {chunk['source_filename']} | Chunk {chunk['chunk_index']}]"
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n".join(parts)


def _build_messages(
    system_prompt: str, user_query: str, context: list[dict] | None = None
) -> list[dict]:
    # cache_control marks content for Anthropic prompt caching (5-min ephemeral cache).
    # LiteLLM passes it through to Anthropic and ignores it for other providers.
    # Minimum cacheable tokens: 2048 (Haiku), 1024 (Sonnet/Opus).
    system_content = [
        {
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},
        }
    ]

    if context:
        system_content.append(
            {
                "type": "text",
                "text": "RETRIEVED DOCUMENT CONTEXT\n\n" + _format_context(context),
                "cache_control": {"type": "ephemeral"},
            }
        )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_query},
    ]


def complete(
    model_id: str,
    system_prompt: str,
    user_query: str,
    context: list[dict] | None = None,
) -> tuple[str, dict]:
    messages = _build_messages(system_prompt, user_query, context)
    response = completion(model=model_id, messages=messages)
    text = response.choices[0].message.content
    usage: dict = {}
    try:
        u = response.usage
        usage = {
            "prompt_tokens": u.prompt_tokens,
            "completion_tokens": u.completion_tokens,
            "total_tokens": u.total_tokens,
        }
    except Exception:
        pass
    return text, usage


def ask(
    model_id: str,
    system_prompt: str,
    user_query: str,
    context: list[dict] | None = None,
    session: Session | None = None,
) -> str:
    if not user_query.strip():
        raise ValueError("Query cannot be empty")

    messages = _build_messages(system_prompt, user_query, context)
    start = time.monotonic()
    result: str | None = None
    error_msg: str | None = None

    try:
        response = completion(model=model_id, messages=messages)
        result = response.choices[0].message.content
        if session is not None:
            session.response = result
            try:
                usage = response.usage
                session.token_usage = {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                }
            except Exception:
                pass
    except AuthenticationError:
        error_msg = "Invalid API key"
        raise RuntimeError("Invalid API key. Check your .env file.")
    except RateLimitError:
        error_msg = "Rate limit hit"
        raise RuntimeError("Rate limit hit. Wait a moment and try again.")
    except APIConnectionError:
        error_msg = "Connection error"
        raise RuntimeError("Could not connect to the API. Check your internet connection.")
    except BadRequestError as e:
        error_msg = str(e)
        raise RuntimeError(f"API rejected the request: {e}")
    finally:
        if session is not None:
            session.latency_ms = round((time.monotonic() - start) * 1000, 1)
            session.error = error_msg
            log_session(session)

    return result


if __name__ == "__main__":
    result = ask(
        model_id=DEFAULT_MODEL_ID,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        user_query="What is the difference between Type 1 and Type 2 diabetes?",
    )
    print(result)
