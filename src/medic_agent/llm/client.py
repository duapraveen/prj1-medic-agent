from litellm import completion
from litellm.exceptions import AuthenticationError, BadRequestError, APIConnectionError, RateLimitError

from medic_agent.config.settings import DEFAULT_MODEL_ID, DEFAULT_SYSTEM_PROMPT


def _build_messages(system_prompt: str, user_query: str) -> list[dict]:
    # cache_control marks content for Anthropic prompt caching (5-min ephemeral cache).
    # LiteLLM passes it through to Anthropic and ignores it for other providers.
    # Minimum cacheable tokens: 2048 (Haiku), 1024 (Sonnet/Opus).
    # V1: add a third content block here for retrieved document context (also marked for caching).
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {"role": "user", "content": user_query},
    ]


def ask(model_id: str, system_prompt: str, user_query: str) -> str:
    if not user_query.strip():
        raise ValueError("Query cannot be empty")

    messages = _build_messages(system_prompt, user_query)

    try:
        response = completion(model=model_id, messages=messages)
        return response.choices[0].message.content
    except AuthenticationError:
        raise RuntimeError("Invalid API key. Check your .env file.")
    except RateLimitError:
        raise RuntimeError("Rate limit hit. Wait a moment and try again.")
    except APIConnectionError:
        raise RuntimeError("Could not connect to the API. Check your internet connection.")
    except BadRequestError as e:
        raise RuntimeError(f"API rejected the request: {e}")


if __name__ == "__main__":
    result = ask(
        model_id=DEFAULT_MODEL_ID,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        user_query="What is the difference between Type 1 and Type 2 diabetes?",
    )
    print(result)
