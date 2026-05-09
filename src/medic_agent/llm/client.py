import os

from dotenv import load_dotenv
from litellm import completion
from litellm.exceptions import AuthenticationError, RateLimitError, APIConnectionError, BadRequestError

load_dotenv()


def ask(model_id: str, system_prompt: str, user_query: str) -> str:
    if not user_query.strip():
        raise ValueError("Query cannot be empty")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]

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
        model_id="claude-haiku-4-5-20251001",
        system_prompt="You are a knowledgeable healthcare assistant.",
        user_query="What is the difference between Type 1 and Type 2 diabetes?",
    )
    print(result)
