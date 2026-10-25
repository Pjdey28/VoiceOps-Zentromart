
import logging
import re
import time
from functools import lru_cache

from groq import Groq, RateLimitError

from src.config import GROQ_API_KEY, GROQ_API_KEY_2, GROQ_MODEL, validate_llm_config


logger = logging.getLogger(__name__)
MAX_RATE_LIMIT_RETRIES = 4


@lru_cache(maxsize=1)
def get_llm_clients() -> tuple[Groq, ...]:
    """Create reusable Groq clients in primary-to-fallback order."""
    validate_llm_config()
    return tuple(
        Groq(api_key=api_key)
        for api_key in (GROQ_API_KEY, GROQ_API_KEY_2)
        if api_key
    )


def _rate_limit_delay(error: RateLimitError, attempt: int) -> float:
    """Return a bounded delay using the provider's retry hint when available."""
    retry_after = getattr(error, "response", None)
    headers = getattr(retry_after, "headers", {}) or {}
    retry_after_value = headers.get("retry-after")

    if retry_after_value:
        try:
            return max(0.0, min(float(retry_after_value), 60.0))
        except ValueError:
            pass

    match = re.search(r"try again in ([\d.]+)(ms|s)", str(error), re.IGNORECASE)
    if match:
        value = float(match.group(1))
        if match.group(2).lower() == "ms":
            value /= 1000
        return min(value, 60.0)

    return min(2 ** (attempt - 1), 60.0)


def generate_response(
    messages: list[dict[str, str]],
    temperature: float = 0.0,
) -> str:
    """Generate a text response from the configured LLM."""
    clients = get_llm_clients()
    response = None

    for client_index, client in enumerate(clients, start=1):
        for attempt in range(1, MAX_RATE_LIMIT_RETRIES + 2):
            try:
                response = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=messages,
                    temperature=temperature,
                )
                break
            except RateLimitError as error:
                if attempt > MAX_RATE_LIMIT_RETRIES:
                    if client_index < len(clients):
                        logger.warning(
                            "Groq key %d exhausted its rate limit; "
                            "switching to fallback key %d.",
                            client_index,
                            client_index + 1,
                        )
                        break
                    raise
                delay = _rate_limit_delay(error, attempt)
                logger.warning(
                    "Groq key %d rate limit reached; retrying in %.2f seconds "
                    "(attempt %d/%d).",
                    client_index,
                    delay,
                    attempt,
                    MAX_RATE_LIMIT_RETRIES,
                )
                time.sleep(delay)

        if response is not None:
            break

    if response is None:
        raise RuntimeError("All configured Groq API keys failed to generate a response.")

    if not response.choices:
        raise RuntimeError("The LLM returned no response choices.")

    content = response.choices[0].message.content

    if not content or not content.strip():
        raise RuntimeError("The LLM returned an empty response.")

    return content.strip()
