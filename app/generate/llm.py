"""The model call.

A thin wrapper over the OpenAI client pointed at the Vercel AI Gateway. The
gateway speaks the OpenAI protocol, so the model is just an environment
variable - including a different one for judging answers in eval.
"""

from __future__ import annotations

from functools import lru_cache

from openai import OpenAI

from app.config import settings

REQUEST_TIMEOUT_SECONDS = settings.llm_timeout_seconds


class MissingApiKey(RuntimeError):
    """Raised instead of letting a bare 401 come back from the API."""


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    if not settings.ai_gateway_api_key:
        raise MissingApiKey(
            "AI_GATEWAY_API_KEY is not set. Paste your gateway key into .env."
        )
    return OpenAI(
        api_key=settings.ai_gateway_api_key,
        base_url=settings.ai_gateway_base_url,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


def complete(
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
    temperature: float | None = None,
) -> str:
    """Send one prompt, return the raw text of the reply."""
    client = get_client()
    response = client.chat.completions.create(
        model=model or settings.llm_model,
        temperature=settings.llm_temperature if temperature is None else temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return (response.choices[0].message.content or "").strip()


def list_models(contains: str | None = None) -> list[str]:
    """Model ids the gateway accepts, optionally filtered.

    Gateway slugs change over time, and a wrong one surfaces as a 404.
    """
    ids = sorted(model.id for model in get_client().models.list().data)
    if contains:
        ids = [model_id for model_id in ids if contains.lower() in model_id.lower()]
    return ids
