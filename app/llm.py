"""Phase 4 - the model call.

One thin wrapper over the OpenAI client pointed at the Vercel AI Gateway. The
gateway speaks the OpenAI protocol, so no vendor-specific code is needed and
the model is chosen by an environment variable - including a *different* model
for judging answers in the evaluation, so the answering model never grades its
own work.

Temperature is 0 from the first line of this file. A retrieval-augmented
answer should be reproducible: if the same question and the same excerpts
produce a different answer each run, no evaluation of it means anything.
"""

from __future__ import annotations

from functools import lru_cache

from openai import OpenAI

from app.config import settings

# Generous, because a slow gateway is a worse failure than a slow answer.
REQUEST_TIMEOUT_SECONDS = 90.0


class MissingApiKey(RuntimeError):
    """Raised with an actionable message rather than a bare 401 from the API."""


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    if not settings.ai_gateway_api_key:
        raise MissingApiKey(
            "AI_GATEWAY_API_KEY is not set. Copy .env.example to .env and paste the key in."
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
    """Model ids the gateway will accept, optionally filtered.

    Useful because gateway model slugs drift - the JUDGE_MODEL this project
    started with, "anthropic/claude-3-5-haiku", no longer exists - and a wrong
    one surfaces as an unhelpful 404 at the worst moment.
    """
    ids = sorted(model.id for model in get_client().models.list().data)
    if contains:
        ids = [model_id for model_id in ids if contains.lower() in model_id.lower()]
    return ids
