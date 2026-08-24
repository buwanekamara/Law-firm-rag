"""Multi-turn support.

Nothing is stored: the client posts its previous turns with each request, so
there is no session state on the server.

The hard part is not memory, it is that follow-ups stop being searchable -
"what about the other one?" has nothing worth embedding or matching. Dependent
follow-ups are rewritten into standalone questions before retrieval runs;
everything downstream then behaves as it does for a single question.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from app.config import settings
from app.generate.llm import complete
from app.generate.prompting import render

# Older context drags retrieval back towards whatever was discussed first.
MAX_TURNS = settings.history_turns

# Words that make a question depend on what came before.
_DEPENDENT = re.compile(
    r"\b(it|its|it's|that|this|those|these|they|them|their|there|he|she|his|her"
    r"|the same|the other|instead|also|too|as well)\b",
    re.IGNORECASE,
)

# A very short message is almost always a follow-up: "and the supplier?"
SHORT_ENOUGH_TO_BE_A_FOLLOW_UP = 7


@dataclass
class Turn:
    question: str
    answer: str

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Turn:
        return cls(
            question=str(payload.get("question", ""))[:2000],
            answer=str(payload.get("answer", ""))[:4000],
        )


def recent(history: Iterable[Turn] | None) -> list[Turn]:
    turns = [turn for turn in (history or []) if turn.question]
    return turns[-MAX_TURNS:]


def render_history(history: list[Turn]) -> str:
    if not history:
        return "(this is the first message)"
    return "\n\n".join(f"User: {turn.question}\nAssistant: {turn.answer}" for turn in history)


def depends_on_history(question: str, history: list[Turn]) -> bool:
    """Cheap test for whether rewriting is worth a model call.

    Most follow-ups stand on their own; only very short ones, or ones pointing
    at something with a pronoun, need the rewrite.
    """
    if not history:
        return False
    if len(question.split()) <= SHORT_ENOUGH_TO_BE_A_FOLLOW_UP:
        return True
    return bool(_DEPENDENT.search(question))


def condense(question: str, history: list[Turn]) -> str:
    """Rewrite a dependent follow-up into a standalone question."""
    if not depends_on_history(question, history):
        return question

    system, user = render(
        "condense_v1", HISTORY=render_history(history), QUESTION=question
    )
    try:
        rewritten = complete(system, user, temperature=settings.llm_temperature).strip()
    except Exception:
        # A failed rewrite is a worse search, not a failed request.
        return question

    rewritten = rewritten.strip('"').strip()
    # Guard against a model that decides to answer rather than rewrite.
    if not rewritten or len(rewritten) > 400 or "\n" in rewritten:
        return question
    return rewritten
