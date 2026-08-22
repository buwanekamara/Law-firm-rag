"""Multi-turn conversation support.

Nothing is stored. The client sends the previous turns with each request and
gets an answer back - so there is no database, no session table and no state on
the server to keep consistent. For a tool answering questions about a fixed
corpus that is the whole requirement, and it means two browser tabs cannot
tread on each other.

The hard part of multi-turn retrieval is not memory, it is that follow-ups stop
being searchable. "What about the other one?" contains no term worth embedding
and no keyword worth matching. So before retrieval runs, a dependent follow-up
is rewritten into a question that stands on its own, using the conversation as
context. Everything downstream then works exactly as it does for a single
question.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from app.config import settings
from app.llm import complete
from app.prompting import render

# Only the last few turns are carried. Older context stops helping and starts
# dragging retrieval towards whatever was discussed earliest.
MAX_TURNS = 6

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
    def from_payload(cls, payload: dict[str, Any]) -> "Turn":
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

    Most follow-ups are self-contained ("what about confidentiality?") and
    rewriting them wastes a call and a second of latency. Only questions that
    are very short, or that point at something with a pronoun, need it.
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
        # A failed rewrite must not fail the request; the original question is
        # a worse search but still a search.
        return question

    rewritten = rewritten.strip('"').strip()
    # Guard against a model that decides to answer rather than rewrite.
    if not rewritten or len(rewritten) > 400 or "\n" in rewritten:
        return question
    return rewritten
