"""Prompt loading.

Prompts live as files rather than as string literals buried in the code, so
that a change to the wording is a visible diff, and so the report can show a
before-and-after between versions. PROMPT_VERSION picks which file is used.

Placeholders are double-braced ({{QUESTION}}) and filled by plain string
replacement rather than str.format, because the prompts contain JSON examples
and every { in them would otherwise have to be escaped.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from app.config import settings

PROMPTS_DIR = Path(__file__).parent / "prompts"

# The user's message is fenced so the model can tell a question from an
# instruction. A fence only works if the person on the other side cannot close
# it: a question that itself contains the fence characters would otherwise end
# it early, leaving whatever follows sitting exactly where instructions go.
# Runs of three or more quote characters are therefore replaced with visually
# similar characters that carry no structural meaning. The reader sees the same
# sentence; the parser sees no fence.
FENCE = "'''"
_FENCE_RUN = re.compile("([`'\"])\\1{2,}")
_LOOKALIKE = {"'": "\u2019", "`": "\u00b4", '"': "\u201d"}


def neutralise_fences(text: str) -> str:
    """Make user-supplied text unable to close the fence it is wrapped in."""
    return _FENCE_RUN.sub(lambda match: _LOOKALIKE[match.group(1)] * len(match.group(0)), text or "")


def fenced(text: str) -> str:
    return f"{FENCE}\n{neutralise_fences(text)}\n{FENCE}"

SYSTEM_MARKER = "# System"
USER_MARKER = "# User"


@lru_cache(maxsize=16)
def load_named_prompt(name: str) -> tuple[str, str]:
    """Return (system_prompt, user_template) for a prompt file stem."""
    path = PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        available = ", ".join(sorted(p.stem for p in PROMPTS_DIR.glob("*.md")))
        raise FileNotFoundError(f"No prompt file {path.name}. Available: {available or 'none'}")

    text = path.read_text(encoding="utf-8")
    if SYSTEM_MARKER not in text or USER_MARKER not in text:
        raise ValueError(f"{path.name} must contain '{SYSTEM_MARKER}' and '{USER_MARKER}' headings")

    _, remainder = text.split(SYSTEM_MARKER, 1)
    system, user = remainder.split(USER_MARKER, 1)
    return system.strip(), user.strip()


def load_prompt(version: str | None = None) -> tuple[str, str]:
    """The answering prompt for a version, defaulting to PROMPT_VERSION."""
    return load_named_prompt(f"answer_{version or settings.prompt_version}")


def render(template_name: str, **values: str) -> tuple[str, str]:
    """Load a prompt and fill its {{PLACEHOLDER}} slots."""
    system, user = load_named_prompt(template_name)
    for key, value in values.items():
        user = user.replace("{{" + key + "}}", value)
    return system, user


def render_user_prompt(
    question: str, excerpts: str, version: str | None = None, history: str = ""
) -> str:
    """Fill an answer template.

    Older prompt versions have no {{HISTORY}} slot; substituting into them is
    a no-op, so one call site serves every version.
    """
    _, template = load_prompt(version)
    # Excerpts are corpus text and are not fenced. The question and the history
    # come from whoever is using the system, so both are.
    return (
        template.replace("{{EXCERPTS}}", excerpts)
        .replace("{{QUESTION}}", fenced(question))
        .replace("{{HISTORY}}", neutralise_fences(history) or "(this is the first message)")
    )


def available_versions() -> list[str]:
    """Answer prompt versions on disk, newest naming aside."""
    return sorted(path.stem.removeprefix("answer_") for path in PROMPTS_DIR.glob("answer_*.md"))
