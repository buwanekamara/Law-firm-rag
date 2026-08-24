"""Prompt loading.

Prompts are files, not string literals, so a wording change is a visible diff.
PROMPT_VERSION picks the file.

{{PLACEHOLDER}} slots are filled by str.replace, not str.format - the prompts
contain JSON examples and every { would need escaping.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from app.config import settings

PROMPTS_DIR = Path(__file__).parent / "prompts"

# The user's message is fenced so the model can tell a question from an
# instruction. That only works if the question cannot close the fence itself,
# so runs of three or more quotes become lookalike characters: same sentence to
# the reader, no fence to the parser.
FENCE = "'''"
_FENCE_RUN = re.compile("([`'\"])\\1{2,}")
_LOOKALIKE = {"'": "\u2019", "`": "\u00b4", '"': "\u201d"}


def neutralise_fences(text: str) -> str:
    """Stop user text closing the fence it is wrapped in."""
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

    Older versions have no {{HISTORY}} slot; substituting is a no-op there.
    """
    _, template = load_prompt(version)
    # Excerpts are corpus text. Question and history come from the user, so
    # both are fenced.
    return (
        template.replace("{{EXCERPTS}}", excerpts)
        .replace("{{QUESTION}}", fenced(question))
        .replace("{{HISTORY}}", neutralise_fences(history) or "(this is the first message)")
    )


def available_versions() -> list[str]:
    """Answer prompt versions on disk."""
    return sorted(path.stem.removeprefix("answer_") for path in PROMPTS_DIR.glob("answer_*.md"))
